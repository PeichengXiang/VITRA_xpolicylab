#!/usr/bin/env bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# gpu_id is a comma-separated list of 4 or 8 unique GPU ids, or "all" for 0-7.
# Four GPUs keep batch_size=8 and total_batch_size=64 (gradient accumulation 2).
if [[ $# -ne 6 ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

if [[ ! "${bench_name}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ ! "${ckpt_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "bench_name and ckpt_name may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ "${env_cfg_type}" != "tianji_marvin_wuji" ]]; then
  echo "VITRA spark0 training supports env_cfg_type=tianji_marvin_wuji only." >&2
  exit 2
fi
if [[ "${action_type}" != "ee" ]]; then
  echo "VITRA is exposed to XPolicyLab as action_type=ee only." >&2
  exit 2
fi
if [[ ! "${seed}" =~ ^[1-9][0-9]*$ ]] || (( seed >= 4294967295 )); then
  echo "VITRA requires seed in the range 1..4294967294, got: ${seed}" >&2
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be exported in the shell before training." >&2
  exit 2
fi

if [[ "${gpu_id}" == "all" ]]; then
  gpu_id="0,1,2,3,4,5,6,7"
fi
IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
nproc_per_node=${#gpu_ids[@]}
if [[ "${nproc_per_node}" != "4" && "${nproc_per_node}" != "8" ]]; then
  echo "Training requires 4 or 8 visible GPUs; got: ${gpu_id}" >&2
  exit 2
fi
declare -A seen_gpu_ids=()
for current_gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${current_gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpu_ids[${current_gpu_id}]:-}" ]]; then
    echo "gpu_id must contain ${nproc_per_node} unique integer IDs; got: ${gpu_id}" >&2
    exit 2
  fi
  seen_gpu_ids[${current_gpu_id}]=1
done

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
VITRA_ROOT="${POLICY_DIR}/VITRA"
base_config="${VITRA_ROOT}/vitra/configs/spark0_finetune.json"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_root="${POLICY_DIR}/checkpoints/${ckpt_setting}"
run_config="${ckpt_root}/train_config.json"
resume_checkpoint=""
run_date="${VITRA_RUN_DATE:-$(date +%F)}"
data_representation="${VITRA_DATA_REPRESENTATION:-wuji20}"

if [[ "${data_representation}" != "wuji20" && "${data_representation}" != "mano45" && "${data_representation}" != "inspire12" ]]; then
  echo "VITRA_DATA_REPRESENTATION must be wuji20, mano45, or inspire12, got: ${data_representation}" >&2
  exit 2
fi

if [[ "${VITRA_RESUME:-0}" == "1" ]]; then
  resume_checkpoint=$(python3 - "${ckpt_root}" "${bench_name}" "${ckpt_name}" <<'PY'
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
bench_name = sys.argv[2]
ckpt_name = sys.argv[3]
pattern = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
checkpoint_dirs = sorted(root.rglob("*.ckpt")) if root.is_dir() else []
complete = []
problems = []
for path in checkpoint_dirs:
    match = pattern.fullmatch(path.name)
    if match is None or not path.is_dir():
        problems.append(f"unexpected checkpoint path: {path}")
        continue
    required = (path / "weights.pt", path / "optimizer.pt", path / "meta.json")
    if not all(item.is_file() and item.stat().st_size > 0 for item in required):
        problems.append(f"incomplete checkpoint: {path}")
        continue
    try:
        meta = json.loads(required[2].read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"invalid checkpoint metadata: {path}: {exc}")
        continue
    epoch = int(match.group(1))
    step = int(match.group(2))
    if meta.get("complete") is not True or meta.get("epoch") != epoch or meta.get("global_step") != step:
        problems.append(f"checkpoint metadata mismatch: {path}")
        continue
    if path.parent.name != "checkpoints" or path.parent.parent.parent.resolve() != root:
        problems.append(f"checkpoint is outside <run_dir>/checkpoints: {path}")
        continue
    complete.append((step, path.resolve()))
if problems:
    raise SystemExit(
        "Refusing unsafe resume; quarantine or repair these paths first:\n- "
        + "\n- ".join(problems)
    )
if not complete:
    raise SystemExit("VITRA_RESUME=1 but no complete checkpoint exists below " + str(root))
steps = sorted(step for step, _ in complete)
expected = list(range(10_000, max(steps) + 1, 10_000))
if steps != expected or max(steps) > 80_000:
    raise SystemExit(f"Refusing non-contiguous checkpoint steps: found={steps}, expected={expected}")
run_dirs = {path.parent.parent for _, path in complete}
if len(run_dirs) != 1:
    raise SystemExit(f"Refusing checkpoints spread across multiple run directories: {sorted(map(str, run_dirs))}")
run_dir = next(iter(run_dirs))
run_date = run_dir.name[:10]
try:
    parsed_run_date = dt.date.fromisoformat(run_date).isoformat()
except ValueError as exc:
    raise SystemExit(f"Checkpoint run directory has no valid ISO date prefix: {run_dir.name}") from exc
expected_run_name = f"{parsed_run_date}-{bench_name}-{ckpt_name}_TB64_B{os.environ.get('VITRA_PER_DEVICE_BATCH', '8')}_bf16True"
if run_dir.name != expected_run_name:
    raise SystemExit(
        f"Checkpoint run directory does not match this training configuration: "
        f"found={run_dir.name}, expected={expected_run_name}"
    )
print(max(complete, key=lambda item: item[0])[1])
PY
  )
  resume_run_dir=$(basename "$(dirname "$(dirname "${resume_checkpoint}")")")
  resume_run_date=${resume_run_dir:0:10}
  if [[ ! "${resume_run_date}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Resume checkpoint run directory has no ISO date prefix: ${resume_run_dir}" >&2
    exit 1
  fi
  if [[ -n "${VITRA_RUN_DATE:-}" && "${VITRA_RUN_DATE}" != "${resume_run_date}" ]]; then
    echo "VITRA_RUN_DATE=${VITRA_RUN_DATE} does not match resumed run date ${resume_run_date}." >&2
    exit 1
  fi
  run_date="${resume_run_date}"
  if [[ "$(basename "${resume_checkpoint}")" =~ ^epoch=[0-9]+-step=80000\.ckpt$ ]]; then
    echo "[VITRA] step 80000 is already complete; verifying the existing eight checkpoints."
    python3 "${POLICY_DIR}/verify_checkpoints.py" --checkpoint-root "${ckpt_root}"
    exit 0
  fi
elif [[ -d "${ckpt_root}" ]] && find "${ckpt_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "Output root already contains files: ${ckpt_root}. Set VITRA_RESUME=1 only for a validated continuation." >&2
  exit 1
fi

if [[ -n "${VITRA_DATA_ROOT:-}" ]]; then
  data_root="${VITRA_DATA_ROOT}"
elif [[ "${data_representation}" == "mano45" ]]; then
  data_root="${WORKSPACE_ROOT}/data_mano"
else
  data_root="${WORKSPACE_ROOT}/data"
fi
data_root="$(realpath -e "${data_root}")"
pretrain_path="${WORKSPACE_ROOT}/pretrain_model/VITRA-VLA-3B/vitra-vla-3b.pt"
paligemma_root="${WORKSPACE_ROOT}/pretrain_model/paligemma2-3b-mix-224-local"
if [[ ! -d "${data_root}/TeleData" ]] || [[ ! -s "${data_root}/teledata_statistics.json" ]]; then
  echo "Prepared TeleData/statistics are missing under ${data_root}; run process_data.sh first." >&2
  exit 1
fi
if [[ "${data_representation}" == "inspire12" ]]; then
  if [[ ! -s "${data_root}/SHA256SUMS" ]]; then
    echo "EgoVLA checksum manifest is missing: ${data_root}/SHA256SUMS" >&2
    exit 1
  fi
  sha256sum -c "${data_root}/SHA256SUMS"
fi
python3 - "${data_root}" "${data_representation}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_root = Path(sys.argv[1]).resolve()
expected = sys.argv[2]
expected_egovla_contract = "egovla_observed_ee_step_future_hand_command_v3"
manifest_path = data_root / "spark0_manifest.json"
statistics_path = data_root / "teledata_statistics.json"
if not manifest_path.is_file():
    raise SystemExit(f"Prepared-data manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
statistics = json.loads(statistics_path.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest_representation = manifest.get("representation", "wuji20")
statistics_representation = statistics.get("representation", "wuji20")
if isinstance(statistics_representation, str) and "Wuji20" in statistics_representation:
    statistics_representation = "wuji20"
if manifest_representation != expected:
    raise SystemExit(
        f"Prepared-data representation mismatch: {manifest_representation!r} != {expected!r}"
    )
if statistics_representation != expected:
    raise SystemExit(
        f"Statistics representation mismatch: {statistics_representation!r} != {expected!r}"
    )
if expected == "inspire12":
    manifest_contract = manifest.get("action_contract_id")
    statistics_contract = statistics.get("action_contract_id")
    if manifest_contract != expected_egovla_contract or statistics_contract != expected_egovla_contract:
        raise SystemExit(
            "EgoVLA action contract mismatch: "
            f"manifest={manifest_contract!r}, statistics={statistics_contract!r}, "
            f"expected={expected_egovla_contract!r}"
        )
    completion_path = data_root / "CONVERSION_COMPLETE.json"
    if not completion_path.is_file():
        raise SystemExit(f"EgoVLA conversion marker is missing: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_completion = {
        "action_contract_id": expected_egovla_contract,
        "manifest_sha256": sha256(manifest_path),
        "statistics_sha256": sha256(statistics_path),
        "checksums_sha256": sha256(data_root / "SHA256SUMS"),
    }
    actual_completion = {key: completion.get(key) for key in expected_completion}
    if actual_completion != expected_completion:
        raise SystemExit(
            f"EgoVLA conversion marker mismatch: {actual_completion!r} "
            f"!= {expected_completion!r}"
        )
expected_dims = {"mano45": (61, 51), "inspire12": (18, 18)}.get(expected, (26, 26))
actual_dims = (
    statistics.get("state_dimension_per_hand", expected_dims[0]),
    statistics.get("action_dimension_per_hand", expected_dims[1]),
)
if actual_dims != expected_dims:
    raise SystemExit(f"Statistics dimension mismatch: {actual_dims} != {expected_dims}")
if manifest.get("episode_count", 0) <= 0 or manifest.get("frame_count", 0) <= 0:
    raise SystemExit(f"Prepared-data manifest is empty: {manifest_path}")
print(
    f"[VITRA] data contract representation={expected} "
    f"episodes={manifest['episode_count']} samples={manifest['frame_count']} "
    f"state/action per hand={expected_dims[0]}/{expected_dims[1]} "
    f"action_contract={manifest.get('action_contract_id')}"
)
PY
if [[ ! -s "${pretrain_path}" ]]; then
  echo "Pretrained VITRA checkpoint is missing: ${pretrain_path}" >&2
  exit 1
fi
for required_file in config.json preprocessor_config.json tokenizer.json tokenizer_config.json; do
  if [[ ! -s "${paligemma_root}/${required_file}" ]]; then
    echo "Local PaliGemma processor/config file is missing: ${paligemma_root}/${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${ckpt_root}"
config_args=(
  --base-config "${base_config}" \
  --output-config "${run_config}" \
  --output-root "${ckpt_root}" \
  --workspace-root "${WORKSPACE_ROOT}" \
  --data-root "${data_root}" \
  --representation "${data_representation}" \
  --bench-name "${bench_name}" \
  --ckpt-name "${ckpt_name}" \
  --run-date "${run_date}" \
  --seed "${seed}"
)
if [[ -n "${resume_checkpoint}" ]]; then
  config_args+=(--resume-checkpoint "${resume_checkpoint}")
  echo "[VITRA] resuming from ${resume_checkpoint}"
fi
python3 "${POLICY_DIR}/make_train_config.py" "${config_args[@]}"

python3 - "${run_config}" "${data_representation}" "${data_root}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
expected_representation = sys.argv[2]
expected_data_root = str(Path(sys.argv[3]).resolve())
config = json.loads(config_path.read_text(encoding="utf-8"))
actual_representation = config["train_dataset"].get("representation")
actual_data_root = str(Path(config["train_dataset"]["data_root_dir"]).resolve())
if actual_representation != expected_representation or actual_data_root != expected_data_root:
    raise SystemExit(
        "Refusing to train with a mismatched dataset config: "
        f"representation={actual_representation!r}, data_root={actual_data_root!r}"
    )
print(
    f"[VITRA] guarded dataset representation={actual_representation} "
    f"data_root={actual_data_root}"
)
PY

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${VITRA_ROOT}"
echo "[VITRA] launching torchrun nproc_per_node=${nproc_per_node} CUDA_VISIBLE_DEVICES=${gpu_id}"
torchrun --standalone --nproc_per_node="${nproc_per_node}" \
  scripts/train.py \
  --config "${run_config}"

python3 "${POLICY_DIR}/verify_checkpoints.py" \
  --checkpoint-root "${ckpt_root}"
