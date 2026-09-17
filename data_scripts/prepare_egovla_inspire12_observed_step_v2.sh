#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <raw-egovla-root> [output-data-root]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="$(realpath -e "$1")"
DATA_ROOT="${2:-${REPO_ROOT}/data/egovla_inspire12_observed_step_v2}"
MAPPING="${REPO_ROOT}/XPolicyLab/policy/VITRA/mapping_inspire12_mano45.json"

if [[ -e "${DATA_ROOT}" ]]; then
  echo "Refusing to overwrite existing output: ${DATA_ROOT}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/prepare_egovla_inspire12_vitra.py" \
  --source-root "${SOURCE_ROOT}" \
  --data-root "${DATA_ROOT}"

python "${SCRIPT_DIR}/calculate_vitra_statistics.py" \
  --data-root "${DATA_ROOT}" \
  --xpolicylab-root "${REPO_ROOT}/XPolicyLab" \
  --mapping "${MAPPING}" \
  --representation inspire12 \
  --batch-size "${VITRA_STATS_BATCH_SIZE:-256}" \
  --num-workers "${VITRA_STATS_WORKERS:-8}"

python - "${SOURCE_ROOT}" "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
data = Path(sys.argv[2]).resolve()
manifest = json.loads((data / "spark0_manifest.json").read_text())
stats = json.loads((data / "teledata_statistics.json").read_text())
assert manifest["converter"] == "egovla_inspire12_observed_step_v2"
assert manifest["source_root"] == str(source)
assert manifest["representation"] == "inspire12"
assert manifest["episode_count"] == 1903
assert manifest["frame_count"] == 510546
assert stats["representation"] == "inspire12"
assert stats["state_dimension_per_hand"] == 18
assert stats["action_dimension_per_hand"] == 18
assert stats["num_episodes"] == 1903
assert stats["num_samples"] == 510546
assert stats["valid_action_samples_left"] == 508643
assert stats["valid_action_samples_right"] == 508643
links = list((data / "TeleData").glob("*/episode_*.hdf5"))
assert len(links) == 1903
assert all(path.is_symlink() and path.exists() for path in links)
print("EgoVLA observed-step-v2 data contract OK")
print("episodes=1903 frames=510546 valid_actions_per_side=508643")
PY

sha256sum \
  "${DATA_ROOT}/spark0_manifest.json" \
  "${DATA_ROOT}/teledata_statistics.json" \
  "${MAPPING}" \
  "${REPO_ROOT}/XPolicyLab/policy/VITRA/VITRA/vitra/datasets/egovla_inspire_dataset.py" \
  > "${DATA_ROOT}/SHA256SUMS"

echo "Prepared corrected EgoVLA data at ${DATA_ROOT}"
