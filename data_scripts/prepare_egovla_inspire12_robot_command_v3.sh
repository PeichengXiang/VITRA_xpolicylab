#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <raw-egovla-root> [output-data-root]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="$(realpath -e "$1")"
DATA_ROOT="${2:-${REPO_ROOT}/data/egovla_inspire12_robot_command_v3}"
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
expected_contract = "egovla_observed_ee_step_future_hand_command_v3"
assert manifest["converter"] == "egovla_inspire12_robot_command_v3"
assert manifest["source_root"] == str(source)
assert manifest["representation"] == "inspire12"
assert manifest["action_contract_id"] == expected_contract
assert manifest["episode_count"] == 1903
assert manifest["frame_count"] == 510546
assert stats["representation"] == "inspire12"
assert stats["action_contract_id"] == expected_contract
assert stats["state_dimension_per_hand"] == 18
assert stats["action_dimension_per_hand"] == 18
assert stats["num_episodes"] == 1903
assert stats["num_samples"] == 510546
assert stats["valid_action_samples_left"] == 508643
assert stats["valid_action_samples_right"] == 508643
links = list((data / "TeleData").glob("*/episode_*.hdf5"))
assert len(links) == 1903
assert all(path.is_symlink() and path.exists() for path in links)
print("EgoVLA robot-command-v3 data contract OK")
print("episodes=1903 frames=510546 valid_actions_per_side=508643")
PY

CHECKSUM_TMP="${DATA_ROOT}/SHA256SUMS.tmp"
sha256sum \
  "${DATA_ROOT}/spark0_manifest.json" \
  "${DATA_ROOT}/teledata_statistics.json" \
  "${MAPPING}" \
  "${SCRIPT_DIR}/prepare_egovla_inspire12_vitra.py" \
  "${SCRIPT_DIR}/calculate_vitra_statistics.py" \
  "${SCRIPT_DIR}/prepare_egovla_inspire12_robot_command_v3.sh" \
  "${REPO_ROOT}/XPolicyLab/policy/VITRA/VITRA/vitra/datasets/egovla_inspire_dataset.py" \
  > "${CHECKSUM_TMP}"
mv "${CHECKSUM_TMP}" "${DATA_ROOT}/SHA256SUMS"
sha256sum -c "${DATA_ROOT}/SHA256SUMS"

python - "${DATA_ROOT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

data = Path(sys.argv[1]).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

completion = {
    "schema_version": 1,
    "action_contract_id": "egovla_observed_ee_step_future_hand_command_v3",
    "manifest_sha256": sha256(data / "spark0_manifest.json"),
    "statistics_sha256": sha256(data / "teledata_statistics.json"),
    "checksums_sha256": sha256(data / "SHA256SUMS"),
}
target = data / "CONVERSION_COMPLETE.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(f"Published conversion marker: {target}")
PY

echo "Prepared EgoVLA robot-command-v3 data at ${DATA_ROOT}"
