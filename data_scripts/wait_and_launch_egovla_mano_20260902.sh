#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/personal/xiangpc/0813_Xpolicylab_bench/VITRA"
DATA_ROOT="${MODEL_ROOT}/data/egovla_humanoid_sim_mano_20260902"
LAUNCH="${MODEL_ROOT}/data_scripts/launch_egovla_mano_train_20260902.sh"

echo "[VITRA] waiting for statistics: ${DATA_ROOT}/teledata_statistics.json"
while [[ ! -s "${DATA_ROOT}/teledata_statistics.json" ]]; do
  sleep 30
done

# Do a final contract check immediately before touching the GPUs.
python3 - "${DATA_ROOT}/teledata_statistics.json" "${DATA_ROOT}/spark0_manifest.json" <<'PY'
import json
import sys
stats = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
if stats.get("representation") != "mano45":
    raise SystemExit(f"statistics representation is {stats.get('representation')!r}")
if (stats.get("state_dimension_per_hand"), stats.get("action_dimension_per_hand")) != (61, 51):
    raise SystemExit("statistics dimensions are not MANO45 61/51")
if manifest.get("representation") != "mano45" or manifest.get("episode_count", 0) != 1903:
    raise SystemExit("prepared-data manifest does not match the intended MANO45 full source")
excluded = manifest.get("deprecated_exclusion", {}).get("count")
if excluded != 100:
    raise SystemExit(f"Deprecated exclusion count is {excluded!r}, expected 100")
print("[VITRA] final data contract check passed: mano45, 1903 episodes, 100 Deprecated excluded")
PY

exec "${LAUNCH}"
