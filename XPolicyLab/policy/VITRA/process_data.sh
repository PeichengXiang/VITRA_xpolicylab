#!/usr/bin/env bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
if [[ $# -lt 4 ]] || [[ $# -gt 5 ]]; then
  echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

if [[ ! "${bench_name}" =~ ^[A-Za-z0-9_.-]+$ ]] || [[ ! "${ckpt_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "bench_name and ckpt_name may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if [[ "${env_cfg_type}" != "tianji_marvin_wuji" ]]; then
  echo "VITRA spark0 conversion supports env_cfg_type=tianji_marvin_wuji only." >&2
  exit 2
fi
if [[ "${action_type}" != "ee" ]]; then
  echo "VITRA is exposed to XPolicyLab as action_type=ee only." >&2
  exit 2
fi
if [[ -n "${expert_data_num}" ]] && [[ ! "${expert_data_num}" =~ ^[1-9][0-9]*$ ]]; then
  echo "expert_data_num must be a positive integer when provided." >&2
  exit 2
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
DATA_SCRIPTS_DIR="${WORKSPACE_ROOT}/data_scripts"
REPRESENTATION="${VITRA_DATA_REPRESENTATION:-wuji20}"
if [[ "${REPRESENTATION}" != "wuji20" && "${REPRESENTATION}" != "mano45" ]]; then
  echo "VITRA_DATA_REPRESENTATION must be wuji20 or mano45, got: ${REPRESENTATION}" >&2
  exit 2
fi
if [[ "${REPRESENTATION}" == "mano45" ]]; then
  SOURCE_ROOT="${SPARK0_SOURCE_DATA:-/personal/zijian/Spark_0/data/spark0_bench_mano}"
  OUTPUT_ROOT="${VITRA_DATA_ROOT:-${WORKSPACE_ROOT}/data_mano}"
else
  SOURCE_ROOT="${SPARK0_SOURCE_DATA:-/mnt/xspark-data/tjy/spark0_bench}"
  OUTPUT_ROOT="${VITRA_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
fi

prepare_args=(
  --source-root "${SOURCE_ROOT}"
  --data-root "${OUTPUT_ROOT}"
  --representation "${REPRESENTATION}"
)
if [[ -n "${expert_data_num}" ]]; then
  prepare_args+=(--expert-data-num "${expert_data_num}")
fi
if [[ -n "${VITRA_EXPECTED_TASKS:-}" ]]; then
  prepare_args+=(--expected-tasks "${VITRA_EXPECTED_TASKS}")
fi
if [[ -n "${VITRA_EXPECTED_EPISODES:-}" ]]; then
  prepare_args+=(--expected-episodes "${VITRA_EXPECTED_EPISODES}")
fi
if [[ "${VITRA_ALLOW_COUNT_MISMATCH:-0}" == "1" ]]; then
  prepare_args+=(--allow-count-mismatch)
fi

python3 "${DATA_SCRIPTS_DIR}/prepare_vitra_data.py" "${prepare_args[@]}"
python3 "${DATA_SCRIPTS_DIR}/calculate_vitra_statistics.py" \
  --data-root "${OUTPUT_ROOT}" \
  --representation "${REPRESENTATION}"
python3 "${DATA_SCRIPTS_DIR}/validate_vitra_data.py" \
  --data-root "${OUTPUT_ROOT}" \
  --representation "${REPRESENTATION}" \
  --decode-images \
  --output "${OUTPUT_ROOT}/validation_report.json"
