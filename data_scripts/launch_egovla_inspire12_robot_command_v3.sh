#!/usr/bin/env bash
set -euo pipefail

# Train VITRA with the official robot action split: observed EEF transition for
# the wrist and the same-row future controller command for Inspire12 joints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${VITRA_WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
POLICY_ROOT="${MODEL_ROOT}/XPolicyLab/policy/VITRA"
SOURCE_ROOT="${EGOVLA_SOURCE_ROOT:-/vepfs-cnbje63de6fae220/xiangpc/data/EgoVLA_raw_remove_deprecated}"
DATA_ROOT="${VITRA_DATA_ROOT:-${MODEL_ROOT}/data/egovla_inspire12_robot_command_v3}"
VITRA_ENV_ROOT="${VITRA_ENV_ROOT:-/vepfs-cnbje63de6fae220/xiangpc/.miniforge3/envs/vitra_xpolicylab}"

if [[ ! -x "${VITRA_ENV_ROOT}/bin/python" ]]; then
  echo "Missing VITRA Python environment: ${VITRA_ENV_ROOT}" >&2
  exit 1
fi
export PATH="${VITRA_ENV_ROOT}/bin:${PATH}"
export VITRA_DATA_ROOT="${DATA_ROOT}"
export VITRA_DATA_REPRESENTATION=inspire12
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VITRA_RUN_DATE="${VITRA_RUN_DATE:-$(date +%F)}"
export VITRA_PER_DEVICE_BATCH="${VITRA_PER_DEVICE_BATCH:-8}"
GPU_IDS="${VITRA_GPU_ID:-0,1,2,3,4,5,6,7}"

if [[ -n "${VITRA_WANDB_ENV:-}" ]]; then
  if [[ ! -f "${VITRA_WANDB_ENV}" ]]; then
    echo "Missing VITRA_WANDB_ENV file: ${VITRA_WANDB_ENV}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VITRA_WANDB_ENV}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Export WANDB_API_KEY or set VITRA_WANDB_ENV before training." >&2
  exit 1
fi
export WANDB_PROJECT=xpolicylab-vitra-egovla-inspire
export WANDB_RUN_GROUP=egovla_inspire12_robot_command_v3
export WANDB_MODE=online
export WANDB_NAME="${WANDB_NAME:-vitra-egovla-inspire12-robot-command-v3}"

cd "${POLICY_ROOT}"
if [[ ! -s "${DATA_ROOT}/CONVERSION_COMPLETE.json" ]]; then
  if [[ -e "${DATA_ROOT}" ]]; then
    echo "Incomplete EgoVLA data exists at ${DATA_ROOT}; quarantine it before retrying conversion." >&2
    exit 1
  fi
  bash "${SCRIPT_DIR}/prepare_egovla_inspire12_robot_command_v3.sh" \
    "${SOURCE_ROOT}" "${DATA_ROOT}"
fi

exec bash train.sh egovla_inspire12_robot_command v3 tianji_marvin_wuji ee 42 "${GPU_IDS}"
