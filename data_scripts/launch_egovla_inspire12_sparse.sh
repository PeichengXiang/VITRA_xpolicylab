#!/usr/bin/env bash
set -euo pipefail

# Train VITRA on EgoVLA-Humanoid-Sim using inspire12_mano45_xyz_sparse_v1
# table-index mapping. Official robot path: normalize Inspire12, then inject.

MODEL_ROOT=/personal/xiangpc/0813_Xpolicylab_bench/VITRA
POLICY_ROOT="${MODEL_ROOT}/XPolicyLab/policy/VITRA"
SOURCE_ROOT="/personal/xiangpc/EgoVLA benchmark/XPolicyLab/data/EgoVLA/raw_remove_deprecated"
DATA_ROOT="${MODEL_ROOT}/data/egovla_inspire12_sparse_v1"
WANDB_ENV=/personal/xiangpc/training_0908-real/wandb.env

export PATH=/personal/miniconda3/envs/vitra_xpolicylab_0813/bin:$PATH
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

if [[ ! -f "${WANDB_ENV}" ]]; then
  echo "Missing wandb env: ${WANDB_ENV}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${WANDB_ENV}"
export WANDB_PROJECT=xpolicylab-vitra-egovla-inspire
export WANDB_RUN_GROUP=egovla_inspire12_sparse_v1
export WANDB_MODE=online
export WANDB_NAME="${WANDB_NAME:-vitra-egovla-inspire12-sparse-v1}"

cd "${POLICY_ROOT}"
if [[ ! -s "${DATA_ROOT}/spark0_manifest.json" ]]; then
  python3 "${MODEL_ROOT}/data_scripts/prepare_egovla_inspire12_vitra.py" \
    --source-root "${SOURCE_ROOT}" \
    --data-root "${DATA_ROOT}"
fi
if [[ ! -s "${DATA_ROOT}/teledata_statistics.json" ]]; then
  python3 "${MODEL_ROOT}/data_scripts/calculate_vitra_statistics.py" \
    --data-root "${DATA_ROOT}" \
    --representation inspire12 \
    --mapping "${POLICY_ROOT}/mapping_inspire12_mano45.json"
fi

exec bash train.sh egovla_inspire12_sparse v1 tianji_marvin_wuji ee 42 "${GPU_IDS}"
