#!/usr/bin/env bash
set -euo pipefail

# Train VITRA on spark0_bench_7tasks using wenwei wuji2_mano45_xyz_sparse_v1
# table-index mapping. Official robot path: normalize Wuji20, then inject.

MODEL_ROOT=/personal/xiangpc/0813_Xpolicylab_bench/VITRA
POLICY_ROOT="${MODEL_ROOT}/XPolicyLab/policy/VITRA"
SOURCE_ROOT=/personal/xspark_shared/hand_data/hdf5/spark0_bench_7tasks
DATA_ROOT="${MODEL_ROOT}/data/spark0_bench_7task_wuji2mano_v1"
WANDB_ENV=/personal/xiangpc/training_0908-real/wandb.env

export PATH=/personal/miniconda3/envs/vitra_xpolicylab_0813/bin:$PATH
export SPARK0_SOURCE_DATA="${SOURCE_ROOT}"
export VITRA_DATA_ROOT="${DATA_ROOT}"
export VITRA_DATA_REPRESENTATION=wuji20
export VITRA_EXPECTED_TASKS=7
export VITRA_EXPECTED_EPISODES=700
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VITRA_RUN_DATE="${VITRA_RUN_DATE:-$(date +%F)}"
# 4090 24G OOMs at per-device 8. Keep global 64 via smaller microbatch.
export VITRA_PER_DEVICE_BATCH="${VITRA_PER_DEVICE_BATCH:-4}"
GPU_IDS="${VITRA_GPU_ID:-0,1,2,3,4,5,6,7}"

if [[ ! -f "${WANDB_ENV}" ]]; then
  echo "Missing wandb env: ${WANDB_ENV}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${WANDB_ENV}"
export WANDB_PROJECT=xpolicylab-vitra-wuji2mano
export WANDB_RUN_GROUP=7task_wuji2mano_v1
export WANDB_MODE=online
export WANDB_NAME="${WANDB_NAME:-vitra-7task-wuji2mano-v1}"

cd "${POLICY_ROOT}"
if [[ ! -s "${DATA_ROOT}/teledata_statistics.json" ]]; then
  bash process_data.sh spark0_bench_7task_wuji2mano_v1 7task_sparse tianji_marvin_wuji ee
fi

exec bash train.sh spark0_bench_7task_wuji2mano_v1 7task_sparse tianji_marvin_wuji ee 42 "${GPU_IDS}"
