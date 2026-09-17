#!/usr/bin/env bash
set -euo pipefail

# Persistent A800_06 launcher for the active EgovLA MANO45 dataset.
# WANDB_API_KEY is intentionally supplied by the caller/systemd environment;
# it is never written to this file or to the generated training config.

MODEL_ROOT="/personal/xiangpc/0813_Xpolicylab_bench/VITRA"
POLICY_ROOT="${MODEL_ROOT}/XPolicyLab/policy/VITRA"
DATA_ROOT="${MODEL_ROOT}/data/egovla_humanoid_sim_mano_20260902"

export PATH="/personal/miniconda3/envs/vitra_xpolicylab_0813/bin:${PATH}"
export VITRA_DATA_ROOT="${DATA_ROOT}"
export VITRA_DATA_REPRESENTATION="mano45"
export VITRA_RUN_DATE="2026-09-02"
export HDF5_USE_FILE_LOCKING="FALSE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be supplied by the service environment." >&2
  exit 2
fi

cd "${POLICY_ROOT}"
exec bash train.sh egovla_humanoid_mano 20260902 tianji_marvin_wuji ee 42 all
