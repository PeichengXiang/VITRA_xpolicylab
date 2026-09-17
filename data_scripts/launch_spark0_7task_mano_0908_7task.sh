#!/usr/bin/env bash
set -euo pipefail
MODEL_ROOT=/personal/xiangpc/0813_Xpolicylab_bench/VITRA
POLICY_ROOT="$MODEL_ROOT/XPolicyLab/policy/VITRA"
DATA_ROOT="$MODEL_ROOT/data/spark0_bench_7task_0908_mano"
export PATH=/personal/miniconda3/envs/vitra_xpolicylab_0813/bin:$PATH
export VITRA_DATA_ROOT="$DATA_ROOT" VITRA_DATA_REPRESENTATION=mano45 VITRA_RUN_DATE=2026-09-09 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
source /root/.config/training_0908-real/wandb.env
export WANDB_PROJECT=xpolicylab-0908_7task WANDB_RUN_GROUP=0908_7task WANDB_MODE=online WANDB_NAME=vitra-0908_7task
cd "$POLICY_ROOT"
exec bash train.sh spark0_bench_7task_0908_mano 0908_7task tianji_marvin_wuji ee 42 all