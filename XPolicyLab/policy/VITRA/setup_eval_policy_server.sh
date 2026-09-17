#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: $0 bench task checkpoint env_cfg action_type seed gpu_id policy_env port [host]" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}
if [[ ! "$policy_gpu_id" =~ ^[0-9]+$ || ! "$policy_server_port" =~ ^[0-9]+$ ]] || ((10#$policy_server_port < 1 || 10#$policy_server_port > 65535)); then
    echo "[ERROR] gpu_id must be nonnegative and port must be in 1..65535" >&2
    exit 2
fi
case "$policy_conda_env" in auto|vitra_xpolicylab_0813) policy_conda_env=/personal/miniconda3/envs/vitra_xpolicylab_0813 ;; esac

if [[ "${action_type}" != "ee" ]]; then
    echo "[ERROR] VITRA policy server requires action_type=ee" >&2
    exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ ! -d "${XPOLICYLAB_ROOT}/policy/VITRA" ]]; then
    echo "[ERROR] resolved XPOLICYLAB_ROOT is not the VITRA checkout: ${XPOLICYLAB_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${XPOLICYLAB_ROOT}/env_cfg/${env_cfg_type}.yml" && ! -f "${XPOLICYLAB_ROOT}/../env_cfg/${env_cfg_type}.yml" ]]; then
    echo "[ERROR] resolved VITRA checkout has no env_cfg/${env_cfg_type}.yml: ${XPOLICYLAB_ROOT}" >&2
    exit 2
fi
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, gpu=${policy_gpu_id}"

# Eval Web normally supplies an absolute policy-environment path.  Resolve its
# interpreter directly so a non-login worker shell does not depend on a
# `conda` executable being present on PATH.  Named environments retain the
# historical conda activation path.
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    POLICY_PYTHON="${policy_conda_env}/bin/python"
    export CONDA_PREFIX="${policy_conda_env}"
    export PATH="$(dirname "${POLICY_PYTHON}"):${PATH}"
elif [[ -x "${policy_conda_env}" ]]; then
    POLICY_PYTHON="${policy_conda_env}"
    export PATH="$(dirname "${POLICY_PYTHON}"):${PATH}"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    POLICY_PYTHON="$(command -v python)"
else
    echo "[ERROR] cannot resolve policy Python environment: ${policy_conda_env}" >&2
    exit 2
fi

# Use the installed CUDA-compatible packages; serving stays in the original model.py.
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME TRANSFORMERS_CACHE
export PYTHONPATH="/personal/wenwei/Spark0-eval/server_scripts/.runtime_vitra_dependencies:${XPOLICYLAB_ROOT}/..:${XPOLICYLAB_ROOT}:${SCRIPT_DIR}/VITRA"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VITRA_MANO_INVERSE_MODE="${VITRA_MANO_INVERSE_MODE:-geometric}"
if [[ "$VITRA_MANO_INVERSE_MODE" == linear ]]; then
    export VITRA_MANO_LINEAR_INVERSE_PATH="${VITRA_MANO_LINEAR_INVERSE_PATH:-/personal/xiangpc/0813_Xpolicylab_bench/VITRA/XPolicyLab/policy/VITRA/checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True/mano45_to_wuji20_linear_0902_v1.json}"
    export VITRA_MANO_LINEAR_INVERSE_SHA256="${VITRA_MANO_LINEAR_INVERSE_SHA256:-f0a760427d6fb7fd7aa9c7d63aea8666c67f4f7344c25e93f8d7f1adf05caba5}"
elif [[ "$VITRA_MANO_INVERSE_MODE" != geometric ]]; then
    echo "[ERROR] VITRA_MANO_INVERSE_MODE must be geometric or linear" >&2
    exit 2
fi

echo "[SERVER] policy_python=${POLICY_PYTHON}"
if ! POLICY_YAML_SOURCE="$("${POLICY_PYTHON}" -c 'import yaml; print(yaml.__file__)' 2>/dev/null)"; then
    echo "[ERROR] PyYAML is missing from the policy runtime: ${POLICY_PYTHON}" >&2
    exit 2
fi
echo "[SERVER] yaml_source=${POLICY_YAML_SOURCE}"

OVERRIDES=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    gpu_id=0
    ws_ping_timeout_s=120
)

if [[ -n "${VITRA_MODEL_PATH:-}" ]]; then
    OVERRIDES+=(model_path="${VITRA_MODEL_PATH}")
fi
if [[ -n "${VITRA_CONFIG_PATH:-}" ]]; then
    OVERRIDES+=(model_config_path="${VITRA_CONFIG_PATH}")
fi
if [[ -n "${VITRA_STATISTICS_PATH:-}" ]]; then
    OVERRIDES+=(statistics_path="${VITRA_STATISTICS_PATH}")
fi
if [[ -n "${VITRA_STATISTICS_SHA256:-}" ]]; then
    OVERRIDES+=(statistics_sha256="${VITRA_STATISTICS_SHA256}")
fi
if [[ -n "${VITRA_MAPPING_PATH:-}" ]]; then
    OVERRIDES+=(mapping_path="${VITRA_MAPPING_PATH}")
fi
if [[ -n "${VITRA_DATA_REPRESENTATION:-}" ]]; then
    OVERRIDES+=(data_representation="${VITRA_DATA_REPRESENTATION}")
fi
if [[ -n "${VITRA_MANO_TOOLS_ROOT:-}" ]]; then
    OVERRIDES+=(mano_tools_root="${VITRA_MANO_TOOLS_ROOT}")
fi
if [[ -n "${VITRA_MANO_INVERSE_MODE:-}" ]]; then
    OVERRIDES+=(mano_inverse_mode="${VITRA_MANO_INVERSE_MODE}")
fi
if [[ -n "${VITRA_MANO_LINEAR_INVERSE_PATH:-}" ]]; then
    OVERRIDES+=(mano_linear_inverse_path="${VITRA_MANO_LINEAR_INVERSE_PATH}")
fi
if [[ -n "${VITRA_MANO_LINEAR_INVERSE_SHA256:-}" ]]; then
    OVERRIDES+=(mano_linear_inverse_sha256="${VITRA_MANO_LINEAR_INVERSE_SHA256}")
fi
if [[ -n "${VITRA_MANO_INVERSE_MAX_ITERATIONS:-}" ]]; then
    OVERRIDES+=(mano_inverse_max_iterations="${VITRA_MANO_INVERSE_MAX_ITERATIONS}")
fi
if [[ -n "${VITRA_MANO_INVERSE_TIMEOUT_S:-}" ]]; then
    OVERRIDES+=(mano_inverse_timeout_s="${VITRA_MANO_INVERSE_TIMEOUT_S}")
fi
if [[ -n "${VITRA_H1_RETARGET_ROOT:-}" ]]; then
    OVERRIDES+=(h1_retarget_root="${VITRA_H1_RETARGET_ROOT}")
fi
if [[ -n "${VITRA_H1_RETARGET_SOLVER:-}" ]]; then
    OVERRIDES+=(h1_retarget_solver="${VITRA_H1_RETARGET_SOLVER}")
fi
if [[ -n "${VITRA_H1_RETARGET_MAX_ITERATIONS:-}" ]]; then
    OVERRIDES+=(h1_retarget_max_iterations="${VITRA_H1_RETARGET_MAX_ITERATIONS}")
fi
if [[ -n "${VITRA_H1_RETARGET_TOLERANCE_M:-}" ]]; then
    OVERRIDES+=(h1_retarget_tolerance_m="${VITRA_H1_RETARGET_TOLERANCE_M}")
fi
if [[ -n "${VITRA_H1_INVERSE_MAX_RMS_M:-}" ]]; then
    OVERRIDES+=(h1_inverse_max_rms_m="${VITRA_H1_INVERSE_MAX_RMS_M}")
fi
if [[ -n "${VITRA_H1_MIMIC_TOLERANCE_RAD:-}" ]]; then
    OVERRIDES+=(h1_mimic_tolerance_rad="${VITRA_H1_MIMIC_TOLERANCE_RAD}")
fi
if [[ -n "${VITRA_H1_INVERSE_MODE:-}" ]]; then
    OVERRIDES+=(h1_inverse_mode="${VITRA_H1_INVERSE_MODE}")
fi
if [[ -n "${VITRA_H1_LINEAR_INVERSE_PATH:-}" ]]; then
    OVERRIDES+=(h1_linear_inverse_path="${VITRA_H1_LINEAR_INVERSE_PATH}")
fi
if [[ -n "${VITRA_H1_LINEAR_INVERSE_SHA256:-}" ]]; then
    OVERRIDES+=(h1_linear_inverse_sha256="${VITRA_H1_LINEAR_INVERSE_SHA256}")
fi
if [[ -n "${VITRA_H1_CAMERA_CALIBRATION_PATH:-}" ]]; then
    OVERRIDES+=(h1_camera_calibration_path="${VITRA_H1_CAMERA_CALIBRATION_PATH}")
fi
if [[ -n "${VITRA_H1_CAMERA_CALIBRATION_SHA256:-}" ]]; then
    OVERRIDES+=(h1_camera_calibration_sha256="${VITRA_H1_CAMERA_CALIBRATION_SHA256}")
fi
if [[ -n "${VITRA_H1_CAMERA_CALIBRATION_SCOPE:-}" ]]; then
    OVERRIDES+=(h1_camera_calibration_scope="${VITRA_H1_CAMERA_CALIBRATION_SCOPE}")
fi
if [[ -n "${VITRA_NUM_DDIM_STEPS:-}" ]]; then
    OVERRIDES+=(num_ddim_steps="${VITRA_NUM_DDIM_STEPS}")
fi
if [[ -n "${VITRA_EXECUTE_ACTION_CHUNK:-}" ]]; then
    OVERRIDES+=(execute_action_chunk="${VITRA_EXECUTE_ACTION_CHUNK}")
fi

# MANO->Wuji solves many tiny 20x20 systems.  OpenBLAS defaults to all 64 host
# cores here, making one 16-step action chunk hundreds of times slower through
# thread-launch overhead.  Keep this process-local and overridable;
# mano45_runtime.py also scopes its solver as a defence for non-script entry.
export PYTHONWARNINGS=ignore::UserWarning
export OPENBLAS_NUM_THREADS="${VITRA_OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
server_command=("${POLICY_PYTHON}" -B -u "${XPOLICYLAB_ROOT}/setup_policy_server.py"
    --config_path "${yaml_file}" --protocol ws --host "$policy_server_host" --port "$policy_server_port"
    --overrides "${OVERRIDES[@]}")
cd "${XPOLICYLAB_ROOT}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${server_command[@]}"
    printf '\n'
    exit 0
fi
exec "${server_command[@]}"
