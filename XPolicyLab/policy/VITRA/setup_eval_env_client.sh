#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}
dry_run=${12:-""}

if [[ "${action_type}" != "ee" ]]; then
    echo "[ERROR] VITRA environment client requires action_type=ee" >&2
    exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ ! -d "${XPOLICYLAB_ROOT}/policy/VITRA" ]]; then
    echo "[ERROR] resolved XPOLICYLAB_ROOT is not the VITRA checkout: ${XPOLICYLAB_ROOT}" >&2
    exit 2
fi
if [[ -f "${XPOLICYLAB_ROOT}/env_cfg/ego_h1_inspire.yml" ]]; then
    ENV_CFG_PATH="${XPOLICYLAB_ROOT}/env_cfg/ego_h1_inspire.yml"
elif [[ -f "${XPOLICYLAB_ROOT}/../env_cfg/ego_h1_inspire.yml" ]]; then
    ENV_CFG_PATH="${XPOLICYLAB_ROOT}/../env_cfg/ego_h1_inspire.yml"
    echo "[WARN] env_cfg is in the private checkout parent of XPOLICYLAB_ROOT" >&2
else
    echo "[ERROR] missing env_cfg/ego_h1_inspire.yml beside resolved VITRA checkout" >&2
    exit 2
fi
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# Eval Web passes either an absolute conda environment root or an environment
# name.  Resolve the exact interpreter before any YAML parsing or Isaac launch
# and use it for the dependency check and downstream client process.
if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    EVAL_RUNTIME_PYTHON="${eval_env_conda_env}/bin/python"
elif [[ -x "${eval_env_conda_env}" ]]; then
    EVAL_RUNTIME_PYTHON="${eval_env_conda_env}"
elif command -v conda >/dev/null 2>&1; then
    EVAL_RUNTIME_PYTHON="$(conda run --no-capture-output -n "${eval_env_conda_env}" python -c 'import sys; print(sys.executable)' 2>/dev/null)"
else
    EVAL_RUNTIME_PYTHON=""
fi
if [[ -z "${EVAL_RUNTIME_PYTHON}" || ! -x "${EVAL_RUNTIME_PYTHON}" ]]; then
    echo "[ERROR] cannot resolve Eval Web runtime Python for ${eval_env_conda_env}" >&2
    exit 2
fi
echo "[CLIENT] eval_runtime_python=${EVAL_RUNTIME_PYTHON}"
if ! YAML_SOURCE="$("${EVAL_RUNTIME_PYTHON}" -c 'import yaml; print(yaml.__file__)' 2>/dev/null)"; then
    echo "[ERROR] PyYAML is missing from the Eval Web runtime: ${EVAL_RUNTIME_PYTHON}" >&2
    echo "[ERROR] install PyYAML into this exact environment (not policy env): ${EVAL_RUNTIME_PYTHON} -m pip install PyYAML" >&2
    exit 2
fi
echo "[CLIENT] yaml_source=${YAML_SOURCE}"

if [[ "${dry_run}" == "--dry-run" ]]; then
    echo "[CLIENT] dry-run root=${XPOLICYLAB_ROOT} env_cfg=${ENV_CFG_PATH} policy=${policy_name} action_type=${action_type} env_cfg_type=${env_cfg_type}"
    exit 0
fi

export EGOVLA_EVAL_PYTHON="${EVAL_RUNTIME_PYTHON}"
export EGOVLA_EVAL_CONDA_ENV="${eval_env_conda_env}"
export XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT}"
export EGOVLA_POLICY_ADAPTER_ROOT="${XPOLICYLAB_ROOT}"
# The policy checkout is often mounted beside (or through a symlink to) the
# benchmark workspace.  Prefer an explicitly supplied workspace, then the
# bridge's workspace variable, and finally discover the first candidate that
# owns the stable scripts/eval_policy.sh entrypoint.  This keeps the policy
# root resolution independent from the mount spelling and avoids passing the
# VITRA checkout parent as a benchmark root.
BENCH_ROOT=""
for candidate in \
    "${EVAL_MAIN_ROOT:-}" \
    "${EGOVLA_WORKSPACE_ROOT:-}" \
    "${EGOVLA_ROOT:-}" \
    "$(cd "${XPOLICYLAB_ROOT}/.." && pwd)"; do
    [[ -n "${candidate}" ]] || continue
    candidate="$(readlink -f "${candidate}")"
    if [[ -x "${candidate}/scripts/eval_policy.sh" ]]; then
        BENCH_ROOT="${candidate}"
        break
    fi
done
if [[ -z "${BENCH_ROOT}" ]]; then
    echo "[ERROR] cannot resolve benchmark workspace containing scripts/eval_policy.sh" >&2
    echo "[ERROR] checked EVAL_MAIN_ROOT=${EVAL_MAIN_ROOT:-<unset>}, EGOVLA_WORKSPACE_ROOT=${EGOVLA_WORKSPACE_ROOT:-<unset>}, EGOVLA_ROOT=${EGOVLA_ROOT:-<unset>}" >&2
    exit 2
fi
echo "[CLIENT] benchmark_root=${BENCH_ROOT}"
echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
    "${policy_server_port}" "${bench_name}" "${task_name}" \
    "${env_cfg_type}" "${policy_name}" "${additional_info}" \
    "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
