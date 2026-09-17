#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

# Eval Web invokes this script from a non-login shell.  Use the explicit
# environment path when supplied; only named environments need conda.
if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    EVAL_PYTHON="${eval_env_conda_env}/bin/python"
    export CONDA_PREFIX="${eval_env_conda_env}"
    export PATH="$(dirname "${EVAL_PYTHON}"):${PATH}"
elif [[ -x "${eval_env_conda_env}" ]]; then
    EVAL_PYTHON="${eval_env_conda_env}"
    export PATH="$(dirname "${EVAL_PYTHON}"):${PATH}"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate || true
    conda activate "${eval_env_conda_env}"
    EVAL_PYTHON="$(command -v python)"
else
    echo "[ERROR] cannot resolve Eval Web Python environment: ${eval_env_conda_env}" >&2
    exit 2
fi
echo "[CLIENT] eval_python=${EVAL_PYTHON}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

# Relay the scheduler's per-background contract explicitly.  In particular,
# use a longer first-call timeout while VITRA loads its diffusion model and
# preserve the child output/run id used by the strict official-suite checker.
egovla_cli_args=(
    --episodes "${EGOVLA_EPISODES:-${EVAL_NUM:-1}}"
    --num-envs "${EGOVLA_NUM_ENVS:-1}"
    --request-timeout-s "${EGOVLA_REQUEST_TIMEOUT_S:-600}"
)
if [[ -n "${EGOVLA_OUTPUT_DIR:-}" ]]; then
    egovla_cli_args+=(--output-dir "${EGOVLA_OUTPUT_DIR}")
fi
if [[ -n "${EGOVLA_OFFICIAL_SPLIT:-}" ]]; then
    egovla_cli_args+=(
        --official-split "${EGOVLA_OFFICIAL_SPLIT}"
        --official-episodes "${EGOVLA_OFFICIAL_EPISODES:-${EGOVLA_EPISODES:-1}}"
        --official-trials "${EGOVLA_OFFICIAL_TRIALS:-1}"
    )
fi
if [[ -n "${EGOVLA_ROOM_IDX:-}" ]]; then
    egovla_cli_args+=(--room-idx "${EGOVLA_ROOM_IDX}")
fi
if [[ -n "${EGOVLA_TABLE_IDX:-}" ]]; then
    egovla_cli_args+=(--table-idx "${EGOVLA_TABLE_IDX}")
fi
if [[ -n "${EGOVLA_CHECKPOINT:-}" ]]; then
    egovla_cli_args+=(--checkpoint "${EGOVLA_CHECKPOINT}")
fi
if [[ -n "${EGOVLA_ACTION_TYPE:-}" ]]; then
    egovla_cli_args+=(--action-type "${EGOVLA_ACTION_TYPE}")
fi
xpolicy_root_arg="${EGOVLA_XPOLICY_ROOT:-${XPOLICYLAB_ROOT:-}}"
if [[ -n "${xpolicy_root_arg}" ]]; then
    egovla_cli_args+=(--xpolicy-root "${xpolicy_root_arg}")
fi

bash "${root_dir}/scripts/eval_policy.sh" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --host "${policy_server_ip}" \
    --port "${policy_server_port}" \
    --protocol "${protocol}" \
    --eval_batch "${eval_batch}" \
    --root_dir "${root_dir}" \
    --device_id "${env_gpu_id}" \
    --additional_info "${additional_info}" \
    --seed "${seed}" \
    "${egovla_cli_args[@]}"
