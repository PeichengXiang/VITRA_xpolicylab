#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VITRA_ROOT="${SCRIPT_DIR}/VITRA"

if [[ ! -f "${VITRA_ROOT}/pyproject.toml" ]]; then
    echo "[ERROR] Missing official VITRA checkout at ${VITRA_ROOT}" >&2
    exit 1
fi

echo "[VITRA] Installing official VITRA and XPolicyLab into the active environment"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${VITRA_ROOT}"
python -m pip install -e "${XPOLICYLAB_ROOT}"
python -m pip check

python - <<'PY'
import torch
import vitra
import XPolicyLab

print(f"[VITRA] torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"[VITRA] vitra={vitra.__path__}")
print(f"[VITRA] XPolicyLab={XPolicyLab.__path__}")
PY

echo "[VITRA] Installation finished"
