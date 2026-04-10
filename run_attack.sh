#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_attack.sh --model ViT-B-32 [other main.py args...]
#
# Optional environment variables:
#   PYTHON_BIN            Python executable (default: python3)
#   MODEL_NAME            CLIP model name passed to --model (default: ViT-B-32)
#   CONDA_ENV             Conda env name to activate before running (optional)
#   EXTRA_ARGS            Extra args appended after CLI args (optional)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "[WARN] CONDA_ENV is set but conda is not available. Continuing without activation." >&2
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-ViT-B-32}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable '$PYTHON_BIN' not found." >&2
  exit 1
fi

echo "[INFO] Running attack pipeline from: $ROOT_DIR"
echo "[INFO] Command: $PYTHON_BIN main.py --model $MODEL_NAME $* ${EXTRA_ARGS:-}"

exec "$PYTHON_BIN" main.py --model "$MODEL_NAME" "$@" ${EXTRA_ARGS:-}
