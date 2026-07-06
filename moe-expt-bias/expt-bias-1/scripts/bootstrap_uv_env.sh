#!/bin/bash
set -euo pipefail

ENV_DIR="${1:-}"
if [[ -z "$ENV_DIR" ]]; then
  echo "usage: $0 <env-dir>"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH"
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"
if [[ ! -d "$ENV_DIR" ]]; then
  uv venv "$ENV_DIR" --python python3.11
fi

READY_MARKER="$ENV_DIR/.moe_bias_shapley_ready"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

if [[ "${FORCE_UV_REINSTALL:-0}" != "1" && -f "$READY_MARKER" ]]; then
  echo "[uv] reusing existing environment at $ENV_DIR"
  exit 0
fi

uv pip install --upgrade pip setuptools wheel
uv pip install -e .

# torch pinned to a CUDA 12.1-compatible wheel to match the `cuda/12.1.1` module.
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.0}"
uv pip install "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cu121

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.44}"
uv pip install "$TRANSFORMERS_SPEC" accelerate datasets sentencepiece protobuf

# Optional: 4-bit/8-bit loading for the larger models on the ladder
# (Qwen3-30B-A3B, Mixtral-8x7B, ERNIE-4.5-21B-A3B) on single-GPU nodes.
if [[ "${INSTALL_BITSANDBYTES:-1}" == "1" ]]; then
  uv pip install bitsandbytes
fi

touch "$READY_MARKER"
