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

# torch>=2.6.0 required for torch.accelerator (added in 2.6.0; needed by the
# transformers MXFP4 quantizer for openai/gpt-oss-120b).
# TORCH_INDEX defaults to cu126; override with TORCH_INDEX=.../cu121 for older nodes.
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
uv pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.44}"
# tiktoken required by alpindale/dbrx-instruct custom modeling code (trust_remote_code)
uv pip install "$TRANSFORMERS_SPEC" accelerate datasets sentencepiece protobuf tiktoken

# Optional: 4-bit/8-bit loading for the larger models on the ladder
# (Phi-3.5-MoE, Mixtral-8x7B, Llama-3.1-8B) on single-GPU nodes.
if [[ "${INSTALL_BITSANDBYTES:-1}" == "1" ]]; then
  uv pip install bitsandbytes
fi

touch "$READY_MARKER"
