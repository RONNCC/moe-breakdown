#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${STUDY_CONFIG:-}" ]]; then
  echo "STUDY_CONFIG is required"
  exit 1
fi

WORKDIR="${WORKDIR:-$PWD}"
VENV="${VENV:-}"
MODULES="${MODULES:-}"
OUT_DIR="${OUT_DIR:-}"
UV_ENV_DIR="${UV_ENV_DIR:-${TMPDIR:-}/moe-bias-shapley-venv}"

ROOT_DIR="$WORKDIR/moe-expt-bias/expt-bias-1"
cd "$ROOT_DIR"

module purge || true
if [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

# HF_HOME persists on /storage (not $TMPDIR) so multi-GB model checkpoints
# (Phi-3.5-MoE, Mixtral-8x7B, Phi-3.5-mini, Llama-3.1-8B, OLMoE-1B-7B, OLMo-7B) are downloaded once and reused
# across study runs instead of every job re-downloading to ephemeral node-local
# scratch.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ -n "${TMPDIR:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMPDIR/uv-cache}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton-cache}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor-cache}"
  mkdir -p "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
else
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
  mkdir -p "$UV_CACHE_DIR"
fi

if [[ -n "$UV_ENV_DIR" ]]; then
  bash "$ROOT_DIR/scripts/bootstrap_uv_env.sh" "$UV_ENV_DIR"
  VENV="$UV_ENV_DIR"
fi

if [[ -n "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[slurm] WORKDIR=$WORKDIR"
echo "[slurm] ROOT_DIR=$ROOT_DIR"
echo "[slurm] STUDY_CONFIG=$STUDY_CONFIG"
echo "[slurm] OUT_DIR=$OUT_DIR"
echo "[slurm] GPUs visible: $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown (torch not yet importable)')"

ARGS=(--config "$STUDY_CONFIG")
if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out-dir "$OUT_DIR")
fi

python3 "$ROOT_DIR/scripts/run_bias_study.py" "${ARGS[@]}"
