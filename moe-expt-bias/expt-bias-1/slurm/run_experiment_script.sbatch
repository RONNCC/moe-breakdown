#!/bin/bash
# Generic sbatch wrapper for Experiment 3 (interactions) and Experiment 4
# (ablation cross-check) driver scripts — same env/venv bootstrap as
# run_bias_study.sbatch, but runs an arbitrary script under scripts/ with
# arbitrary extra CLI args instead of hardcoding run_bias_study.py.
#
# Required env vars:
#   SCRIPT        - path relative to expt-bias-1/, e.g. scripts/run_experiment3_interactions.py
#   STUDY_CONFIG  - path to the study YAML (--config)
# Optional env vars:
#   EXTRA_ARGS    - extra CLI args appended verbatim, e.g. "--max-pairs 20 --max-layers 2"
#   OUT_DIR, WORKDIR, VENV, MODULES, UV_ENV_DIR - same meaning as run_bias_study.sbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${SCRIPT:-}" ]]; then
  echo "SCRIPT is required (e.g. scripts/run_experiment3_interactions.py)"
  exit 1
fi
if [[ -z "${STUDY_CONFIG:-}" ]]; then
  echo "STUDY_CONFIG is required"
  exit 1
fi

WORKDIR="${WORKDIR:-$PWD}"
VENV="${VENV:-}"
MODULES="${MODULES:-}"
OUT_DIR="${OUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
UV_ENV_DIR="${UV_ENV_DIR:-${TMPDIR:-}/moe-bias-shapley-venv}"

ROOT_DIR="$WORKDIR/moe-expt-bias/expt-bias-1"
cd "$ROOT_DIR"

module purge || true
if [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

# HF_HOME persists on /storage (not $TMPDIR) so multi-GB model checkpoints
# are downloaded once and reused across experiment runs.
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
echo "[slurm] SCRIPT=$SCRIPT"
echo "[slurm] STUDY_CONFIG=$STUDY_CONFIG"
echo "[slurm] OUT_DIR=$OUT_DIR"
echo "[slurm] EXTRA_ARGS=$EXTRA_ARGS"
echo "[slurm] GPUs visible: $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown (torch not yet importable)')"

ARGS=(--config "$STUDY_CONFIG")
if [[ -n "$OUT_DIR" ]]; then
  ARGS+=(--out-dir "$OUT_DIR")
fi
# shellcheck disable=SC2206
if [[ -n "$EXTRA_ARGS" ]]; then
  EXTRA_ARGS_ARR=($EXTRA_ARGS)
  ARGS+=("${EXTRA_ARGS_ARR[@]}")
fi

python3 "$ROOT_DIR/$SCRIPT" "${ARGS[@]}"
