#!/usr/bin/env bash
# stats_analysis series — per-pair-phi rerun submissions (primary + backup copies)
# Located/staged: expt-bias-1/stats_analysis/slurm/submit_perpair_reruns.sh
# Run from: cluster login node (cwd anywhere; WORKDIR resolves the repo)
# Relies on cluster copy of repo having --save-per-pair-phi (synced 2026-08-08).
set -u

WORKDIR=~/scratch/moe-breakdown
ROOT_DIR=$WORKDIR/moe-expt-bias/expt-bias-1
CFG=$ROOT_DIR/configs
SLURM=$ROOT_DIR/slurm
OUT_BASE=$HOME/scratch/moe-breakdown-bias-runs/expt-bias-1
SBNAME=run_experiment_script.sbatch
EXTRA="--save-per-pair-phi"

# usage: submit_job <jobname> <config> <gres> <mem> <cpus> <time> <mods> <outdir> <shard_id> <num_shards> [extra_env,...]
submit_job () {
  local name=$1 cfg=$2 gres=$3 mem=$4 cpus=$5 time=$6 mods=$7 out=$8 sid="${9:-}" ns="${10:-1}"
  shift 10
  local export_str="ALL,SCRIPT=scripts/run_bias_study.py,STUDY_CONFIG=$cfg,OUT_DIR=$out,WORKDIR=$WORKDIR,MODULES=$mods,EXTRA_ARGS=$EXTRA"
  [[ "$ns" -gt 1 ]] && export_str="$export_str,SHARD_IDX=$sid,NUM_SHARDS=$ns"
  [[ $# -gt 0 ]] && export_str="$export_str,$*"
  echo ">> submitting $name -> $out (${gres}, ${time})"
  sbatch --job-name="$name" --partition=ice-gpu --qos=coc-ice \
    --cpus-per-task="$cpus" --mem="$mem" --gres="$gres" \
    --time="$time" \
    --export="$export_str" \
    "$SLURM/$SBNAME"
}

# --- primary + backup jobs (each model: same config, identical gres, fresh out dir) ---

submit_job pp-olmoe         "$CFG/study.olmoe.concentration.v1.yaml"        "gpu:l40s:1" 64G 8 04:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-olmoe-1b-7b-v1-perpair"
submit_job bkp-pp-olmoe     "$CFG/study.olmoe.concentration.v1.yaml"        "gpu:l40s:1" 64G 8 04:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-olmoe-1b-7b-v1-perpair-b"

submit_job pp-phi35         "$CFG/study.phi3.5-moe.concentration.v1.yaml"    "gpu:h100:2" 96G 8 05:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-phi3.5-moe-v1-perpair"
submit_job bkp-pp-phi35     "$CFG/study.phi3.5-moe.concentration.v1.yaml"    "gpu:h100:2" 96G 8 05:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-phi3.5-moe-v1-perpair-b"

submit_job pp-mixtral-s0    "$CFG/study.mixtral-8x7b.concentration.v1.yaml"   "gpu:h100:4" 96G 8 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-mixtral-8x7b-v1-perpair" 0 2
submit_job pp-mixtral-s1    "$CFG/study.mixtral-8x7b.concentration.v1.yaml"   "gpu:h100:4" 96G 8 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-mixtral-8x7b-v1-perpair" 1 2
submit_job bkp-pp-mixtral-s0 "$CFG/study.mixtral-8x7b.concentration.v1.yaml"  "gpu:h100:4" 96G 8 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-mixtral-8x7b-v1-perpair-b" 0 2
submit_job bkp-pp-mixtral-s1 "$CFG/study.mixtral-8x7b.concentration.v1.yaml"  "gpu:h100:4" 96G 8 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-mixtral-8x7b-v1-perpair-b" 1 2

submit_job pp-dbrx-s0       "$CFG/study.dbrx.concentration.v1.yaml"          "gpu:h100:4" 200G 16 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.6.1" \
  "$OUT_BASE/exp1-concentration-dbrx-v1-perpair" 0 2
submit_job pp-dbrx-s1       "$CFG/study.dbrx.concentration.v1.yaml"          "gpu:h100:4" 200G 16 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.6.1" \
  "$OUT_BASE/exp1-concentration-dbrx-v1-perpair" 1 2
submit_job bkp-pp-dbrx-s0   "$CFG/study.dbrx.concentration.v1.yaml"          "gpu:h100:4" 200G 16 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.6.1" \
  "$OUT_BASE/exp1-concentration-dbrx-v1-perpair-b" 0 2
submit_job bkp-pp-dbrx-s1   "$CFG/study.dbrx.concentration.v1.yaml"          "gpu:h100:4" 200G 16 03:45:00 "gcc/12.3.0 python/3.11 cuda/12.6.1" \
  "$OUT_BASE/exp1-concentration-dbrx-v1-perpair-b" 1 2

submit_job pp-gptoss        "$CFG/study.gpt-oss-120b.concentration.yaml"      "gpu:h100:2" 200G 16 03:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-gpt-oss-120b-perpair"
submit_job bkp-pp-gptoss    "$CFG/study.gpt-oss-120b.concentration.yaml"      "gpu:h100:2" 200G 16 03:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-gpt-oss-120b-perpair-b"

submit_job pp-gemma         "$CFG/study.gemma4-27b.concentration.yaml"        "gpu:h100:1" 96G 8 06:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-gemma4-26b-perpair"
submit_job bkp-pp-gemma     "$CFG/study.gemma4-27b.concentration.yaml"        "gpu:h100:1" 96G 8 06:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp1-concentration-gemma4-26b-perpair-b"

submit_job pp-exp5          "$CFG/study.olmoe.demographic.v1.yaml"            "gpu:l40s:1" 64G 8 04:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp5-demographic-specificity-olmoe-1b-7b-v1-perpair"
submit_job bkp-pp-exp5      "$CFG/study.olmoe.demographic.v1.yaml"            "gpu:l40s:1" 64G 8 04:00:00 "gcc/12.3.0 python/3.11 cuda/12.1.1" \
  "$OUT_BASE/exp5-demographic-specificity-olmoe-1b-7b-v1-perpair-b"

echo "=== submitted; queue ==="
squeue -u "$USER"