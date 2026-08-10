# GPT-OSS-120B rerun with `force_eager_moe` (eager MLP path)

Why: `results/exp1-concentration-gpt-oss-120b-v1/result.json` is a BROKEN run
(all-zero phi: `entropy=0, gini=0, n_nonzero=0`). transformers>=5.13 auto-attaches
the fused MXFP4 hub kernel (`transformers.integrations.mxfp4.mlp_forward`) to every
`GptOssMLP` on load; that forward bypasses the `GptOssTopKRouter` module, so
`output_router_logits` capture gets nothing (and on torch 2.6 it also crashes with
`AttributeError: _CudaDeviceProperties has no 'shared_memory_per_block_optin'`).

The fix is a config knob, `force_eager_moe`, consumed by
`src/moe_bias_shapley/modeling.py::load_model_and_tokenizer` (called from
`scripts/run_bias_study.py`), which rebinds every `GptOssMLP.forward` back to the
eager class method via `modeling.force_eager_gpt_oss` so the router executes and capture works.

## 1. Study YAML — `configs/study.gpt-oss-120b.concentration.v1.yaml`

`force_eager_moe: true` has been added (this is the only change vs the broken v1
run). Current content:

```yaml
study_name: exp1-concentration-gpt-oss-120b-v1

model_id: openai/gpt-oss-120b
model_family: gpt-oss
trust_remote_code: false
torch_dtype: bfloat16
load_in_4bit: false
device_map: auto
force_eager_moe: true      # <— REQUIRED: eager GptOssMLP/GptOssRouter path

benchmarks: [stereoset, bbq, winogender]
shapley_method: routing_contrast
max_prompts: 5000
seed: 42

output_root: ~/scratch/moe-breakdown-bias-runs/expt-bias-1

slurm:
  partition: ice-gpu
  qos: coc-ice
  time: 05:00:00
  cpus_per_task: 16
  gpus_per_node: 2          # 2×H100
  gpu_type: h100
  use_gres: true
  mem: 200G
  modules: [gcc/12.3.0, python/3.11, cuda/12.6.1]
  workdir: /home/hice1/sghose7/scratch/moe-breakdown
```

Do NOT touch `force_eager_moe` absence in the v0 config
(`configs/...-120b.concentration.yaml`, 400 pairs) — it stays as the v0 slot.

## 2. Submit command

From the cluster checkout (`/home/hice1/sghose7/scratch/moe-breakdown` — run on a
login node) or via `ssh login-ice.pace.gatech.edu` from the repo root:

```bash
python3 scripts/submit_slurm_study.py \
  --config configs/study.gpt-oss-120b.concentration.v1.yaml
```

This prints and runs (dry-run equivalent, verified against
`submit_slurm_study.py`):

```
sbatch --job-name=bias-exp1-concentration-gpt-oss-120b-v1 --nodes=1 \
  --cpus-per-task=16 --mem=200G --time=05:00:00 \
  --output=<out>/exp1-concentration-gpt-oss-120b-v1/slurm-logs/%x-%j.out \
  --error=<out>/exp1-concentration-gpt-oss-120b-v1/slurm-logs/%x-%j.err \
  --gres=gpu:h100:2 --partition ice-gpu --qos coc-ice \
  --export=ALL,STUDY_CONFIG=<config>,OUT_DIR=<out>/exp1-concentration-gpt-oss-120b-v1,WORKDIR=/home/hice1/sghose7/scratch/moe-breakdown,MODULES='gcc/12.3.0 python/3.11 cuda/12.6.1' \
  slurm/run_bias_study.sbatch
```

(Resource profile = the previous gpt-oss jobs: 2×H100, 200GB, 16 CPUs, 5h,
`ice-gpu`/`coc-ice`. To tag the run, add
`slurm.extra_sbatch_args: ["--job-name=bias-gpt-oss-120b-v1-eager"]` to the YAML.)

Optional pre-flight (diagnostic, ~15 min on 2×H100):

```bash
sbatch --gres=gpu:h100:2 --mem=64G --time=02:00:00 --partition ice-gpu --qos coc-ice \
  --export=ALL,SCRIPT=scripts/diag_gptoss_capture.py,EXTRA_ARGS=EAGER,STUDY_CONFIG=configs/study.gpt-oss-120b.concentration.v1.yaml,WORKDIR=/home/hice1/sghose7/scratch/moe-breakdown,MODULES='gcc/12.3.0 python/3.11 cuda/12.6.1' \
  slurm/run_experiment_script.sbatch
```

## 3. Verify the eager path actually ran

The job log lands in
`/storage/ice1/0/2/sghose7/moe-breakdown-bias-runs/expt-bias-1/exp1-concentration-gpt-oss-120b-v1/slurm-logs/`
(OUT_DIR as exported by the submit script; the `.out`/`.err` names are `<job-name>-<jobid>.out|.err`).

**Eager-path marker (must be present, 36 out of 36 layers):**

```
force_eager_gpt_oss: rebound 36/36 gpt-oss MLP layers
```

per-layer lines also appear: `force_eager_gpt_oss: rebound layer N mlp.forward ... -> eager GptOssMLP.forward`.

```bash
squeue -u sghose7
grep "force_eager_gpt_oss" /storage/ice1/0/2/sghose7/moe-breakdown-bias-runs/expt-bias-1/exp1-concentration-gpt-oss-120b-v1/slurm-logs/*.out
# expect: force_eager_gpt_oss: rebound 36/36 gpt-oss MLP layers
grep "rebound layer" .../slurm-logs/*.out   # expect 36 per-layer lines
grep n_nonzero .../exp1-concentration-gpt-oss-120b-v1/result.json  # expect > 0 (broken v1 had 0)
```

Negative check: a fused-kernel load still shows `transformers/integrations/mxfp4`
in the trace (torch 2.6 `shared_memory_per_block_optin` error) or leaves
`n_nonzero = 0`; the eager run has neither.

For the diag run: `grep "EAGER CAPTURE OK"` in that job's out file.

## 4. Stats pipeline pickup

The study writes `result.json` to
`/storage/ice1/0/2/sghose7/moe-breakdown-bias-runs/expt-bias-1/exp1-concentration-gpt-oss-120b-v1/`
(same slot name as the broken run, so it overrides it — no new dir needed). Pipe it
into the local analysis dir:

```bash
scp -r login-ice.pace.gatech.edu:/storage/ice1/0/2/sghose7/moe-breakdown-bias-runs/expt-bias-1/exp1-concentration-gpt-oss-120b-v1 \
    results/
```

Then `stats_analysis` picks it up automatically by slot name:

| script | mechanism |
|---|---|
| `stats_analysis/scripts/s03_h1_verdict.py` | ladder rows read `results/exp1-concentration-gpt-oss-120b[-v1]/result.json`; prefers `-v1` **unless** it looks broken (`v1_broken`, `H==0` or `n_nonzero==0`). New run → broken flag clears, v1 used, `dH/dG` vs v0 computed. |
| `stats_analysis/scripts/s01_exp1_stability.py` | same `-v1` slot reads; broken-flag logic flips automatically. |
| `stats_analysis/scripts/paper_figures_seaborn.py` | hardcoded list; GPT-OSS entry currently points at `exp1-concentration-gpt-oss-120b` (v0, comment "gpt-oss -v1 capture is broken"). After the rerun lands, flip that tuple entry to `("exp1-concentration-gpt-oss-120b-v1", ...)` so figures/paper use the full 5000-pair run. |
| `stats_analysis/scripts/s04_bootstrap_cis.py` | needs `per_pair_phi*.npy` from a `--save-per-pair-phi` run (submit via `slurm/run_experiment_script.sbatch` with `EXTRA_ARGS="--save-per-pair-phi"` if you want CIs for GPT-OSS). |

Then re-run `stats_analysis/scripts/s03_h1_verdict.py` and
`scripts/paper_figures_seaborn.py`; inspect `stats_analysis/outputs/s03_h1_verdict.json`
for the new `gpt-oss-120b` row (`v1_broken: false`) before quoting any numbers.