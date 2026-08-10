# ICE Cluster Status & Experiment Tracking (expt-bias-1)

Last updated: 2026-08-10 by the revision agent (all facts observed via `ssh login-ice.pace.gatech.edu` + local `results/` this session).

## GPU Inventory (sinfo -p ice-gpu, 2026-08-10)

Partition `ice-gpu`, ALL state `up`, TIMELIMIT `16:00:00`.

| GPU type | nodes | GPUs/node | notes |
|---|---|---|---|
| H100 | 6 | 8 | biggest pool; headroom after study jobs |
| H200 | 6 | 8 | used by DBRX (2xH200 fits 132B bf16) |
| L40S | 4 | 8 | OLMoE-class runs |
| A100 | 4 | 2 | max 2/node (160GB) — DBRX/GPT-OSS need H200 instead |
| A40 | 2 | 2 | — |
| RTX_6000 | 2 | 4 | — |
| MI210 | 2 | 2 | — |
| V100 | 11 | 2 | slowest; use only for tiny runs |

QOS `coc-ice` GPU-minute cap: MaxTRESMinsPerJob = gres/gpu=960 (i.e. 2 GPUs x 8h or 4 x 4h). Submissions above that are rejected (observed: 24h/12h requests rejected; 4xH100 4h accepted at ~2s/pair).

## Queue (2026-08-10, `squeue -u sghose7`)

- **RUNNING — sbatch 5575070**: GPT-OSS-120B exp1 re-run at 5000 pairs (ladder uniformity: this rung was previously the only one at 2000). 4xH100 (--gres=gpu:h100:4), walltime 04:00:00, per-pair phi saved,
  ETA ~2.7h (~1.9 s/pair). Config: `configs/study.gpt-oss-120b.concentration.v2.yaml`
  (study_name exp1-concentration-gpt-oss-120b-5000). New result dir on cluster:
  exp1-concentration-gpt-oss-120b-5000 (v1 capture untouched).
- On completion: pull result.json + per_pair_phi.npy into local `results/exp1-concentration-gpt-oss-120b-5000/`,
  rerun s03/s04 + figures (n_pairs 5000 now uniform across rungs), and extend the H1 CI/power story
  (2000->5000 pairs narrows the bootstrap CI but does not change n<=6 permutation-power limits).
- All prior study jobs aside from 5575070 are finished; queue otherwise empty.

## Experimental status

### Captures complete & synced locally (results/)

All of these have result.json + per_pair_phi.npy locally (bootstrap CIs computable):

| exp dir | model | pairs | per-pair phis | CI available |
|---|---|---|---|---|
| exp1-concentration-olmoe-1b-7b-v1 | OLMoE-1B-7B | 5000 | yes | yes |
| exp1-concentration-phi3.5-moe-v1 | Phi-3.5-MoE | 5000 | yes | yes |
| exp1-concentration-mixtral-8x7b-v1 | Mixtral-8x7B | 5000 | yes | yes |
| exp1-concentration-dbrx-v1 | DBRX-132B | 5000 | yes | yes |
| exp1-concentration-gemma4-26b-v1 | Gemma-4-26B | 5000 | yes | yes |
| exp1-concentration-gpt-oss-120b-v1 | GPT-OSS-120B | 2000 | yes | yes — VALID (old broken zero-phi capture superseded) |
| exp1-concentration-gemma4-26b | Gemma (v0) | 400 | yes | yes |
| exp5-demographic-specificity-olmoe-1b-7b-v1 | OLMoE demographics | 5000 | yes | yes |

Dense baselines complete (no per-pair payloads — LOO method, N=32 layers): exp2-dense-baseline-{olmo-7b,phi3.5-mini}-v1, exp2-dense-crosscheck-{llama-2-7b,llama-3.1-8b}-v1.

### Missing / pending

- **gemma4-27b**: result dir exists but no result.json / no per-pair phi. Resolved as non-issue: the Gemma-4-26B row uses google/gemma-4-26B-A4B-it (30 layers x 8 experts = N_A=240). No action needed unless a second Gemma rung is wanted.
- exp8-lloo (layer-LLOO on MoE, Exp8 same-mechanism check): dirs exist under results/ with result.json (no per-pair). Not needed for v2.
- No currently queued GPU jobs — any new experiment (e.g. a GPT-OSS re-extension to 5000 pairs, or extra rungs) must be submitted.

## How to submit a new run

```bash
cd moe-expt-bias/expt-bias-1
python3 scripts/submit_slurm_study.py --config configs/<study>.yaml   # from a login node
```

Resource facts: GPT-OSS-120B bf16 needs 2xH100 (4xH100 ~2s/pair; 2xH100 CPU-offloads at ~25s/pair — too slow). Large-MoE bf16 fits best in H200x2. QOS cap 960 GPU-min/job.

## Kaggle data release

Per-pair phi payloads (totalling ~360MB, 8 captures) are published as Kaggle dataset `sghose0/moe-bias-routing-shapley-perpair-phi` (see repo README/data note). GitHub rejects >100MB files, so these payloads ship via Kaggle; the repo carries result.json + pair_meta.json + phi.npy aggregates instead.
