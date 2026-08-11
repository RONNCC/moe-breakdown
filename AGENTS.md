# Project Memory

## Tooling preferences
- Use `uv` for all Python environment management (uv venv, uv pip install, uv run).
  Never use `pip install` directly into system or user envs (PEP 668 blocks it).
- Reference venv for figure/analysis scripts: create with
  `uv venv --system-site-packages ...` then `uv pip install --python <venv>/bin/python <pkgs>`.

## Repo map
- `moe-expt-bias/expt-bias-1/` — main study code (src/moe_bias_shapley), run scripts,
  results/*/result.json per study run, stats_analysis/{scripts,outputs,figures,paper}.
- `stats_analysis/scripts/paper_figures_seaborn.py` — regenerates all paper figures from result.json
  (MOE tuple hardcodes result dirs; `-v1`/`-5000` dirs are current, `v0` superseded).
- `moe-expt-bias-2/moe_bias_report_acm_v2.tex` — **the current ACM/TIST draft** (top-level, not
  under expt-bias-1; `\graphicspath` points back into
  `moe-expt-bias/expt-bias-1/stats_analysis/figures/`). Self-contained (embedded
  thebibliography), compiles clean with 2x pdflatex, 11 pages.
- Cluster: `ssh login-ice.pace.gatech.edu` (PACE ICE, slurm, user `sghose7`). VPN must be OFF
  or SSH times out during banner exchange. Run via `submit_slurm_study.py` /
  `submit_slurm_experiment3.py` / `submit_slurm_experiment6.py`.
  Pull artifacts via `scp`/mcp-ssh downloadFile (remote root:
  /home/hice1/sghose7/scratch/moe-breakdown-bias-runs/...).
- `moe-expt-bias/expt-bias-1/CLUSTER-STATUS.md` / `GAP-ANALYSIS.md` /
  `experiment-registry.yaml` — live cluster/job/experiment tracking, updated each session.

## Status (Aug-11-2026)
- H1 (post-correction): gemma N_A corrected 960->240 (N_A/N 0.0625) and GPT-OSS-120B v1
  capture VALIDATED (2000 pairs, H=0.8764, G=0.7274, plus a 5000-pair stability
  replication confirming it). EXACT two-sided permutation p-values used for all ladder
  sets (s03_h1_verdict.py, permutation cap n<=7): all-6 rho_H=0.754 p=0.106,
  minus-gpt-oss-5 rho_H=0.872 p=0.100, paper-valid-4 rho_H=0.949 p=0.167,
  unique-sparsity-3 rho_H=1.000 p=0.333. Verdicts: SUPPORTED (point+drift) in EVERY
  subset, but NO set significant at 0.05 (p>=0.100); Monte Carlo power analysis shows
  n~8-13 rungs needed for 80% power. Frame in paper: directionally consistent,
  under-powered.
- Bootstrap CIs (s04) landed for **all 13 model payloads** (n_boot=5000, seed=42,
  stratified-by-group block bootstrap): 6 MoE ladder rungs (v1), 4 dense baselines (v1),
  GPT-OSS-120B 5000-pair replication, and both Exp8 same-mechanism LOO captures
  (Phi-3.5-MoE H in [0.842,0.941]; OLMoE H in [0.692,0.921], landed 2026-08-11 via a
  cluster-maintenance-window node-pinning workaround, see CLUSTER-STATUS.md). Only
  gemma4-27b is MISSING (placeholder dir, phantom HF repo, non-issue — real rung is
  gemma4-26b).
- Robustness suite beyond the core ladder: Exp3 pairwise-synergy (collectivity check,
  4/6 models DONE, synergy 0.18-0.75 of attribution mass, mechanistic explanation for
  causal-ablation reversals), Exp6 ladder-wide causal ablation (6/7 models DONE,
  mixed verdict — phi-ranking beats controls on OLMoE/Phi-3.5-MoE but loses on
  Mixtral/DBRX), Exp7 proxy-vs-exact-Shapley agreement (3/3 tractable models DONE,
  all null, |rho|<=0.24), Exp8 same-mechanism LOO (2/2 models DONE). DBRX/GPT-OSS-120B
  Exp3 and GPT-OSS-120B Exp6 were RUNNING (jobs 5575799/5575791/5575800) when PACE's
  scheduled quarterly maintenance window closed the cluster on 2026-08-11 06:00
  (through 2026-08-13 23:59) — check `sacct` for their final state once the cluster
  reopens.
- Figures regenerated + VISION-VERIFIED with corrected ladder tuple
  (stats_analysis/scripts/paper_figures_seaborn.py; MOE tuple is (dir,label,N_players,N_active)).
- Per-pair payloads (~360MB+, growing) cannot live on GitHub (>100MB files); published as
  Kaggle dataset `sghose0/moe-bias-routing-shapley-perpair-phi` (note: owner namespace is
  `sghose0`, not `ronncc`).
- GPT-OSS capture gotcha (historical): transformers MXFP4 fused kernel path crashes on
  torch 2.6; fix = bf16-dequantized load so eager GptOssMLP/router is used
  (config `force_eager_moe: true`).
- Cluster scheduling gotcha (new, 2026-08-11): a job stuck `PENDING` with reason
  `ReqNodeNotAvail, Reserved for maintenance` despite idle nodes and no visible
  `scontrol show reservation` entry is very likely a **real, non-admin-visible upcoming
  maintenance reservation** — the scheduler rejects any job whose requested `--time`
  would run past the reservation's start. Workaround: pin to a specific idle node
  (`--nodelist=...`) and binary-search the largest safe `--time` with short `--wrap`
  probe jobs; once genuinely inside the maintenance window, login itself is refused
  with an announcement banner (not just job scheduling) — at that point nothing is
  actionable until the window's announced end time.