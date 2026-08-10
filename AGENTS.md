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
  (MOE tuple at ~line 34 hardcodes result dirs; flip to `-v1` dirs as captures land).
- `stats_analysis/paper/moe_bias_report_aug8/moe_bias_report_aug8.tex` — current ACM draft
  (tables hand-typed; pdflatex 2-pass; fig:scatter/limits/TODO sections track result status).
- Cluster: `ssh login-ice.pace.gatech.edu` (PACE ICE, slurm). Run via submit_slurm_study.py.
  Pull artifacts with mcp-ssh downloadFile (remote root: /home/hice1/sghose7/scratch/moe-breakdown-bias-runs/...).

## Status (Aug-10-2026)
- H1 (post-correction): gemma N_A corrected 960->240 (N_A/N 0.0625) and GPT-OSS-120B v1
  capture VALIDATED (2000 pairs, H=0.8764, G=0.7274 — old broken zero-phi run superseded).
  EXACT two-sided permutation p-values now used for all ladder sets (s03_h1_verdict.py,
  permutation cap n<=7): all-6 rho_H=0.754 p=0.106, minus-gpt-oss-5 rho_H=0.872 p=0.100,
  paper-ladder-5 = 0.872/0.100, paper-valid-4 rho_H=0.949 p=0.167, unique-sparsity-3 rho_H=1.000
  p=0.333. Verdicts: SUPPORTED (point+drift) in EVERY subset, but NO set significant at 0.05
  (p>=0.100). Frame in paper: directionally consistent, under-powered.
- Bootstrap CIs (s04) landed for all 12 models with per-pair payloads (n_boot=5000, seed=42;
  GPT-OSS H [0.8750,0.8858] G [0.7069,0.7307]; covers v1 + v0 + exp5). Only gemma4-27b MISSING
  (is a placeholder dir; no run).
- Figures regenerated + VISION-VERIFIED with corrected ladder tuple
  (stats_analysis/scripts/paper_figures_seaborn.py; MOE tuple now (dir,label,N_players,N_active)).
- Per-pair payloads (~360MB, 8 captures) cannot live on GitHub (>100MB files); published as Kaggle
  dataset `ronncc/moe-bias-routing-shapley-perpair-phi`. Cluster/GPU/experiment inventory:
  moe-expt-bias/expt-bias-1/CLUSTER-STATUS.md.
- Paper base draft: stats_analysis/paper/moe_bias_report_acm_v2.tex (supersedes aug8/ subdir draft;
  aug8 carries a legacy acmart.cls watermark).
- GPT-OSS capture gotcha (historical): transformers 5.14.1 MXFP4 fused kernel path crashes on torch 2.6;
  fix = bf16-dequantized load so eager GptOssMLP/router is used (config force_eager_moe: true).