# Project Memory

## Tooling preferences
- Use `uv` for all Python environment management (uv venv, uv pip install, uv run).
  Never use `pip install` directly into system or user envs (PEP 668 blocks it).
- Reference venv for figure/analysis scripts: create with
  `uv venv --system-site-packages ...` then `uv pip install --python <venv>/bin/python <pkgs>`.

## Repo map
- `moe-expt-bias/expt-bias-1/` — main study code (src/moe_bias_shapley), run scripts,
  results/*/result.json per study run, stats_analysis/{scripts,outputs,figures,paper}.
- `stats_analysis/scripts/paper_figures_seaborn.py` — regenerates all paper figures from result.json.
- `stats_analysis/paper/moe_bias_report_topic_rev.tex` — current ACM sigconf draft.
- Cluster: `ssh login-ice.pace.gatech.edu` (PACE ICE, slurm). Run via submit_slurm_study.py.

## Status (Aug-2026)
- H1 (sparsity => concentration) NOT supported: entropy flat across ladder (rho=0.2, p=0.63).
- GPT-OSS capture: transformers 5.14.1 MXFP4 kernel path crashes on torch 2.6 (shared_memory_per_block_optin);
  fix = load bf16-dequantized so eager GptOssMLP/router is used.
- DBRX per-pair phi exists; bootstrap CI flights pending for other models.