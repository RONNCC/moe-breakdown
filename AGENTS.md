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

## Status (Aug-09-2026)
- H1: pre-correction ladder flat (rho=0.2, p=0.63); CORRECTED ladder (Gemma N_A=240, v1 capture)
  now rho_H=0.81 (exact p=0.058) on all 6, rho_H=0.97 (p=0.017) minus un-certified GPT-OSS v0 row;
  core-4 rho=0.95 (p=0.083); 3-rung rho=1.0 (knife-edge). All p's are exact permutation two-sided.
- Gemma-4-26B v1 DONE (Aug-09): 5000 pairs + per-pair payload in
  results/exp1-concentration-gemma4-26b-v1/ (H=0.7888, G=0.8551, CI in s04 json). Folded into
  Tables 1-2 + figures (tuple flipped). Zero fraction 29% (1132/3840).
- GPT-OSS full v1 run RUNNING as sbatch 5573613 (2000 pairs incl. --save-per-pair-phi, 4xH100
  gpus_per_node:4, --time 4h, started Aug-09 ~08:30 UTC, ETA ~09:45 UTC; 2-GPU job 5573608 cancelled
  at ~125 pairs: bf16 120B (240GB) CPU-offloads on 2xH100 -> 25s/pair -> 14h projected; 4xH100 fits,
  ~2s/pair. 24h/12h requests rejected by slurm: partition ice-gpu cap 16:00:00, QOSMaxGRESMinutesPerJob).
  On completion: pull exp1-concentration-gpt-oss-120b-v1/{result.json,per_pair_phi.npy,
  pair_meta.json,...} into local results/, flip figure tuple gpt-oss-120b->gpt-oss-120b-v1, rerun
  s01->s03->s04 + figures, patch tex Table 1 gpt-oss row / Table 2 CI row / footnotes (#871).
- Bootstrap CIs landed for 5 models (DBRX, OLMoE, Mixtral, Phi, Gemma v1); s04_bootstrap_cis.json
  regenerated; GPT-OSS CI row still "---" pending the rerun.
- GPT-OSS capture: transformers 5.14.1 MXFP4 kernel path crashes on torch 2.6 (shared_memory_per_block_optin);
  fix = load bf16-dequantized so eager GptOssMLP/router is used.