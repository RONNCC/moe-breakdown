# moe-expt-bias — working memory / handoff notes

Portable notes for `moe-expt-bias/` (the bias-Shapley study, `expt-bias-1/`),
meant to be usable by any AI coding tool or a human picking this up cold.
Last updated: 2026-07-06/07 (post model-ladder policy fix + first full
experiment run).

## What this study is
Post-hoc bias-attribution study via expert-level Shapley values, comparing
concentration/localizability of bias across a sparsity ladder of MoE models
and matched dense baselines. Full design: `study_design_C1_C4.md`. Code +
configs + runbook: `expt-bias-1/` (see `expt-bias-1/study-catalog.txt` for
the experiment index and how to run things).

## Institutional policy constraint (IMPORTANT, non-negotiable)
**Chinese-origin models are prohibited on Georgia Tech ICE systems.** The model
ladder uses only US/EU-origin models (Apache-2.0 / MIT / Llama Community License).
Current compliant ladder:
- MoE: OLMoE-1B-7B (sparsest), GPT-OSS-120B, Gemma 4 26B, Phi-3.5-MoE (mid),
  Mixtral-8x7B (densest in ladder), DBRX-instruct (replicate at N_A/N=0.25)
- Dense: OLMo-7B (primary, matched to OLMoE), Phi-3.5-mini (secondary, matched
  to Phi-3.5-MoE), Llama-3.1-8B (tertiary cross-check),
  Llama-2-7B (cross-generation, job 5484008 running)
If you ever see configs or results for prohibited Chinese-origin models (e.g. from
an old branch or stale scratch dir), delete them — don't leave them alongside the
compliant ones.

## Cluster access (GaTech ICE)
- SSH: `ssh login-ice.pace.gatech.edu` (user `sghose7`). Requires GaTech VPN if
  off-campus; **corp VPN must be OFF** or SSH hangs at "Connection timed out
  during banner exchange" (raw TCP still connects, so it's not obviously a VPN
  issue from the error alone). Sanity check: `ssh login-ice.pace.gatech.edu echo CONNECTED`.
- Use an ssh-agent to avoid repeated passphrase prompts:
  `export SSH_AUTH_SOCK="$HOME/.ssh/ssh-agent.sock"`, start one if needed
  (`ssh-agent -a "$SSH_AUTH_SOCK"`), then `ssh-add ~/.ssh/id_ed25519` (type the
  passphrase yourself — never pass secrets through an AI tool/API).
- Repo on cluster: `~/scratch/moe-breakdown` (symlink to
  `/storage/ice1/0/2/sghose7/moe-breakdown`). The cluster checkout's local
  `HEAD` is on an unrelated/stale commit (long story, unrelated repo
  restructuring) — **don't `git pull`**. Instead:
  `git fetch origin -q && git checkout origin/main -- moe-expt-bias/` to pull
  just this subtree from the latest remote commit. Push from your laptop as
  normal (`git push origin main`), then re-run that checkout command on the
  cluster.
- GPUs: partition `ice-gpu`, qos `coc-ice`. 8 A100s total (4 nodes x 2). Check
  real capacity with `scontrol show partition ice-gpu | grep TRES` (not just
  `sinfo`, which can be misleading).
- `HF_HOME`/`HF_HUB_CACHE` should point at `$HOME/scratch/hf_cache` (persistent,
  not `$TMPDIR`) so multi-GB checkpoints aren't re-downloaded every job.
- `HF_TOKEN` is set in cluster `~/.bashrc`. Having it set does NOT mean every
  gated repo is actually approved for this account — see gotcha below.
- Storage policy: `$TMPDIR` (node-local, ephemeral) for venvs/build caches;
  `~/scratch` for results/datasets/model caches that need to persist across
  jobs. Never dump large stuff directly under `$HOME` (small quota).

## Running studies
```
cd moe-expt-bias/expt-bias-1
# Dry-run (no GPU/model download, just validates config parsing):
python3 scripts/run_bias_study.py --config configs/study.olmoe.concentration.yaml --dry-run
# Submit to Slurm (from the ICE login node):
uv run --with pyyaml python3 scripts/submit_slurm_study.py --config configs/study.olmoe.concentration.yaml
```
The login node's `module load python/3.11` python has **no pyyaml and no pip**
(no user site-packages available) — `submit_slurm_study.py` needs pyyaml just
to parse the YAML locally to build the `sbatch` command. Use
`uv run --with pyyaml python3 ...` (uv lives at `~/.local/bin/uv`) instead of
trying to install anything into the module python.

Results land in `~/scratch/moe-breakdown-bias-runs/expt-bias-1/<study_name>/`
(`result.json`, `phi.npy`, `player_ids.json`, `slurm-logs/`).

**Dense LOO compute constraint.** `dense_loo` costs 2×N_layers + 2 forward
passes per pair (32-layer models → 66 passes/pair). On 1×A100 at ~0.3s/pass
for a 7-8B model, max throughput ≈ 900 pairs/5hr. This is a hard physical
ceiling — it cannot be worked around by changing wall time without hitting the
GPU-minute QOS cap (MaxTRESMinsPerJob = 960 GPU-min). The fix is **horizontal
sharding across multiple 1-GPU jobs**: `submit_slurm_study.py --num-shards 2`
submits two independent jobs that each process half the pair list
(`pairs[i::n]`). Results land as `result_shard{i}of{n}.json` in the same study
directory and are merged post-hoc by globbing. For comparison, `routing_contrast`
costs only 2 passes/pair — 5000 pairs on 1×A100 takes ~50 min.

## Known gotchas / bugs already fixed (don't re-discover these)
1. **StereoSet repo id**: use `McGill-NLP/stereoset` (bare `"stereoset"` no
   longer resolves under current `huggingface_hub`).
2. **StereoSet nested ClassLabel**: `gold_label` is a nested `List(ClassLabel)`
   inside `sentences` — NOT auto-decoded to strings on row access. Comparing
   `label == "stereotype"` directly silently always returns False (no error,
   just 0 loaded pairs). Resolve via
   `ds.features["sentences"]["gold_label"].feature.names[int_value]`.
3. **BBQ (`heegyu/bbq`)**: legacy loading script no longer runs
   ("Dataset scripts are no longer supported"). Load from
   `revision="refs/convert/parquet"`, and select category via
   `data_dir=<category>` (the parquet branch collapses all per-category
   configs into a single "default" config).
4. **OLMoE / newer fused MoE architectures**: don't assume `.gate` is
   `nn.Linear` or `.experts` is `nn.ModuleList`. OLMoE's `OlmoeTopKRouter` has
   its own `.num_experts`/`.top_k`, and `OlmoeExperts` is a fused module (no
   `len()`). Same pattern for Phimoe (see #7).
5. **bitsandbytes quantization**: `load_in_4bit=True` kwarg directly to
   `from_pretrained` fails on current transformers — must wrap in
   `BitsAndBytesConfig(load_in_4bit=...)` and pass as `quantization_config=`.
   (In practice we just don't quantize — 2x A100 + `device_map="auto"` shards
   30-47B bf16 models fine.)
6. **Llama-3.1-8B is HF-gated** and this account's `HF_TOKEN` does NOT have an
   approved access grant (403 "you are not in the authorized list" even with
   the token present/propagated). Fixed by switching the config to
   `NousResearch/Meta-Llama-3.1-8B` (ungated community mirror, identical
   weights, same Llama Community License terms).
7. **Phi-3.5-mini / Phi-3.5-MoE**: use `trust_remote_code: false`, not `true`.
   - Phi-3.5-mini's *custom* modeling code (what `trust_remote_code=true`
     pulls) calls the removed `DynamicCache.from_legacy_cache` API ->
     `AttributeError`. Native `Phi3ForCausalLM` (transformers>=4.41) doesn't
     have this problem.
   - Phi-3.5-MoE's native `phimoe` module avoids mandatory `einops`/
     `flash_attn` deps that the custom modeling code requires.
   - BUT: native `PhimoeSparseMoeBlock` names its router submodule
     `self.router`, not `self.gate` (Mixtral/OLMoE convention). `hooks.py`'s
     generic MoE-block discovery (`_looks_like_moe_block`/`_get_gate`) checks
     both `.gate` and `.router` now — if you see "Discovered 0 MoE layers"
     for some new architecture, it's probably using yet another attribute
     name and needs the same treatment.
8. **Per-group phi arrays must be flattened** in
   `shapley.py::compute_routing_contrast` before saving/metric computation —
   `per_group_phi[group]` stays 2D `(num_layers, max_experts)` while the main
   `phi` gets `.flatten()`ed; forgetting this produces nonsensical Gini/entropy.

## Terminal/tooling flakiness (AI-tool-specific, skip if you're a human)
- A persistent sync SSH terminal sometimes returns completely blank output
  ("Command produced no output") for trivial commands with no error/timeout —
  not a real hang. Don't keep retrying the same terminal; open a **fresh**
  terminal (or async mode) and it works immediately.
- If several consecutive fresh-terminal SSH attempts all fail identically,
  suspect the corp VPN being on / campus VPN being off, not the cluster.

## Status as of 2026-07-07 — MAIN RESULTS COMPLETE; REVIEWER EXTENSIONS RUNNING

Main experiment results complete. Reviewer-response extensions submitted 2026-07-07.
Full results and analysis in `RESEARCH-JOURNAL.md`.

**Exp1 (routing_contrast, 400 pairs) — all 6 models done:**
- OLMoE-1B-7B → H=0.900, Gini=0.599, mean_bias_gap=0.203
- Phi-3.5-MoE → H=0.889, Gini=0.617, mean_bias_gap=0.244
- Mixtral-8x7B → H=0.917, Gini=0.516, mean_bias_gap=0.187
- GPT-OSS-120B → H=0.880, Gini=0.724, mean_bias_gap=0.165
- Gemma 4 26B → H=0.822, Gini=0.821, mean_bias_gap≈0 (null bias signal)
- DBRX-instruct → H=0.919, Gini=0.554, mean_bias_gap=0.187

**Exp2 (dense LOO) — 3 done, 1 pending:**
- OLMo-7B → H=0.719 | Phi-3.5-mini → H=0.758 | Llama-3.1-8B → H=0.630
- Llama-2-7B → pending (job 5484008)

**Exp3** (exact Shapley interactions, OLMoE/Phi/Mixtral): done.
**Exp4** (ablation, OLMoE only, n=30): done.
**Exp5** (demographic, OLMoE, 600 pairs, 66 groups): done.

**Reviewer extensions — all pending (submitted 2026-07-07):**
- Exp6 (ladder-wide ablation + controls): OLMoE n=100 (5483990), Mixtral n=60
  (5483991), Phi-3.5-MoE n=60 (5483992), OLMo-7B n=60 (5483993)
- Exp7 (proxy-vs-causal Spearman ρ on Mixtral, 20 pairs): job 5483995
- Exp8 (same-mechanism MoE LOO — method confound resolution): OLMoE (5483997),
  Phi-3.5-MoE (5483998)

**v1 full-dataset reruns — pending (submitted 2026-07-07):**
- Exp1 v1 (routing_contrast, 5000 pairs): OLMoE (5484028), Phi-3.5-MoE
  (5484029), Mixtral (5484030), DBRX (5484031), GPT-OSS-120B (5484032)
- Exp2 v1 (dense LOO, sharded — 2 jobs each, max_prompts per config is the
  total pool and each shard gets half via `pairs[i::2]`):
  - OLMo-7B: 1800 pairs → shards 5484046/5484047
  - Phi-3.5-mini: 4000 pairs → shards 5484048/5484049
  - Llama-3.1-8B: 1800 pairs → shards 5484050/5484051
  - Llama-2-7B: 1800 pairs → shards 5484052/5484053
- Exp5 v1 (demographic, OLMoE, 5000 pairs): job 5484037
- v1 output files: `result_shard0of2.json` + `result_shard1of2.json`; merge
  by pooling `n_pairs` and concatenating `phi` arrays before recomputing metrics.

**Key findings:** H0 strongly supported. H1/H2 rejected. DBRX H=0.919 ≈ Mixtral
at same sparsity — architecture-agnostic diffuseness. See RESEARCH-JOURNAL.md.

Report draft: `gemini_report_draft.tex` / `report_draft.pdf`. Figures: `figures/`.
Figure generation code: `expt-bias-1/scripts/build_figures.py`.
