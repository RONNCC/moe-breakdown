# Experiment Catalog — moe-expt-bias

## expt-bias-1 — Expert-Level Bias Attribution in MoE LLMs via Shapley Values

**Goal:** Measure whether demographic/social bias in open MoE language models is *concentrated* in a small set of identifiable experts (localizable and fixable) or *diffuse* across the full expert population. Tests three hypotheses: H1 (sparser MoE → more concentrated bias), H2 (MoE more localizable than dense), H0 (null: bias is diffuse in all MoE models).

**Method:** routing_contrast Shapley (φ_e ∝ routing-weight difference × bias payoff) across StereoSet + BBQ + WinoGender benchmarks (400 prompt pairs). Concentration measured via normalized entropy H and Gini coefficient. Also: exact Shapley interactions (Exp3), causal ablation cross-check (Exp4), demographic specificity (Exp5).

**Models run:**
- MoE (Exp1): OLMoE-1B-7B, Phi-3.5-MoE, Mixtral-8x7B, GPT-OSS-120B, Gemma 4 26B, DBRX-instruct
- Dense (Exp2): OLMo-7B, Phi-3.5-mini, Llama-3.1-8B
- Interactions (Exp3): OLMoE, Phi-3.5-MoE, Mixtral-8x7B
- Ablation (Exp4): OLMoE only
- Demographic (Exp5): OLMoE (600 pairs, 66 groups)

**Hardware:** GT ICE cluster (PACE); 1–2× H200 (141 GB/GPU) for large models; 2× A100 80GB for smaller; SLURM with partition=ice-gpu, qos=coc-ice.

**Code:** `expt-bias-1/src/moe_bias_shapley/`
- `runner.py` — main Exp1/Exp2 driver
- `hooks.py` — MoE layer discovery + router hook capture
- `shapley.py` — routing_contrast + dense LOO implementations
- `modeling.py` — model loading (with DBRX/transformers-5.x patches)
- `reporting.py` — concentration metrics (H, Gini, top-fraction)

**Configs:** `expt-bias-1/configs/`
- `study.*.yaml` — per-model SLURM job configs

**Cluster storage:** `~/scratch/moe-breakdown-bias-runs/expt-bias-1/<model>/result.json`

**Key findings (complete as of 2026-07-07):**
- H0 STRONGLY SUPPORTED: all MoE models with detectable bias show H ≈ 0.88–0.92 (near-uniform diffusion)
- H1 REJECTED: no monotone trend across sparsity ladder; GPT-OSS (0.031, H=0.880) more concentrated than OLMoE (0.016, H=0.900) — direct violation
- H2 REJECTED (with method confound caveat): H_dense (0.63–0.76) < H_MoE (0.88–0.92), but dense uses causal LOO vs MoE correlational routing_contrast
- DBRX H=0.919 ≈ Mixtral H=0.917 at same N_A/N=0.25 — architecture-agnostic diffuseness
- Exp3: 70–74% of early-layer bias attribution mass is pairwise interactions, not individual marginals — routing_contrast underestimates diffuseness
- Gemma 4 null: mean_bias_gap ≈ 0, model too well-aligned for benchmarks to detect bias

**Full results and analysis:** `moe-expt-bias/RESEARCH-JOURNAL.md`
**Study design:** `moe-expt-bias/study_design_C1_C4.md`
