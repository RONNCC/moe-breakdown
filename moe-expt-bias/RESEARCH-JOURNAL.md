# Research Journal — MoE Bias Attribution Study (expt-bias-1)
*Last updated: 2026-07-06*

---

## Status snapshot (updated 2026-07-07)

| Experiment | What it tests | Status |
|---|---|---|
| Exp1 — Concentration ladder (OLMoE, Phi-3.5-MoE, Mixtral) | H1: sparsity → concentration | **Done** |
| Exp1 ext — DBRX (top-4/16, N_A/N=0.25) | H1 arch replicate at N_A/N=0.25 | **Running** — job 5483610 (2× H200) |
| Exp1 ext — GPT-OSS-120B (top-4/128, N_A/N=0.031) | H1 ladder gap fill | **Running** — job 5483611 (2× H200) |
| Exp1 ext — Gemma 4 26B (top-8/128, N_A/N=0.063) | H1 ladder, Google MoE | **Ready to submit** — correct ID: `google/gemma-4-26B-A4B-it`; config updated |
| Exp2 — Dense baselines (OLMo-7B, Phi-3.5-mini, Llama-3.1-8B) | H2: MoE more localizable than dense | **Done** |
| Exp3 — Interaction/synergy check (OLMoE, Phi, Mixtral) | C2: marginal vs synergy structure | **Running** — jobs 5483577, 5483584 (Mixtral exp3 completed already) |
| Exp4 — Ablation cross-check (OLMoE, Phi, Mixtral) | Shapley rankings vs causal ablation | **Done** (30 pairs per model) |
| Exp5 — Demographic specificity (OLMoE) | C3: different experts per demographic group | **Done** (600 pairs, 66 groups) |

**Gemma 4 ready to submit:** Correct model ID is `google/gemma-4-26B-A4B-it` (26B total, ~4B active/token, top-8/128). Config updated. Run `python3 scripts/submit_slurm_study.py --config configs/study.gemma4-27b.concentration.yaml` from the ICE login node after pulling.

---

## Full results table (Exp1 + Exp2, routing_contrast / dense LOO)

| Model | Type | N_A/N | n_players | H (norm) | Gini | top-5 frac | top-10% frac | mean_bias_gap |
|---|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | MoE, top-1/64 | 0.016 | 1024 | **0.900** | 0.599 | 0.069 | 0.451 | 0.203 |
| Phi-3.5-MoE | MoE, top-2/16 | 0.125 | 512 | **0.889** | 0.617 | 0.087 | 0.452 | 0.244 |
| Mixtral-8x7B | MoE, top-2/8 | 0.250 | 256 | **0.917** | 0.516 | 0.109 | 0.364 | 0.187 |
| OLMo-7B | dense | 1.0 | 32 | **0.719** | 0.693 | 0.687 | 0.605 | 0.146 |
| Phi-3.5-mini | dense | 1.0 | 32 | **0.758** | 0.613 | 0.625 | 0.528 | 0.187 |
| Llama-3.1-8B | dense | 1.0 | 32 | **0.630** | 0.730 | 0.749 | 0.698 | 0.179 |

H is Shannon entropy normalized by log(N), so H ∈ [0,1] and is comparable across models with different player counts. H=0 is maximally concentrated (one player holds all mass); H=1 is perfectly uniform. Gini is the opposite direction: higher = more concentrated. top-5 frac = fraction of total |φ| mass held by the top-5 players.

---

## Verdict on hypotheses

### H1 — Concentration increases with sparsity (C1, RQ1)

**Result: Weakly and inconsistently supported. Not a clean monotonic relationship.**

The prediction was OLMoE (H lowest) < Phi < Mixtral (H highest). What we observe:

- By H: Phi (**0.889**) < OLMoE (0.900) < Mixtral (0.917)
  → Phi is most concentrated, Mixtral least. OLMoE is in the middle, not the most concentrated.
- By Gini: Phi (**0.617**) > OLMoE (0.599) > Mixtral (0.516)
  → Same story: Phi most concentrated.
- By top-5 fraction: Mixtral (**0.109**) > Phi (0.087) > OLMoE (0.069)
  → **Contradicts H1** — the sparsest model (OLMoE) holds the LEAST mass in its top-5 experts.

Summary: Two out of three pairwise comparisons partially align with H1 (both Phi and OLMoE are "more concentrated" than Mixtral by H and Gini), but the key OLMoE-vs-Phi reversal is unexpected. H1 is NOT cleanly supported. The null (H0: bias attribution is diffuse across all MoE models) is the better description of the data.

**Important caveat on top-fraction metrics:** The top-5 fraction comparison is contaminated by different player counts (1024 vs 512 vs 256). Normalized entropy H/log(N) is the most valid cross-model metric; the top-fraction metrics are mostly meaningful within-model.

### H2 — MoE more localizable than dense (C4, RQ2)

**Result: Clearly rejected across all matched pairs.**

Localizability ratio LR = H_dense / H_MoE. LR > 1 means MoE is more localizable.
- OLMoE vs OLMo-7B: LR = 0.719/0.900 = **0.799** (< 1, wrong direction)
- Phi-3.5-MoE vs Phi-3.5-mini: LR = 0.758/0.889 = **0.853** (< 1, wrong direction)
- Cross-check: 0.630/0.900 = **0.700** (< 1)

Dense models have significantly lower H (0.63–0.76) than MoE models (0.89–0.92). H2 is rejected by the data. Dense attribution is **more concentrated**, not less.

**Critical methodological caveat:** Dense and MoE players are not equivalent units. Dense LOO uses 32 transformer layers as "players" (coarse, each representing billions of parameters). MoE uses 256–1024 individual experts (fine-grained, each representing millions of parameters). Even with H normalized by log(N), the attribution methods are fundamentally different:
- Dense: causal intervention (ablate a layer → measure bias change)
- MoE: routing-weight correlation (φ_e ∝ routing-weight difference between stereo/anti prompts)

This means the H2 result should be interpreted carefully: it may reflect a difference in **method** (causal LOO vs correlational routing_contrast) as much as a difference in **architecture**. This is the dominant threat to validity for H2.

### H0 — Diffuse bias across all MoE models (null)

**Result: Most supported by the data, and still publishable/interesting.**

All MoE models have H ≈ 0.89–0.92 (near-maximum diffusion). Bias attribution spreads across hundreds or thousands of experts nearly uniformly. This is the strongest finding: **MoE routing does NOT concentrate bias into a small committee of identifiable experts**, at least as measured by routing-contrast Shapley.

The null result is interesting precisely because it contradicts the "MoE = more interpretable = more controllable" narrative with fairness evidence. A key implication: if bias attribution is diffuse in MoE models, single-expert debiasing recipes will be ineffective.

---

## Exp4 ablation curves (cross-check quality)

Exp4 tested whether ablating the top-φ experts (ranked by routing_contrast Shapley) causes a steep bias drop, as a causal validation.

Key observations (n=30 pairs each — small, interpret with caution):

| Model | k=1 effect | k=10 effect | k=10% effect |
|---|---|---|---|
| OLMoE | +0.2% (noise) | −6.2% | 102 experts → **−92%** |
| Phi-3.5-MoE | −0.5% | −0.8% | 51 experts → **−37%** |
| Mixtral-8x7B | −0.5% | +3.6% | 26 experts → **+15% (increase!)** |

Critical issues:
1. **n=30 pairs is too small** for the early curve to be meaningful. All k<10 results are dominated by noise.
2. **The large OLMoE drop at 10% looks impressive but involves ablating 102 of 1024 experts** — roughly 6 experts per layer. That is a major intervention, not "a few key bias experts."
3. **Mixtral bias increases when its top experts are ablated**, which is either an interference/compensation effect or evidence that routing_contrast mis-ranked the experts.
4. The ablation curve shapes are NOT the "steep early drop" signature of concentrated attribution (which would show large reductions at k=1–5). Instead they're flat early then gradually decline at high ablation fraction.

**Conclusion:** The Exp4 results are consistent with the H0 (diffuse) interpretation. They do NOT validate the Shapley rankings as identifying causal "bias experts." They also suggest routing_contrast's top-players ranking is weakly predictive of ablation impact. This strengthens the case for running exact Shapley (Exp3 currently running).

---

## Known methodological limitations and confounds

1. **routing_contrast is correlational, not causal.** It measures ∑_e (w_e,stereo − w_e,anti) × bias_gap, not the marginal contribution of each expert to bias. If the model is biased but routing doesn't change between stereo/anti prompt pairs, routing_contrast will assign φ≈0 to all experts regardless of their actual contribution. Exact Shapley (Exp3, currently running) addresses this.

2. **H1 confound: different player-set sizes.** OLMoE has 1024 players (16 layers × 64 experts), Phi has 512 (32×16), Mixtral has 256 (32×8). H is normalized by log(N), which should make it comparable, but the effective granularity differs. With 1024 fine-grained players, routing noise may inflate apparent diffuseness.

3. **H2 confound: different attribution methods.** Dense models use causal LOO (intervention-based), MoE uses routing_contrast (correlation-based). This is a method confound on top of the architecture comparison. To remove it, we would need to run exact Shapley on dense models too (expensive: 2^32 coalitions impractical; would need approximation like RGIS or a bounded subset).

4. **OLMoE top-1 routing edge case.** With top-1 routing, each token activates exactly one expert per layer — a binary assignment. The routing weight is either 1.0 (active) or 0.0 (inactive). routing_contrast for OLMoE is therefore: φ_e = P(e is active | stereo) × bias_gap − P(e is active | anti) × bias_gap. This spreads attribution across all 64 experts proportional to their frequency difference, which is a coarser signal than the exact marginal contribution.

5. **Benchmark composition (StereoSet + BBQ + WinoGender).** These benchmarks are mixed together into one pool. The bias_gap payoff may have different character across benchmarks (StereoSet = cloze, BBQ = QA, WinoGender = coreference). Per-benchmark analysis would reveal whether concentration varies by bias type.

6. **Exp4 sample size.** n=30 pairs for ablation curves is too small. Need at least n=100 for meaningful early-curve significance.

---

## What additional experiments would help validate or reject H1

### High priority (run soon)

**1. Wait for Exp3 (currently running) — exact Shapley interaction check.**
Exp3 computes 2^K ablation Shapley exactly on 2 MoE layers × 20 pairs. This will give:
- Per-layer concentration metrics from exact (not routing_contrast) Shapley
- Synergy fraction: how much of bias is from expert interactions vs individual contributions
- If exact Shapley still shows diffuse H, H0 is much stronger. If concentrated, the routing_contrast method was underestimating concentration.

**2. Per-layer H analysis from existing Exp1 data.**
The Exp1 result.json files have `top_players` ranked by φ. These can be grouped by layer to compute per-layer concentration. It may be that a FEW layers are highly concentrated while the rest are diffuse — the aggregated H hides this. This can be done NOW without new jobs (analysis only).

**3. Re-run Exp4 with n=100+ pairs.**
Current n=30 is too small. Larger samples would give error bars on the ablation curves and determine whether the flat early curve is real or noise.

### Medium priority (new model runs)

**4. DBRX-instruct (databricks/dbrx-instruct).**
- Architecture: 16 experts, top-4 → N_A/N = 4/16 = **0.25** (same sparsity as Mixtral)
- Total 132B params → needs 4× A100-80GB in bf16; or 2×A100 in 4-bit (check torch/bnb compat)
- Value: architectural replicate at N_A/N=0.25. If DBRX shows H similar to Mixtral (~0.917), it validates that the N_A/N=0.25 region genuinely has that level of diffuseness regardless of architecture. If very different, architecture matters more than sparsity.
- Config template: see `configs/study.dbrx.concentration.yaml` (created alongside this journal)

**5. GPT-OSS-120B** (if available on HuggingFace and not blocked on GT ICE).
- Study design mentions "GPT-OSS" as a major deployed MoE. If 120B is the active-param count, it would be very large; if total param count, may be more tractable.
- Need to check: HF availability, routing scheme (N_A/N), license compatibility with GT ICE policy.
- Could be a very high-value addition if it provides a genuinely different point on the sparsity ladder.

**6. Llama-4-Scout** (mentioned in study design as optional).
- 17B active / 109B total, 128 experts → if top-1/128: N_A/N ≈ 0.008 (even sparser than OLMoE!)
- Would be the most extreme point on the H1 sparsity ladder.
- Need to check: HF availability, memory requirements (active is 17B → 34GB bf16 → fits 1 A100).

**7. Gemma 4** (check if MoE).
- Gemma 1/2/3 were dense. Gemma 4 may be MoE (trend in Google's architecture).
- If MoE with a different routing scheme, valuable for architectural diversity.
- Need to check: HF model ID, architecture, routing scheme.

### Longer-term analysis

**8. Exact Shapley on all 3 MoE models for Exp1 (not just interactions).**
Run routing_contrast and exact Shapley on the same 50-pair subset for each MoE model. Compare H from both methods. If exact Shapley shows lower H (more concentrated), it means routing_contrast systematically over-estimates diffuseness. This would rehabilitate H1/H2 as possibilities but would require attributing the current results to method bias.

**9. Per-benchmark split.**
Separate StereoSet / BBQ / WinoGender results. Attribution patterns may differ sharply by benchmark — gender bias (WinoGender) might be more concentrated than racial/religious bias (StereoSet) even in the same model.

**10. Bootstrap confidence intervals.**
Resample the 400 prompt pairs with replacement, recompute H and Gini. This gives 95% CI on all concentration metrics. Currently there are no error bars on any result, making pairwise comparisons uncertain.

---

## Interpretation for the research report

**Most defensible framing (as of 2026-07-06):**

The data supports H0: bias attribution in current open MoE models is **diffuse**, with normalized entropy H ≈ 0.89–0.92 across all three MoE models (OLMoE, Phi-3.5-MoE, Mixtral). No model shows the sharp concentration that would enable single-expert debiasing. This challenges the "MoE modularity → bias localizability" thesis.

The H1 prediction (sparser MoE → more concentrated bias) finds weak/mixed support: the direction is partially right for H and Gini (Mixtral is least concentrated), but the critical OLMoE vs Phi ordering is reversed. The effect size is small (H ranges from 0.889 to 0.917 across all three models).

The H2 comparison (MoE vs dense) is confounded by method (routing_contrast vs causal LOO) and player-set granularity. Dense models show lower H (0.63–0.76), but this likely reflects the method difference as much as genuine architectural difference. This is a validity threat that must be front-and-center in the paper.

**The null result is publishable and scientifically useful:** it corrects a growing assumption ("MoE interpretability advantage transfers to fairness auditing") with empirical evidence, and provides the first quantitative bias-Shapley framework for MoE models.

**Pending before finalizing:** Exp3 (exact Shapley interactions, currently running) may substantially change the H1/H2 picture if exact Shapley shows different concentration than routing_contrast.
