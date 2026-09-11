# Literature Context for MoE Non-Localizability Study

> **Purpose**: Condensed, agent-ready summaries of the 8 papers requested (2026-09-11). Each ends with *Implication for this project* — what a NeurIPS reviewer will weaponize and how to pre-empt it.
>
> Core thesis: **Bias in MoE LLMs is non-localizable — even expert surgery or router skewing does not isolate it — shown via complementary Shapley-style attributions.** Every paper is read against that thesis.

---

### 1. Who Does What in Deep Learning? Multidimensional Game-Theoretic Attribution (Dixit et al., arXiv:2506.19732, Jun 2025)

**What it does**: Introduces **Multiperturbation Shapley-value Analysis (MSA)** + **Shapley Modes**. Standard SHAP attributes *inputs* to a scalar; MSA perturbs (lesions) *neural units* in coalitions and returns a full **output-dimensional map per unit** (pixel-wise for GANs, token/logit-wise for LLMs). Approximated via Monte-Carlo over orderings: `φ_i = E_R[ v(S_i(R) ∪ {i}) − v(S_i(R)) ]`. Demonstrated from MLPs to 56B **Mixtral-8×7B** and DCGANs. Findings: (i) regularization concentrates compute into hubs, (ii) **language-specific experts emerge inside Mixtral**, (iii) inverted generative hierarchy in GANs. Open-source package.

**Methods to steal**: Exact `v(S)` definition via lesioning; Shapley Modes (don't average over tokens/outputs before attribution); Monte-Carlo Shapley for large `N`; interaction-aware analysis.

**Implication**: Your `compute_routing_contrast` (`Δrouting_weight × whole-gap`) is **not** Shapley — it never evaluates `v(S)` and cannot claim efficiency/symmetry. Rename to *routing-contrast heuristic* everywhere; reserve "Shapley" for lesion-based Exp3/Exp7. Crucially, MSA **does find localization for language** in the same model you find bias diffuse. Frame as *bias-specific* diffuseness, not "MoE never specializes" — otherwise reviewer cites MSA as counterexample. Propose **per-layer/per-token Shapley Modes** on a 50-pair subset to show averaging didn't hide concentration.

---

### 2. Beyond Feature Attribution (Shrey Dixit, Hamburg MSc Thesis, edoc 292, Feb 2025)

**What it does**: Full thesis behind (1). Adds: (i) **large weights ≠ high Shapley contribution** without regularization, (ii) regularization (L1/L2/dropout, load-balancing) concentrates computation, (iii) synthetic interaction analysis, (iv) scaling to Mixtral-8×7B showing redundant experts and language/knowledge/arithmetic experts, (v) inverted hierarchy in GAN.

**Implication**: Your `routing_freq` control is expected to mimic `phi` if `phi` tracks frequency — thesis predicts it. More importantly, the ladder entangles `k/N` with **regularization / auxiliary loss / dataset**. Without controlling load-balance loss, you cannot attribute `H` differences to sparsity. **New experiment**: tabulate auxiliary-loss coefficients + router entropy per model, correlate with `H` (0 GPU). Do not claim causal sparsity effect from 6 heterogeneous models.

---

### 3. Explainable Multilingual NMT with Adapters and MoE: Indic Languages (Nath, Int J Speech Tech 2026, May 2026)

**What it does**: Transformer + **language-conditioned adapters** + sparse MoE for Assamese/Bodo/Khasi/Manipuri/Mizo/Nepali (Indo-Aryan, Tibeto-Burman, Austroasiatic). Trained with back-translation; BLEU 29.1 Assamese, 26.1 Nepali. Diagnostics: **attention viz + SHAP + LIME token- and layer-wise**. Findings: adapters preserve family-specific specialization while sharing parameters; explainability tools diagnose failure modes for Khasi/Mizo. Follow-up **AGER-MNMT** feeds token-level attribution into routing (**Attribution-Guided Expert Router**) to stabilize utilization.

**Implication**: MoE *can* localize **when conditioned** on an explicit semantic signal (language/family). Your generic MoE diffuse result is *strengthened* by this positive control — cite it as such. Missing in your study: token/layer attribution workflow (you aggregate), and any alternative mitigation beyond pruning. Propose **conditioned-router control** (e.g., attribute-guided router or demographic prefix) to test if bias localizes when explicitly routed — if still diffuse, thesis is much stronger. AGER is the alternative mitigation to discuss vs. surgery.

---

### 4. FEAMOE: Fair, Explainable and Adaptive Mixture of Experts (Sharma et al., arXiv:2210.04995, Oct 2022)

**What it does**: MoE of **linear experts** with fairness constraints (demographic parity / equalized odds variants) + adaptive gating handling **drift in both accuracy and fairness** on HMDA mortgage streaming data. Shows: mixture-of-linear stays competitive with DNNs while fairer, drift in fairness occurs even when accuracy stable, and Shapley explanations are fast for linear experts.

**Implication**: Fairness is **temporal and definition-dependent** — snapshot `H` ≠ fairness. Your `gap = logp(stereo)−logp(anti)` is never mapped to a formal fairness criterion (demographic parity, equalized odds). Reviewer will ask about drift: does `H` vary across checkpoints/shards? Report **stability across shards** and map `gap` to Gallegos/Blodgett taxonomies. Discuss FEAMOE-style fair routing (constraint on gate) as alternative mitigation to ablation — shows MoE fairness is achievable, just not via pruning.

---

### 5. Stability-aware Shapley-guided MoE for Event-aligned EEG Anomaly Early Warning (Pattern Recognition, 2026, SciDirect S0031320326017243)

**What it does** (inferred from title + EEG Shapley literature; paywalled): Uses **stability-aware feature selection** to guide MoE gating for event-aligned EEG anomaly detection; Shapley scores channel/time stability, MoE aggregates.

**Implication**: "Stability-aware" is exactly what Exp5 lacks. You report mean `D_JS=0.22` across 85 cohorts with uneven `n`, no shrinkage. Lesson: **filter to stable experts/cohorts before claiming subgroup specificity**. Implement **stability-weighted JS** (bootstrap per-cohort `phi` stability, leave-one-cohort-out pooled reference, label permutation within strata) — otherwise `D_JS` is routing-structure heterogeneity, not demographic specialization. Your `s11_sanity.py` already codes the needed controls.

---

### 6. CALM: Culturally Self-Aware Language Models (Shen et al., NeurIPS 2025)

**What it does**: Endows LLMs with **cultural self-awareness**: (i) disentangles task semantics from explicit cultural concepts + latent signals into contrastive clusters, (ii) cross-attention alignment, (iii) **culture-specific MoE** along communicative dimensions, (iv) residual fusion + self-prompted reflective correction. Beats SOTA on cross-cultural commonsense/value/hate benchmarks. Culture as internal adaptive state.

**Implication**: Strongest parallel to your RQ3. **Experts can encode culture *if* architecture explicitly disentangles and routes per culture** — generic MoE diffuse vs. CALM specialized is the "conditioned vs. generic" contrast. Highlights your construct is **US-centric, binary, English-only, intrinsic-logprob**. Scope claims to "US-centric intrinsic stereotype gap" and report **per-cultural-dimension `H`** (e.g., BBQ religion vs. gender). Cite CALM as future architecture that *might* localize cultural bias if explicitly routed.

---

### 7. What Really Matters for Person Re-Identification? A Mixture-of-Experts Framework for Semantic Attribute Importance (Psalta et al., arXiv:2512.08697, Dec 2025)

**What it does**: **MoSAIC-ReID** — Transformer ReID with **LoRA experts each aligned to one semantic attribute** (clothing color, backpack, hat…), **oracle router** enabling controlled attribution, plus GLMs/statistical tests/feature-importance to quantify attribute importance. Finding: upper/lower clothing colors dominate; infrequent accessories (hat) have limited effect; oracle routing + ablation gives *causal* importance, not correlation.

**Implication**: **Methodological gold standard for your claim**. To argue "bias not localizable," first show you *can* localize *something* when experts are semantically aligned. MoSAIC's scaffold (expert→attribute + oracle router + ablation + GLM) should be replicated for bias: even when experts are forced per demographic attribute with oracle routing, does bias stay diffuse? That's the decisive test your ladder alone cannot provide. Also copy their statistical layering (not just Spearman `ρ`) and note their "rare cue looks unimportant" lesson — mirrors your `t5` artifact when `N` large.

---

### 8. Scalable and Interpretable Mixture of Experts Models: Foundations, Applications, and Challenges (Survey, Preprints 202507.0283, Jul 2025)

**What it does**: Rigorous survey of MoE foundations, optimization, generalization, **attribution methods leveraging modular structure**, quantitative interpretability metrics, applications (NLP/CV/RL/healthcare), trustworthiness challenges.

**Implication**: Provides **taxonomy to structure your Related Work** (foundations / optimization / attribution-via-modularity / metrics). Use it to justify `H/G/t10pct/exp(H)` choices vs. alternatives (attention rollout, ROAR), and to checklist optimization assumptions (load-balance loss, shared experts, quantization). Elevates gaps around `k/N` semantics, shared experts, and metric incomparability from nitpicks to standard checklist items.

---

### Cross-Paper Meta-Lesson (read this before writing the intro)

Three independent lines **converge**:
- MSA finds **language specialization** in same MoE family you study.
- Nath/AGER and CALM show MoE **does localize when explicitly conditioned/routed** per semantic/cultural signal.
- FEAMOE/MoSAIC show **fairness/attribute importance is measurable with MoE when formalized**.

Therefore your strongest, reviewer-proof contribution is **not** "MoE never localizes" but:

> **"Generic off-the-shelf MoE shows diffuse, synergy-dominated, non-causal bias attribution; language/family/culture *can* localize when explicitly routed, so bias diffuseness is a property of social-stereotype encoding, not proof that MoE never specializes; mitigation must be routing-level, not expert-surgery-level."**

Frame every one of these papers as *supporting contrast*, not hostile counterexample, and your negative result becomes a methods-caution paper — exactly what NeurIPS/ICML now rewards.

---

### Minimal Citation Checklist for Resubmission

Add to Related Work (+16 total):
- MSA (Dixit et al. 2506.19732) + Thesis Hamburg 292
- Nath 2026 IJST + AGER-MNMT preprint
- Sharma FEAMOE 2210.04995
- EEG stability-aware Shapley-MoE (PR 2026)
- CALM NeurIPS 2025 (Shen et al.)
- Psalta MoSAIC-ReID 2512.08697
- Interpretable MoE Survey 202507.0283
- Plus 8 already-flagged missing: Covert JMLR 2022, Sundararajan ICML 2017, Lundberg 2018 interaction, Zhou NeurIPS 2022 (Expert Choice), Nangia 2020 (CrowS-Pairs), Zhao 2018 (WinoGender/BLS), Gallegos 2024, Frantar 2023 (SparseGPT)
- Label arXiv/preprints as such; keep peer-reviewed vs. preprint distinct.
