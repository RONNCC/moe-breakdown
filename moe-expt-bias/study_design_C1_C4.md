# Study Design: Is Bias in Mixture-of-Experts Concentrated and More Localizable than in Dense LLMs?
### A post-hoc bias-attribution study via expert-level Shapley values, with a sparsity-scaling and MoE-vs-dense analysis
*Working paper / study design — July 2026. Builds on the "localizability thesis" for MoE bias (claims C1 + C4).*

---

## PAGE 1 — Motivation, Questions, and Hypotheses

### 1.1 Why this matters
Mixture-of-Experts (MoE) is now the dominant scaling architecture (OLMoE, GPT-OSS, Mixtral, Phi-3.5-MoE, DBRX). Its entire value proposition is **modularity**: sparse routing to specialized experts. A fast-growing interpretability literature argues this modularity makes MoE *more interpretable* than dense models — experts are less polysemantic ("modular monosemanticity"), and the *expert* is a cleaner unit of analysis than the neuron [Expert Strikes Back, arXiv 2604.02178; Sparsity & Superposition in MoE, OpenReview bZqopmfZDE]. 

But a central fairness question has gone unasked: **does MoE's modularity make demographic/social bias more *localizable* — and therefore more *fixable* — than in dense models, or does it merely *concentrate* bias into a few experts while the rest stay clean (a fairness risk if those experts are triggered by demographic cues)?** This is the localizability thesis. If true, it reframes debiasing: edit a handful of experts instead of retraining. If false, it corrects the "MoE is more interpretable" narrative with a fairness caveat.

### 1.2 What already exists (and what does not)
The *method* of attributing behaviour to experts in MoE is established, but **not for bias**:
- **Shapley-MoE** [OpenReview 7kQjbCQwtT] computes expert-Shapley for MoE with payoff = 1/PPL, for *pruning* (router-guided Monte-Carlo). No bias payoff.
- **Knowledge Localization in MoE LLMs** [arXiv 2603.17102, Mar 2026] localizes *factual knowledge* to experts via contrastive router-logit analysis + causal ablation. Construct = knowledge, not bias.
- **Dissecting Bias in LLMs** [arXiv 2506.05166, 2025] localizes bias circuits via EAP in **dense** GPT-2/Llama-2. The MoE counterpart is open.
- **DeM-MoE** [arXiv 2508.02853 / ACL 2026] shows experts align with demographic subgroups — but in a *trained* model for *annotation disagreement*, not social-stereotype bias in deployed MoEs.
- **Can MoE Surpass Dense LLMs** [arXiv 2506.12119, ICLR 2026] gives a careful MoE-vs-dense comparison *methodology* for performance — not fairness.

**Gap:** no post-hoc *bias/fairness* attribution to experts/routing in *open* MoE LLMs, and no MoE-vs-dense *localizability* comparison. This design fills both.

### 1.3 Research questions — spine + extensions
**Spine (one metric — attribution concentration/localizability):**
- **RQ1 (C1 — concentration + sparsity scaling):** Across MoE models of increasing routing sparsity, does bias attribution become more *concentrated* in fewer experts? I.e., does sparsity → concentration?
- **RQ2 (C4 — MoE vs dense localizability):** At matched capability, is bias more *localizable* (lower entropy of attribution) in MoE than in a dense model?

**Extensions (mechanism & demographics — round out the "how is bias organized in MoE" story):**
- **RQ2b (C2 — collectivity):** Is concentrated bias a *property of individual experts* or of *expert coalitions/committees* (synergy)? — explains *why* concentration arises.
- **RQ3 (C3 — demographic specificity):** Across demographic cohorts (he/she; Black/White/Asian name contexts), do *different experts* fire, or does the *same expert flip* its contribution? I.e., is bias subgroup-specific at the expert level?

### 1.4 Hypotheses
- **H1:** Bias-Shapley concentration is *monotonically increasing* as routing sparsity (N_A/N) *decreases* — OLMoE (top-1, N_A/N≈0.06) > Phi-3.5-MoE (~0.15) > Mixtral-8x7B (top-2, 0.25). Motivated by [Expert Strikes Back], where monosemanticity scales with sparsity.
- **H2:** An MoE (OLMoE-1B-7B) shows *more localizable* bias than its dense sibling (OLMo-7B) at matched-or-better capability — the modularity that aids interpretability also aids bias attribution.
- **H0 (null, also publishable):** Bias attribution is *diffuse and stable* across sparsity/density — correcting the localizability hype.

---

## PAGE 2 — Method: Models, Data, Metric, and the Shapley Formulation

### 2.1 Model ladder (sparsity regimes)
*Institutional note: All models on the ladder are from US/EU research groups (Apache-2.0 / MIT / Llama Community License) in compliance with Georgia Tech ICE system policies.*

| Model | License / Origin | Active/Total | N_A/N | Role |
|---|---|---|---|---|
| OLMoE-1B-7B | Apache-2.0 / AllenAI | 1B / 7B | 0.016 (top-1/64) | sparsest MoE |
| GPT-OSS-120B *(added)* | Apache-2.0 / Microsoft | ~30B / 120B | 0.031 (top-4/128) | ladder gap fill |
| Gemma 4 26B *(added)* | Gemma License / Google | ~4B / 26B | 0.063 (top-8/128) | ladder gap fill |
| Phi-3.5-MoE | MIT / Microsoft | 6.6B / 42B | 0.125 (top-2/16) | mid-sparsity MoE |
| Mixtral-8x7B | Apache-2.0 / Mistral | 13B / 47B | 0.250 (top-2/8) | densest MoE in ladder |
| DBRX-instruct *(added)* | Databricks Open Model | 36B / 132B | 0.250 (top-4/16) | arch. replicate at N_A/N=0.25 |
| Llama-4-Scout *(optional, not run)* | Llama Community / Meta | 17B / 109B | ≈0.008 (top-1/128) | extreme-sparsity extension |
| OLMo-7B (dense) | Apache-2.0 / AllenAI | 7B / 7B | 1.0 | **dense baseline (same family as OLMoE)** |
| Phi-3.5 (dense, 3.8B) | MIT / Microsoft | 3.8B / 3.8B | 1.0 | dense sibling of Phi-3.5-MoE |
| Llama-3.1-8B (dense) | Llama Community / Meta | 8B / 8B | 1.0 | dense cross-check |

The **OLMoE-1B-7B vs OLMo-7B dense** pair is the primary matched comparison: same family, same training data, only the architecture differs [Expert Strikes Back uses exactly this pair]. This controls for data/capability confounds — the cleanest possible MoE-vs-dense test. The **Phi-3.5-MoE vs Phi-3.5-dense** pair is a secondary matched comparison. The ladder spans N_A/N ≈ 0.06 → 0.16 → 0.25 → 1.0, covering the sparsity→density axis needed for C1.

### 2.2 Benchmarks (bias signals)
- **StereoSet** [Nadeem et al. 2020, arXiv 2004.09456] — stereotype vs anti-stereotype intrasentence/intersentence logit gaps (gender, race, religion, profession).
- **BBQ** [Parrish et al. 2022, arXiv 2110.08193] — ambiguous/neutral social-group QA; measures biased-by-default vs unknown.
- **WinoGender** [Rudinger et al. 2018, arXiv 1804.09301] — coreference gender stereotypes.
- **C-Eval fairness subset** [Huang et al. 2023, arXiv 2305.08322] — for Chinese-language demographic bias.

### 2.3 Bias payoff (scalar V)
For a prompt set P, define the **bias score** B(P) = mean over P of the stereotype-minus-anti-stereotype logit gap (StereoSet SS-style), or the group-conditional logit difference for counterfactual prompt pairs (e.g., "The nurse is a ___" vs "The engineer is a ___"). Higher = more biased. This is the scalar payoff for the cooperative game.

### 2.4 Expert-Shapley formulation
Treat the active experts in a layer (or across layers for a token) as players in a cooperative game Γ = (E, V), where E is the set of *active* experts and V(S) = B(P) when the model is restricted to expert subset S (all other active experts zero-ablated for that layer/token). The Shapley value of expert e:

φ_e = Σ_{S ⊆ E\{e}} [ |S|! (|E|−|S|−1)! / |E|! ] · ( V(S∪{e}) − V(S) )

**MoE-specific efficiency win:** because only top-K experts are active per layer, |E| = K (e.g., K=1 for OLMoE → trivial; K=2 for Mixtral → 4 coalitions; K=8 → 256). We compute **near-exact Shapley over the active expert set** (2^K coalitions) per token/layer, then aggregate φ_e over all tokens where e is active. This is a genuine MoE-only capability — dense models have no analogous small player set. For models where we also want all-N expert attribution, we use **router-guided importance sampling (RGIS)** from Shapley-MoE [OpenReview 7kQjbCQwtT].

The **efficiency axiom** of Shapley (Σ_e φ_e = V(full) − V(∅)) means the φ_e form a *partition* of total bias — so "expert e carries X% of the bias" is a principled, auditable number (unlike IG baselines, which are path-/baseline-dependent; see FoolSHAP attacks [arXiv 2505.08345] and the 2026 LLM-bias survey [arXiv 2411.10915]).

### 2.5 Concentration & localizability metrics
From the aggregated expert bias-Shapley vector φ over all experts:
- **Normalized entropy** H = −Σ (|φ̂_i| log |φ̂_i|), φ̂ = |φ|/Σ|φ|. *Lower = more concentrated.*
- **Gini coefficient** of |φ|.
- **Top-fraction** = share of Σ|φ| held by the top 10% / top-5 experts.
- **Localizability ratio** LR = H_dense / H_moE (LR > 1 ⇒ MoE more localizable).

### 2.6 MoE-vs-dense comparison protocol
Follow [arXiv 2506.12119]: compare under *matched* total parameters, training compute, and data where possible; report capability (MMLU, perplexity) alongside to confirm "matched-or-better." Primary pair = OLMoE-1B-7B vs OLMo-7B (same family).

---

## PAGE 3 — Experiments, Expected Results, and Figures

### 3.1 Experiment 1 — Concentration vs sparsity (RQ1 / C1)
Compute H and top-fraction per model across StereoSet+BBQ+WinoGender. **Expected (H1):** monotonic decrease in H as N_A/N decreases — OLMoE most concentrated, Mixtral-8x7B least. *Figure 2.*

### 3.2 Experiment 2 — MoE vs dense localizability (RQ2 / C4)
Compute H for OLMoE-1B-7B vs OLMo-7B (and Phi-3.5-MoE vs Phi-3.5-dense). **Expected (H2):** H_MoE < H_dense ⇒ LR > 1. *Figure 3.*

### 3.3 Experiment 3 — Collectivity check (C2-lite, strengthens the story)
Compute **Shapley interaction values** [cf. RealExp, SciDirect S0306457325000949] to split bias into *marginal* vs *synergy* components. Tests whether bias is an *expert-committee* (standing-committees) effect [arXiv 2601.03425]. If synergy dominates, debiasing must target committees, not single experts.

### 3.4 Experiment 4 — Independent cross-check (robustness)
Re-derive "biased experts" using **Knowledge Localization's** contrastive + ablation method [arXiv 2603.17102]: build stereotype-success vs failure router-logit buckets (Mann-Whitney U), ablate the top experts, measure disparity drop. If the experts Shapley flags ≈ the experts the contrastive method finds, the Shapley numbers are validated against an independent causal method. *Figure 4 (ablation curve).*

### 3.5 Experiment 5 — Demographic specificity (C3)
Compute the expert bias-Shapley vector separately for each demographic cohort (he/she prompts; Black/White/Asian name contexts from BBQ/StereoSet), then measure the **divergence of the Shapley distributions** across cohorts (Jensen–Shannon divergence, or a `group_difference_plot` lifted inside the MoE [SHAP fairness doc]). This answers RQ3: *different experts fire per group* vs *same expert flips*. **Differentiation from DeM-MoE** [arXiv 2508.02853]: DeM-MoE studies *annotation disagreement* in a *trained* model via KL on *routing distributions*; we study *social stereotype bias* in *deployed open MoEs* via *Shapley attribution* over *counterfactual bias payoffs* — a different construct, model class, and method. *Expected:* bias is subgroup-specific at the expert level (consistent with DeM-MoE's subgroup-specialization finding), yielding a **demographic disparity-in-attribution** metric that complements output-level fairness scores.

### 3.6 Expected figures
- **Fig 1** — schematic: sparsity ladder + the cooperative-game formulation (players = active experts, payoff = bias score).
- **Fig 2** — concentration (H, y) vs N_A/N (x) across the ladder; monotonic if H1 holds.
- **Fig 3** — localizability bars: H for MoE vs dense siblings (OLMoE vs OLMo-7B; Phi-3.5-MoE vs Phi-3.5-dense); LR annotated.
- **Fig 4** — expert-ablation curve: x = fraction of top-biased experts ablated, y = disparity reduction (StereoSet/BBQ). Steep early drop ⇒ concentration ⇒ validates Shapley ranking. Cross-check line from Exp 4 overlaid.

### 3.6 Connection to safety (optional framing)
If bias is route-concentrated, it dovetails with **Sparse Safety / Unsafe Routes** [OpenReview JRtldw5Mpw]: routing is a controllable surface for *both* safety and bias. A router-sensitivity analysis (does the router itself encode demographic cues?) ties the two together.

---

## PAGE 4 — Validation, Risks, Impact, and Plan

### 4.1 Robustness must-haves (to survive review)
- **Counterfactual baselines, not random backgrounds** — FoolSHAP [arXiv 2505.08345] shows background choice hides bias; use counterfactual prompt pairs.
- **Paraphrase stability** — report variance of φ_e across prompt rephrasings (SHAP is unstable in billion-param models [arXiv 2411.10915]).
- **Causal corroboration** — ablation (Exp 4) is the causal check on the correlational Shapley ranking.
- **Multi-benchmark** — StereoSet + BBQ + WinoGender + C-Eval; report consistency.

### 4.2 Threats to validity & mitigations
| Threat | Mitigation |
|---|---|
| "Expert as unit" is contested (polysemous experts) | Restrict primary claims to sparsest models (OLMoE, Phi-3.5-MoE) where [Expert Strikes Back] shows monosemanticity holds; report per-model. |
| Routing marginalization (top-K restricts player set) | Also run router-sensitivity analysis; optionally all-N Shapley via RGIS [OpenReview 7kQjbCQwtT]. |
| Small-model external validity | Include Llama-4-Scout (large, sparse, optional if compute allows) and Llama-3.1-8B (dense cross-check). |
| Causal vs correlational | Ablation (Exp 4) + Shapley interaction (Exp 3) jointly. |

### 4.3 Contribution & broader impact
- **Scientific:** first empirical test of whether MoE's modularity makes bias *localizable/fixable* vs *concentrated/risky* — a fairness dimension of the sparsity debate absent from efficiency/superposition work.
- **Methodological:** first bias (not perplexity/knowledge) expert-Shapley; demonstrates that MoE's top-K sparsity enables *exact, tractable* component-level attribution impossible in dense models.
- **Practical:** if H1/H2 hold, a cheap debiasing recipe (edit/steer a few top-φ experts) vs dense retraining; directly relevant to audits (SHAP `group_difference_plot` fairness tooling).
- **Demographic lens (C3):** a *disparity-in-attribution* metric — which experts shift contribution across groups — gives auditors a locatable target and tests DeM-MoE's subgroup-specialization claim on *social* (not annotation) bias in deployed MoEs.
- **Negative result is also valuable:** if bias is diffuse/stable, it corrects the "MoE is more interpretable ⇒ more controllable" assumption.

### 4.4 Compute & timeline (≈3–4 months, 1–2 A100-80GB)
- **Month 1:** TransformerLens/EasyTransformer hooks for OLMoE + Phi-3.5-MoE (+ Mixtral) routers; reproduce baseline bias scores (StereoSet/BBQ/WinoGender).
- **Month 2:** implement expert-Shapley (top-K exact + RGIS); run C1 across the ladder.
- **Month 3:** C4 (OLMoE vs OLMo-7B; Phi-3.5-MoE vs Phi-3.5-dense); C2-lite interaction values; Exp 4 cross-check.
- **Month 4:** robustness sweeps, writing, figures.
- OLMoE-1B-7B, Phi-3.5-MoE, and Mixtral-8x7B all fit on 1–2 A100-80GB (only active params computed per token); Llama-4-Scout (optional) needs multi-GPU or 4-bit quantization. Per-token Shapley is cheap (2^K coalitions, K small). Verify TransformerLens / EasyTransformer router hooks for each model (OLMoE, Mixtral supported; Phi-3.5-MoE and Llama-4 may require HF-level manual router capture).

### 4.5 Future work
- Extend to the **global-workspace / J-space** locus [Transformer Circuits, July 2026] (deferred per scope).
- Multilingual and multimodal bias attribution; router-level demographic-cue analysis; downstream debiasing (expert steering) as a follow-up paper.

---

---

## PAGE 5 — Actual Results Summary (Post-Study, 2026-07-07)

*This section was added after data collection completed. All results and discussion are in `RESEARCH-JOURNAL.md`.*

### Hypothesis verdicts (final)

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: sparser → more concentrated | **REJECTED** | GPT-OSS (N_A/N=0.031, H=0.880) is more concentrated than OLMoE (0.016, H=0.900) — direct counter-example. No monotone trend on full ladder. |
| H2: MoE more localizable than dense | **REJECTED** (method confound) | H_dense (0.63–0.76) < H_MoE (0.88–0.92) but dense uses causal LOO vs MoE correlational routing_contrast — method difference confounds interpretation. |
| H0: diffuse bias in all MoE models | **STRONGLY SUPPORTED** | All 4 MoE models with detectable bias show H ≈ 0.88–0.92. DBRX H=0.919 ≈ Mixtral H=0.917 at N_A/N=0.25 — architecture-agnostic. |

### Final sparsity ladder (Exp1, all models)

| Model | N_A/N | H | Gini | n_players | mean_bias_gap |
|---|---|---|---|---|---|
| OLMoE-1B-7B | 0.016 | 0.900 | 0.599 | 1024 | 0.203 |
| GPT-OSS-120B | 0.031 | 0.880 | 0.724 | 4608 | 0.165 |
| Gemma 4 26B† | 0.063 | 0.822 | 0.821 | 3840 | ≈0 |
| Phi-3.5-MoE | 0.125 | 0.889 | 0.617 | 512 | 0.244 |
| Mixtral-8x7B | 0.250 | 0.917 | 0.516 | 256 | 0.187 |
| DBRX-instruct | 0.250 | 0.919 | 0.554 | 640 | 0.187 |

†Gemma 4: no detectable bias signal — excluded from H1 analysis.

### Exp3 synergy fractions (pairwise interactions at first/last MoE layer)

Universal finding: 70–74% of early-layer attribution mass is pairwise expert interactions (OLMoE 0.702, Phi 0.744, Mixtral 0.716). Late-layer drops to 0.22–0.28 for OLMoE/Phi; Mixtral stays at 0.503. routing_contrast underestimates diffuseness universally.

### Key new findings not anticipated in original design

1. **Architecture-agnostic diffuseness:** DBRX and Mixtral produce H=0.919 and H=0.917 at the same N_A/N=0.25, from different organizations with different architectures — the null result is architecture-independent.
2. **Gemma 4 benchmark saturation:** model is too well-aligned for StereoSet/BBQ/WinoGender to detect bias.
3. **Interaction dominance:** expert-committee effects (synergy) dominate individual marginals at early layers across all three tested architectures.

---

---

## PAGE 6 — Reviewer-Prioritized Extensions (Post-Study, 2026-07-07)

*Added based on analysis of the draft's current weak points: method confound in the MoE–dense comparison, causal validation concentrated in one model/architecture, and the routing-contrast proxy needing independent validation.*

### 6.1 Exp6 — Ladder-wide causal ablation (highest priority)

**Motivation:** The current causal validation (Exp4 ablation cross-check) is run only on OLMoE with 30 prompt pairs. Reviewers will reject the claim that "single-expert debiasing is misguided" if it rests on one architecture. Running the same ablation curve on Mixtral-8x7B (K=2/8, 256 players) and Phi-3.5-MoE (K=2/16, 512 players) confirms the result generalizes. Including OLMo-7B dense extends the cross-architecture ablation to the matched pair used for C4.

**Method:** Extend `run_experiment4_ablation.py` to Mixtral (n=60 pairs), Phi-3.5-MoE (n=60), OLMoE (n=100, upgraded from 30), and OLMo-7B (n=60). Now includes two control conditions:
- **Random control:** ablate players in random order — flat curve validates that the phi-ranked steep drop is attributional, not arbitrary.
- **High-routing control:** ablate by routing frequency (most-activated experts first, regardless of bias attribution) — steep drop here would suggest the effect is just a "compute proxy"; a flat curve confirms phi-ranking captures bias specifically.

**Configs/scripts:** `submit_slurm_experiment4.py --config <existing exp1 config> --max-pairs N`. Results land in `<result_dir>/experiment4/`.

**Expected:** phi-ranked ablation curve drops faster than both controls (especially at k=10%, where current OLMoE data shows −0.92 disparity drop). Consistency across Mixtral and Phi-3.5-MoE would make "diffuse but causally real" the hardest finding to refute.

### 6.2 Exp7 — Proxy-vs-causal agreement test

**Motivation:** routing_contrast (Exp1 method) is a 2-forward-pass proxy for the true Shapley values, which require 2^K forward passes. If the proxy and the causal method rank experts differently, the H and Gini numbers from Exp1 are unreliable. If they agree, the proxy is vindicated and the full-ladder results hold.

**Method:** On Mixtral (K=2/8, 2^8=256 coalitions — tractable), run a 20-pair subsample on first and last MoE layers:
1. Compute per-pair routing_contrast phi over the active expert set.
2. Compute per-pair exact Shapley phi (active coalitions only) using `compute_exact_shapley_for_pair`.
3. Spearman rank correlation between |phi_rc| and |phi_exact| restricted to active experts.
4. Report mean ρ ± std per layer.

**Script:** `run_experiment7_proxy_agreement.py --config configs/study.mixtral-8x7b.concentration.yaml`

**Interpretation:** ρ ≥ 0.7 → proxy trustworthy, Exp1 results stand. ρ < 0.5 → method section needs a caveat narrowing concentration claims to the proxy metric.

### 6.3 Exp8 — Same-mechanism comparison (method-confound resolution)

**Motivation:** The Exp2 result (H_dense ≈ 0.63–0.76 vs H_MoE ≈ 0.88–0.92) supports the H2 rejection, but the comparison confounds attribution method: dense models use LOO (leave-one-out layer ablation) while MoE models use routing_contrast. Running LOO on MoE models produces an apples-to-apples H comparison at the *layer* level, letting us separate "is H lower in dense because of architecture?" from "is H lower in dense because LOO is a coarser decomposition?"

**Method:** Re-run OLMoE and Phi-3.5-MoE with `shapley_method: dense_loo` (forces layer-level LOO regardless of model family, now supported in `run_bias_study.py`). Compare H_lloo(MoE) vs H_lloo(dense). If they converge, the original gap was a method artifact. If H_lloo(MoE) still > H_lloo(dense), the localizability result survives even on matched methodology.

**Configs:** `study.olmoe.lloo.yaml` (n=100, 1×A100, 3hr) and `study.phi3.5-moe.lloo.yaml` (n=50, 2×A100, 3hr).

### 6.4 Future extensions (not submitted in current batch)

The following are noted for completeness but not yet implemented:

- **Committee ablation (Exp9):** Ablate top expert *pairs/triplets* from Exp3 interaction results, rather than single experts. Tests whether the synergy (70–74% of early-layer mass) is a property of specific standing committees. Requires `compute_committee_ablation_curve` (new code).
- **Capability preservation (Exp10):** Measure MMLU accuracy or perplexity on a held-out set before/after ablation, to confirm "debiasing" does not just degrade the model.
- **Quantization sensitivity:** GPT-OSS-120B uses MXFP4 quantized weights; H=0.880 may reflect weight distribution distortion rather than true routing concentration. A BF16 run or a second quantized comparison point would control for this.
- **Prompt-set robustness:** Bootstrap resampling over the 400 prompt pairs + subgroup-stratified H across demographic slices (gender/race/religion subsets of StereoSet).

---

## References
- The Expert Strikes Back: Interpreting MoE LMs (arXiv 2604.02178, 2026) — https://arxiv.org/html/2604.02178v1
- Sparsity and Superposition in Mixture of Experts (OpenReview 2025) — https://openreview.net/pdf?id=bZqopmfZDE
- Shapley-MoE: Discovering Important Experts for MoE (OpenReview 7kQjbCQwtT) — https://openreview.net/pdf?id=7kQjbCQwtT
- Knowledge Localization in MoE LLMs via Cross-Lingual Inconsistency (arXiv 2603.17102, 2026) — https://arxiv.org/abs/2603.17102
- Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective (arXiv 2506.05166, 2025) — https://arxiv.org/html/2506.05166v2
- DeM-MoE: Modeling Annotator Disagreement with a Demographic-Aware MoE (arXiv 2508.02853 / ACL 2026) — https://arxiv.org/html/2508.02853v1
- Can MoE Surpass Dense LLMs Under Strictly Equal Resource (arXiv 2506.12119, ICLR 2026) — https://arxiv.org/abs/2506.12119
- Sparse Models, Sparse Safety: Unsafe Routes in MoE LLMs (OpenReview JRtldw5Mpw, ICLR 2026) — https://openreview.net/forum?id=JRtldw5Mpw
- The Illusion of Specialization: Standing Committees in MoE (arXiv 2601.03425, 2026) — https://arxiv.org/html/2601.03425
- Robustly Improving LLM Fairness in Realistic Settings (arXiv 2506.10922, 2025) — https://arxiv.org/pdf/2506.10922
- SHAP-based Explanations are Sensitive to Feature Representation / FoolSHAP (arXiv 2505.08345, 2025) — https://arxiv.org/html/2505.08345v1
- Bias in LLMs: Origin, Evaluation, and Mitigation (arXiv 2411.10915, 2026 survey) — https://arxiv.org/html/2411.10915v1
- SHAP fairness tooling (group_difference_plot) — https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Explaining%20quantitative%20measures%20of%20fairness.html
- RealExp: Decoupling correlation bias in Shapley values (SciDirect 2025) — https://www.sciencedirect.com/science/article/abs/pii/S0306457325000949
- StereoSet (Nadeem et al., 2020) — https://arxiv.org/abs/2004.09456
- BBQ (Parrish et al., 2022) — https://arxiv.org/abs/2110.08193
- WinoGender (Rudinger et al., 2018) — https://arxiv.org/abs/1804.09301
- C-Eval (Huang et al., 2023) — https://arxiv.org/abs/2305.08322
