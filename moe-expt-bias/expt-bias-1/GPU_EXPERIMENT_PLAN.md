# GPU Experiment Plan: Strengthening Non-Localizability (Bias in MoE)

> **Goal**: Prove that social bias in MoE LLMs is **non-localizable even with surgery (expert ablation) or router skewing**, using complementary Shapley-style attributions. This is the *only* plan where every GPU hour directly hardens that thesis for NeurIPS/ICML.
>
> **You said you can run more GPU — this is what to run, in exact order, with exact configs and cost.**

---

## Current State (what already supports the thesis)

| Evidence | Finding | Strength | Remaining Risk |
|---|---|---|---|
| **Exp1 routing-contrast ladder** (6 MoE, 4 dense, 5000 pairs) | `H≈0.88–0.92` diffuse, `t5` 2–11% everywhere; no monotone sparsity trend (`ρ=0.2 p=0.63` pre-correction) | Strong observation | **Construct bug**: whole-string logprob, BBQ `target_loc` ignored, no common prompt battery — reviewers will call it mis-specified |
| **Exp3 synergy** (20 pairs, first/last layer) | 70–74% interaction mass early layers | Moderate | `n=20`, only 2 layers, ad-hoc synergy fraction |
| **Exp6 ablation** (OLMoE/Phi/Mixtral, 30–60 pairs, 3 conditions) | Proxy selectivity 0.24 / 1.35 / −0.18; frequency matches/beats proxy in 2/3; DBRX reversal | Moderate | Single random baseline, zero-ablation OOD, no renormalization, capability measured only on bias prompts |
| **Exp7 proxy vs exact** (Mixtral/OLMoE/Phi) | `ρ≈0` (routing-contrast unrelated to causal) | Strong | Only 3 models, 1 layer each |
| **Exp8 same-mechanism LOO** (OLMoE `H=0.736`, Phi `H=0.899`) | Ambiguous split (1 in dense band, 1 in MoE band) | Weak (n=2) | Needs 4 more ladder rungs to be a trend |
| **Exp5 demographic JS** (OLMoE, 85 cohorts) | `D_JS=0.22` vs null `0.31` → no specialist | Weak | Null permutes expert IDs not labels, pool includes cohort, no stability filtering |

**Bottom line**: *Observation* of diffuseness is solid; *causal "surgery fails"* and *"router skewing fails"* are **not yet reviewer-proof**.

---

## Tier 0 — MUST FIX BEFORE ANY NEW SCIENCE (otherwise new runs are wasted)

> These are **validity blockers**. Running Tier 1/2 without Tier 0 lets reviewers reject on pipeline, not thesis.

### T0.1 — Freeze a common prompt battery + fix scoring (0 GPU for manifest, reruns are GPU)

**Why**: Right now `pairs[:5000]` after deterministic concat gives different benchmark mixtures per model (2106/2894 vs 2000 StereoSet-only). BBQ `biased_ans` ignores `target_loc`+`question_polarity`; WinoGender fixes male=stereo; `_sequence_logprob` scores whole string not answer tokens. Any H difference could be benchmark mix, not architecture.

**What to do**:
1. Create `item_manifest.json` (5000 IDs, balanced: 2500 StereoSet + 2500 BBQ; if adding WinoGender, 2000/2000/1000) with stable hash, sampling rule documented.
2. Fix `benchmarks.py`: use `target_loc` for BBQ, derive WinoGender direction from BLS *or* mark unsigned and report separately, score **answer tokens only** conditional on shared prefix.
3. Add unit test: all 6 BBQ answer permutations + WinoGender BLS case.

**GPU to close it**: Rerun Exp1 routing-contrast on **common manifest, answer-conditional scoring** for at least 2 models (OLMoE + Mixtral) to show `H` still diffuse `≈0.88–0.92`. Cost: 1×L40S 3h + 4×H100 1h. **If you only rerun one thing, rerun this**.

**Configs**: `study.olmoe.concentration.v1.yaml` + `study.mixtral-8x7b.concentration.v1.yaml` with `benchmarks: [stereoset, bbq]` + manifest flag + `max_prompts: 5000`.

### T0.2 — Recompute CIs with correct stratification + cluster bootstrap (0 GPU after payload restore)

Code already fixed (`s04_bootstrap_cis.py` now `benchmark:bias_type` fallback), but on-disk `s04_bootstrap_cis.json` still from old 1-stratum code and payloads are gitignored. Restore Kaggle `sghose0/moe-bias-routing-shapley-perpair-phi` then `python stats_analysis/scripts/s04_bootstrap_cis.py` etc. Also upgrade to cluster bootstrap at `context`/`example_id` level where available.

---

## Tier 1 — DIRECTLY HARDEN "SURGERY FAILS" AND "ROUTER SKEWING FAILS" (run these next)

### T1.1 — Robust surgery: does ablation still fail with proper controls? (480 GPU-min, reviewer blocker)

**Gap closed**: L (zero ablation OOD), Z (single random).

**Current**: Single random, zero-ablation only, capability = perplexity on bias prompts.

**Robust version** (E16):
- **20 random sets** per `k` → 95% band (not 1 line)
- **3 intervention operators**: (a) zero-ablation, (b) **router masking + renormalization** (set expert weight 0 then renormalize remaining `k` → models reroute, not OOD), (c) **mean-output replacement** (replace expert output with its mean over 100 neutral prompts, not zero)
- **Held-out capability**: WikiText perplexity + 200-question MMLU subset (not bias prompts)
- **Deletion AUC** with paired CI, report absolute `Δgap` not just ratio.

**What to run** (4 jobs, all under 960 GPU-min):

```bash
# OLMoE (cheapest, 16 experts top-1) — 1×L40S 2h
python scripts/run_experiment6_ablation.py --config configs/study.olmoe.concentration.v1.yaml --max-pairs 60 --operators zero,renorm,mean --n-random 20 --capability wikitext,mmlu

# Phi-3.5-MoE — 2×H100 2h
python scripts/run_experiment6_ablation.py --config configs/study.phi3.5-moe.concentration.v1.yaml --max-pairs 50 --operators zero,renorm,mean --n-random 20

# Mixtral-8x7B — 4×H100 1h (already `study.mixtral-8x7b.concentration.v1.yaml` sized)
python scripts/run_experiment6_ablation.py --config configs/study.mixtral-8x7b.concentration.v1.yaml --max-pairs 50 --operators zero,renorm,mean --n-random 20

# DBRX — 4×H100 1.5h
python scripts/run_experiment6_ablation.py --config configs/study.dbrx.concentration.v1.yaml --max-pairs 50 --operators zero,renorm,mean --n-random 20
```

**Success criterion for thesis**: Even with renormalization + held-out capability, top-10% surgery still gives selectivity ≤1.5 and random band overlaps proxy — surgery not surgical.

### T1.2 — Router skewing: does forcing the router to avoid "biased experts" work? (360 GPU-min, novel contribution)

**Gap closed**: Core thesis mentions router skewing but **no experiment tests it** — biggest missing piece.

**Idea**: Instead of ablating experts, **skew the router** to down-weight experts with high `|phi|` (or high frequency) and up-weight low-`|phi|` experts, *without* removing capacity. If bias were localizable via routing, skewing should reduce `gap` with minimal capability loss. If bias is diffuse/entangled, skewing either does nothing or hurts capability similarly to ablation.

**How** (2-layer implementation, cheap):
- Learn per-layer affine skew: `logits_skewed = logits + α * (−|phi|)` where `α` ∈ {0, 0.5, 1.0, 2.0} (negative skew away from high-phi experts). Renormalize via softmax. No training — pure inference hook like ablation, but preserves total mass (so not OOD zero).
- Evaluate same `Δgap` / WikiText PPL / MMLU as T1.1.

**What to run** (reuse T1.1 model set, one extra script):

```bash
# New script: run_experiment_router_skew.py (create from run_experiment6_ablation.py, swap ablation for router skew hook)
python scripts/run_experiment_router_skew.py --config configs/study.olmoe.concentration.v1.yaml --max-pairs 100 --alphas 0.5,1.0,2.0
python scripts/run_experiment_router_skew.py --config configs/study.mixtral-8x7b.concentration.v1.yaml --max-pairs 100 --alphas 0.5,1.0,2.0
python scripts/run_experiment_router_skew.py --config configs/study.phi3.5-moe.concentration.v1.yaml --max-pairs 100 --alphas 0.5,1.0,2.0
python scripts/run_experiment_router_skew.py --config configs/study.dbrx.concentration.v1.yaml --max-pairs 50 --alphas 0.5,1.0,2.0
```
*Cost*: ~1×L40S 1.5h + 2×H100 2h + 4×H100 1h each — total ~360 GPU-min. **This is the experiment reviewers will ask for by name: "if bias is diffuse, show router intervention also fails."**

### T1.3 — Same-mechanism LOO full ladder: close the granularity confound (570 GPU-min, configs ready)

**Gap closed**: H (player cardinality), Exp8 ambiguous n=2.

**What to run** (4 configs already authored and smoke-tested, including Gemma compound-branch fix):

```bash
python scripts/submit_slurm_study.py --config configs/study.mixtral-8x7b.lloo.yaml   # 4×H100 1h, 50 pairs
python scripts/submit_slurm_study.py --config configs/study.dbrx.lloo.yaml           # 4×H100 1.5h, 50 pairs
python scripts/submit_slurm_study.py --config configs/study.gpt-oss-120b.lloo.yaml    # 2×H100 4h, 30 pairs, force_eager_moe=true
python scripts/submit_slurm_study.py --config configs/study.gemma4-26b.lloo.yaml      # 1×H100 3h, 30 pairs
```

Already `sbatch`-ready, all under 960 GPU-min. OLMoE/Phi already done (`H=0.736` vs `0.899`). If Mixtral/DBRX also land in dense band, thesis needs nuance ("granularity explains part"); if they land high (~0.88), same-mechanism diffuseness is a trend, not artifact — either way, required for honest reporting.

### T1.4 — Multidimensional Shapley Modes: show averaging didn't hide localization (180 GPU-min)

**Gap closed**: AA (averaging hides structure), M (ad-hoc synergy).

**What to run**: On OLMoE + Mixtral, 50 pairs, 2 layers (0 + last), Monte-Carlo lesion Shapley (MSA, Dixit) → per-token/per-bias-type contribution maps. Use `compute_exact_shapley_for_pair` sampled coalitions (not full 2^K if K>10, sample 200 orderings). Compute **STII / Shapley-Taylor** with efficiency check on synthetic additive vs interacting game (validate implementation first).

```bash
python scripts/run_experiment3_interactions.py --config configs/study.olmoe.concentration.v1.yaml --max-pairs 50 --layers 0,15 --method msa --n-orderings 200
python scripts/run_experiment3_interactions.py --config configs/study.mixtral-8x7b.concentration.v1.yaml --max-pairs 50 --layers 0,31 --method msa --n-orderings 200
```
*Cost*: 1×L40S 1.5h + 4×H100 1.5h. Proves synergy 70% is not an artifact of `Δrouting_weight` aggregation.

---

## Tier 2 — NICE-TO-HAVE (do after Tier 0/1 if time/Budget remains)

| ID | Experiment | Why | Cost |
|---|---|---|---|
| **T2.1** | Within-model `k`-sweep (OLMoE top-1/2/4/8 inference; Mixtral 2/4) | Tests if `H` moves with `k` *within* a family, controls for training data/regularization (Thesis AB). Distinguish inference `k` vs trained `k`. | OLMoE 1×L40S 3h, Mixtral 4×H100 4h |
| **T2.2** | Positive-control conditioned router (AGER/MoSAIC style) | If you train a LoRA-per-attribute + oracle router (gender/race) and bias *still* diffuse, thesis is airtight. If it localizes, you show *when* MoE can localize. Future-work if limited. | 8 GPU-h fine-tune |
| **T2.3** | Regularization/router-entropy audit (0 GPU) | Tabulate load-balance coefficients, router entropy per model, correlate with `H` — tests thesis AB without GPU. | 0 |
| **T2.4** | Generation-based bias vs logprob gap (200 BBQ ambiguous, greedy decode) | Closes likelihood-vs-generation gap (V); Gallegos taxonomy demands it. | 1×A100 2h/model |
| **T2.5** | Template robustness (3 prompt formats × 200 pairs) | BBQ sensitive to formatting (Parrish). | 1×L40S 1h/model |

---

## What NOT to run next (saves GPU)

- **Another 5000-pair ladder rung** (e.g., Qwen3) **before Tier 0** — point estimate `n=6→7` does not increase power meaningfully; reviewers will still call pipeline mis-specified. Do Tier 0/1 first.
- **Full AGER training** — too heavy for this revision; mention as future work with pilot signal only.
- **C-Eval multilingual** — deprioritized (all-English ladder); scope to English and cite.

---

## Minimal Resubmission Set (if you have 1 weekend, ~24 GPU-hours)

1. **T0.1 rerun** (OLMoE + Mixtral common manifest, answer-conditional) → proves diffuseness survives correct scoring.
2. **T1.1 robust surgery** (20 randoms + renorm + held-out PPL on 4 models) → proves surgery fails even without OOD.
3. **T1.2 router skewing** (same 4 models) → proves skewing also fails (novel, thesis-specific).
4. **T1.3 LOO full ladder** (4 configs ready) → closes granularity confound.
5. **T1.4 MSA Modes** (2 models, 50 pairs) → shows aggregation didn't hide localization.

Total: ~1800 GPU-min (~30 GPU-hours, ~7–8 jobs under 960 cap) → turns "diffuse observation" into "causal, routing-invariant non-localizability."

---

## Exact Submit Lines (copy-paste from login node, after Tier 0 manifest is frozen)

```bash
cd moe-expt-bias/expt-bias-1

# Tier 0 reruns (common manifest, answer-conditional — requires benchmarks.py fix first)
python scripts/submit_slurm_study.py --config configs/study.olmoe.concentration.v1.yaml --max-prompts 5000
python scripts/submit_slurm_study.py --config configs/study.mixtral-8x7b.concentration.v1.yaml --max-prompts 5000

# Tier 1.1 robust surgery
python scripts/submit_slurm_study.py --config configs/study.olmoe.concentration.v1.yaml --max-prompts 60  # then run_experiment6_ablation with operators
python scripts/run_experiment6_ablation.py --config configs/study.olmoe.concentration.v1.yaml --max-pairs 60 --operators zero,renorm,mean --n-random 20 --capability wikitext,mmlu

# Tier 1.3 LOO ladder (ready now)
python scripts/submit_slurm_study.py --config configs/study.mixtral-8x7b.lloo.yaml --save-per-pair-phi
python scripts/submit_slurm_study.py --config configs/study.dbrx.lloo.yaml --save-per-pair-phi
python scripts/submit_slurm_study.py --config configs/study.gpt-oss-120b.lloo.yaml --save-per-pair-phi
python scripts/submit_slurm_study.py --config configs/study.gemma4-26b.lloo.yaml --save-per-pair-phi

# Tier 1.2 router skew (requires new script — I can draft it next)
python scripts/run_experiment_router_skew.py --config configs/study.olmoe.concentration.v1.yaml --alphas 0.5,1.0,2.0
```

*All Tier 1.3 configs already carry correct `time`/`gpu_type`/`mem` for 960 GPU-min cap; Tier 1.1/1.2 need new `run_experiment*_skew.py` which reuses the same hook scaffolding as `shapley.py:mean_bias_gap_with_players_ablated` but skews `topk_weight` instead of zeroing.*

---

## How to Report Tier 1 Results (so reviewers cannot dismiss)

- **Primary endpoint**: `H` (routing-contrast on common manifest, answer-conditional) — **one number, preregistered, before seeing Tier 1**.
- **Surgery endpoint**: **deletion AUC** (bias) vs **WikiText PPL AUC** — not just `k=10%` snapshot. Report paired difference `AUC_proxy − AUC_random` with 95% CI from 20 randoms. If CI includes 0, surgery has no advantage.
- **Router skew endpoint**: `Δgap` vs `α` and vs `ΔPPL(WikiText)` — same AUC framing. If skew needs `α>1` to move `gap` 10% but PPL already +20%, skewing fails same way ablation fails.
- **Cross with synergy**: correlate per-layer synergy fraction (T1.4) with per-layer `ρ` (Exp7) — high synergy predicts low `ρ` (mechanism).
