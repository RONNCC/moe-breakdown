# Why Router Skewing — when Shapley already says "bias is distributed"

> User question: "the result is shapley shows bias is distributed in experts — why do you need router skewing?"

## The 30-second answer

**Shapley (as you run it now) is an observation. Surgery and skewing are interventions.**
A reviewer will not accept "we observed diffuseness → therefore bias cannot be removed" — they will say you never *tried* to remove it properly. Router skewing is the cleanest, cheapest *try*.

If skewing **also fails**, you have proven: bias is not just diffuse on paper, it is **inseparable in practice even when you keep the model intact and just reroute**. If skewing **succeeded**, your thesis would be falsified — which is exactly why you must run it.

---

## What each method actually measures (they are not redundant)

| Method | What it does | Analogy | What it can prove | What it cannot prove |
|---|---|---|---|---|
| **Routing-contrast** `Δrouting_weight × gap` (Exp1) | For each expert, look at how much more it fires on stereotyped vs anti-stereotyped prompts, multiply by the model's logprob gap. Purely **observational correlation**. Never modifies the model. | Noticing that cars from every road end up polluting the city. | "Bias correlates with many experts diffusely (H≈0.88-0.92)" | That removing those experts would reduce bias, or that bias lives in experts at all vs routing. |
| **Exact lesion Shapley** `v(S)=gap with coalition S intact` (Exp7) | Actually **lesion (zero) coalitions** of experts and average marginal contributions (2^K). Causal, but only on tiny coalitions (K=16 for OLMoE) and with zero-ablation. | Closing roads one-by-one to see which closures actually reduce pollution. | "Observational ranking (routing-contrast) does NOT predict causal ranking (ρ≈0)" — your heuristic is not Shapley. | What happens at scale (256 experts), or whether the *way* you closed the road created the traffic jam. |
| **Zero-ablation surgery** (Exp6) | Delete top-phi experts (set output to 0) and measure Δgap vs Δperplexity. | **Demolishing the busiest roads.** | "Deleting diffuse experts hurts capability 22-233% and barely moves bias (selectivity 0.24–1.35, Mixtral -0.18)" | Whether the damage comes from *losing capacity* (OOD state never seen in training) vs bias being entangled. Reviewer's #1 objection: "Of course demolishing roads breaks the city — you didn't prove pollution is inseparable, you proved demolition is destructive." |
| **Router skewing** (proposed T1.2) | **Do NOT delete experts.** Scale router logits: `logits += α·(−|phi|)` for high-bias experts, then **renormalize** (softmax). Model keeps all experts, just reroutes away from biased ones. | **Redirecting traffic with signs, not demolishing roads.** | "Even when capacity is preserved and routing is explicitly steered away from bias, Δgap is small and/or PPL/MMLU collapses → bias is routing-invariant, not just OOD artifact." | Nothing alone — needs Tier0 validity fix first. |

**Key point**: Shapley tells you *where bias correlates*, skewing tells you *whether you can steer away from it without breaking the model*. They test different counterfactuals. A diffuse Shapley map is *necessary* but not *sufficient* for non-localizability.

## Why zero-ablation alone is not reviewer-proof (your current Exp6)

1. **OOD**: Mixtral/DBRX never saw "expert output = 0" during training. Zero is far from the expert's natural output distribution. Perplexity *must* rise, even if bias were perfectly localized. You are measuring out-of-distribution shock, not entanglement.
2. **Single random baseline**: 1 random set vs 20 needed for 95% band — reviewer will say variance explains DBRX reversal.
3. **Capability on bias prompts**: Measuring PPL on StereoSet prompts confounds bias and fluency. Needs held-out WikiText + MMLU.
4. **No rerouting**: After you zero experts, the router's softmax is *not* renormalized in current code — remaining experts keep same weights, total mass <1. That's not how MoE works; the gating literature (Zhou Expert Choice, Fedus) renormalizes.

Router skewing fixes all four: no deletion, renormalized, same held-out metrics, 20 random skews as control.

## The two hypotheses skewing distinguishes

- **H_A (your thesis)**: Bias is **substrate-entangled** — every expert's weights encode stereotypical associations (like grammar), so no routing can avoid it. → *Prediction*: skewing α=0.5/1/2 barely moves Δgap or PPL/MMLU collapses fast even with small α.
- **H_B (reviewer's alternative)**: Bias is **routing-localized** — a few experts are "biased experts", and the router just happens to send stereotyped prompts there. → *Prediction*: gentle skewing (α=0.5) moves Δgap substantially with small PPL cost; you *can* debias by routing.

Exp1 alone cannot distinguish H_A vs H_B — both produce diffuse *observational* H if the router spreads load (load-balancing loss forces diffusion). Exp7 hints at H_A (ρ≈0) but on tiny K. Skewing at scale on full 256 experts is the direct test.

**This is also why MSA matters**: Dixit finds *language* **does** localize in same Mixtral family (Vietnamese vs French experts). So MoE *can* localize when the signal is cleanly routed. If bias *still* doesn't localize under skewing, it is bias-specific entanglement, not "MoE never specializes" — a much stronger, reviewer-proof framing.

## What "success" and "failure" look like for your paper

Both outcomes are publishable — but you must pre-register:

- **If skewing fails (supports thesis)**: Report `Δgap vs α` and `ΔWikiText PPL vs α` curves. Show proxy-skew beats random-skew but still selectivity ≤1.5 and needs α=2 to move gap 10%, at 30%+ PPL cost. Conclusion: "Bias is routing-invariant; mitigation requires weight-level intervention (training-time), not inference-time routing."
- **If skewing succeeds (falsifies strong thesis)**: Still publishable as "bias is diffuse observationally but reroutable causally" — a *positive* debiasing result. You would then pivot thesis to "observation ≠ causality, but lightweight router steering *is* effective" — also NeurIPS-worthy.

**Not running it leaves you in the worst position**: reviewer says "you never tested the obvious lightweight fix — maybe rerouting works and you just demolished instead."

## Cost: why we push it so hard

- **360 GPU-min** (3 alphas × 50 pairs × 4 models) — same hooks as ablation, no training, no new data.
- Reuses `compute_ablation_curve` logic: replace `zero_expert_output` with `logits[biased] -= α·|phi|` + `softmax` renorm.
- Under 960 cutoff (largest job: Mixtral 4×H100 1h).

In `GPU_EXPERIMENT_PLAN.md` T1.2 we give exact sbatch lines. Minimal resubmit we recommend is T0 rerun + T1.1 renorm surgery + **T1.2 skewing** + T1.3 LOO ladder = ~22h wall-clock, ~30 GPU-hours, and closes the central reviewer objection.

## One-line to put in your rebuttal/related work

> "Observational diffuseness (H≈0.9, ρ≈0) shows routing-contrast does not predict lesion causality; we test whether the causal gap can be closed without capacity loss via router skewing (logit scaling + renormalization). Its failure — if observed — demonstrates routing-invariant entanglement, distinguishing substrate-encoded bias from router-localized specialization as seen for language in MSA (Dixit 2025)."

