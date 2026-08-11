# stats_analysis — "Review-response" robustness series

> **SUPERSEDED (2026-08-11): historical working log from 2026-08-08.**
> Everything marked "decision needed" / "MAJOR DISCREPANCY" / "broken" below
> has since been RESOLVED; this file is kept for provenance, not as current
> state. Resolution map:
> - **GPT-OSS-120B v1 is VALID** (2000-pair capture H=0.8764 + 5000-pair
>   replication H=0.8789), not "broken all-zero phi" — the broken run was
>   discarded and re-captured (`gpt-oss-120b-v1-smoke` is a scratch dir).
> - **s02 "MAJOR DISCREPANCY 0.42 vs 0.22"** resolved: the paper now reports
>   the recomputed values (mean pairwise JSD 0.22, CI [0.206,0.231],
>   expert-identity permutation null mean 0.31 / p95 0.37; abstract + §5.6.5).
> - **H1 verdict** reframed as directionally consistent + exact-permutation
>   p-values + Monte Carlo power analysis (§6.4, repro `s08_power_analysis.py`).
> - **CIs/SEs/tests** landed for all models (`s04_bootstrap_cis.py`).
> Current sources of truth: `REPRODUCIBILITY.md` (pipeline s01–s08 +
> `audit_appendix.py`) and the paper `moe-expt-bias-2/moe_bias_report_acm_v2.tex`.
> Per-script run records below are unchanged history.

Dedicated workstream answering the advisor feedback on `moe_bias_report_draft.tex`.
Everything for this series lives here; nothing in `results/`, `figures/`, or the
paper's numbers should be touched from this series until a task below marks it DONE.

Feedback points addressed:
1. H1 verdict reproducibility + GPT-OSS exclusion handling
2. Metric consistency: "routing structure vs causal bias" framing
3. No CIs / SEs / significance tests anywhere

## Layout

| path | contents |
|---|---|
| `scripts/s01_exp1_stability.py` | Exp1 point-estimate stability: v0(400 pairs) vs v1(5000) vs shard halves; flags broken zero-phi runs |
| `scripts/s02_exp5_js.py` | Full Exp5 JS divergence analysis over the 85 stored per-cohort phi vectors: pairwise JS, block bootstrap CI, expert-identity permutation null, cohort-vs-pool, figure |
| `scripts/s03_h1_verdict.py` | Monotonicity audit of H1 under model-set exclusions (full 6 / minus GPT-OSS / paper-valid / unique-sparsity) with drift-aware tolerance; point AND drift-aware verdicts |
| `scripts/gen_readings_artifact.py` | Generates `artifacts/bias_attribution_readings.json`: 33 verbatim tex quotes where H/Gini/JS are equated with bias, with per-row line anchors and framing-A/B replacement guidance |
| `outputs/*.json` | Machine-readable results of s01–s03 (inputs to paper numbers) |
| `figures/` | Statistical figures (e.g. `s02_js_distribution.png`) |

## Status of analyses (2026-08-08)

### s01 — Exp1 stability (DONE)
- olmoe: v0 H=0.9002/G=0.5987 → v1 0.8786/0.6457 (dH 0.0216)
- phi3.5-moe: 0.5885/0.6170 → 0.8779/0.6435 (dH 0.0106)
- mixtral: 0.9170/0.5164 → 0.9167/0.5189 (stable; shards 0.9148/0.9173)
- dbrx: 0.9190/0.5537 → 0.8970/0.5995 (shards 0.9040/0.9045)
- **gpt-oss-120b: v1 is a BROKEN run (all-zero phi, H=0.0)** — only v0 (400 pairs) is valid: 0.8795/0.7236
- gemma4-26b: v0 (400 pairs) only: 0.8221/0.8215
- Resolvable floor (max v0↔v1 drift across valid runs): **dH 0.0221, dG 0.0470**

### s03 — H1 verdict (point + drift-aware)
- full 6-model ladder: NOT SUPPORTED (REJECTED holds on the full ladder)
- minus GPT-OSS only: NOT SUPPORTED (gemma still breaks)
- paper-valid-4 (olmoe/phi3.5/mixtral/dbrx): SUPPORTED
- unique-sparsity-3 (olmoe/phi3.5/mixtral): SUPPORTED
- **Gist: reviewer says "drop GPT-OSS → H1 supported"; we find H1 is only supported if BOTH GPT-OSS and Gemma are dropped. The verdict must be re-scoped to "no significant gradient" or restricted to the valid set — decision needed.**

### s02 — Exp5 JS numbers: MAJOR DISCREPANCY
- Recomputed from stored v1 cohort vectors: pairwise mean JS = **0.2213** (SD 0.0508, CI of mean [0.206, 0.231], range 0.063–0.413) on 3570 pairs over 85 cohorts
- Paper claims **0.42 (range 0.35–0.48)** — not reproducible from the stored data (likely a different/older run or computation). NEEDS a decision: re-run Exp- div with the instrumented pipeline, or the paper's number needs regeneration.
- vs reported label-permutation null (0.05±0.01): observed 0.22 is 4.4× — significant in direction.
- BUT vs expert-identity permutation null computed here (mean 0.31, p95 0.37): only 0.5% of observed pairs exceed null p95 — i.e., cohort divergence is *below* what random expert identity on the same pool gives. Supports the "shared routing structure" reading, not "specialized sub-networks."

## Cluster rerun sweep (submitted 2026-08-08, PACE `ice-gpu`)

All reruns pass `--save-per-pair-phi` via `slurm/run_experiment_script.sbatch` (EXTRA_ARGS). Output dirs: `<existing>/-perpair` (primary) and `-perpair-b` (backup). Job IDs 5573370–5573387.

| model | config (n_pairs) | GPU | jobs (prim / bkp) | out dir | notes |
|---|---|---|---|---|---|
| olmoe-1b-7b | v1 (5000) | 1×l40s | 5573370 / 5573371 | `-v1-perpair[-b]` | both already RUNNING |
| phi3.5-moe | v1 (5000) | 2×h100 | 5573372 / 5573373 | `-v1-perpair[-b]` | running |
| mixtral-8x7b | v1 (5000, 2 shards) | 4×h100 | 5573374,75 / 5573376,77 | `-v1-perpair[-b]` | shards 0/1 |
| dbrx | v1 (5000, 2 shards) | 4×h100 | 5573378,79 / 5573380,81 | `-v1-perpair[-b]` | shards 0/1 |
| gpt-oss-120b | v0 (400) v1 broken | 2×h100 | 5573382 / 5573383 | `-perpair[-b]` | USE_HUB_KERNELS=NO |
| gemma4-26b | v0 (400) only config | 1×h100 | 5573384 / 5573385 | `-perpair[-b]` | |
| exp5 demographic | v1 (5000) | 1×l40s | 5573386 / 5573387 | `-v1-perpair[-b]` | |

Check: `squeue -u sghose7`; logs in `slurm-logs/` of each out dir. When done, merge shard per-pair arrays with same tooling as existing `result_shard*/phi_shard*` merge, then re-run s01–s03 with per-pair CIs (add `bootstrap CI` section here).

## Paper-update checklist (what to do when stats finalize)

1. Re-run Exp1 (4-5 models) with `--save-per-pair-phi`; then compute bootstrap CIs for H/Gini + pairwise model tests using the tooling built by the instrumentation agent (see git diff in `src/moe_bias_shapley/{shapley,reporting}.py` and `scripts/run_bias_study.py`).
2. Re-run Exp5 with `--save-per-pair-phi` to (a) re-verify the 0.42/0.05 numbers, (b) obtain the exact label-permutation null, (c) compute the cohort JS bootstrap properly.
3. Decide framing A (routing-is-the-bias-substrate) vs B (structure-only) — `artifacts/bias_attribution_readings.json` gives every sentence needing changes.
4. Replace point estimates with CI-equipped tables; add error bars to figs (figs 2–6 in `moe-expt-bias/figures*`).
5. Add take-away boxes (drafted inside `bias_attribution_readings.json`'s readme section).
6. Re-derive every "0.88–0.92 diffuse", "0.42 ≫ 0.05", "REJECTED/SUPPORTED" verdict from the new numbers.