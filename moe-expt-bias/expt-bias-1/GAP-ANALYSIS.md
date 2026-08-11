# PhD-Style Gap Analysis: MoE-Bias Study (expt-bias-1)

**Date**: 2026-08-10 (updated post statistical-audit / paper-integration-II session)
**Paper status**: 11-page ACM draft at `moe-expt-bias-2/moe_bias_report_acm_v2.tex` (compiles clean, verified via 2x pdflatex + page-image inspection of pages 3, 5, 6, 8, 10, 11)
**Repo**: `/Users/ronnie.ghose/src/priv/gatech/research-projs/moe-breakdown`

---

## 0. Second-pass statistical audit (this session)

A re-derivation of the exact-permutation floor claims found a **math error**
repeated in 5 places in the v2 draft: the text claimed the smallest
attainable two-sided exact permutation $p$ was $1/12 \approx 0.083$ at
$n=6$ and $0.10$ at $n=5$. Recomputing the true tie-corrected minimum
(smallest $|\rho|$ achievable divides the permutation space; $6! = 720$
perms at $n=6$, $5! = 120$ at $n=5$) gives $p_{\min} = 0.0056$ at $n=6$ and
$0.0333$ at $n=5$ --- both **below** $0.05$. This flips the paper's own
narrative: the $n=5,6$ non-significance is a genuine **power** shortfall
(sampling noise), not a **floor** artifact as previously claimed; only
$n=3,4$ are floor-bound. Fixed in all 5 locations (abstract, Results 5.1,
Robustness 5.6.3, Discussion, Limitations) plus a new
`Section 6.4 Power analysis` with a Monte Carlo simulation quantifying the
ladder size needed for $80\%$ power ($n\approx12$--$13$ at $\rho_H=0.754$;
$n\approx8$--$9$ at $\rho_H=0.872$). Also added: **Section 5.4 Effect
sizes** (Cohen's $d$ for entropy/Gini/top-5-share, $LR$ localizability
ratio per model) showing the dense/MoE geometry split is huge ($|d_H|
\approx 3.9$--$6.1$) while the underlying **bias-gap magnitude** is
statistically indistinguishable between dense and MoE ($d=0.011$
excl.\ Gemma) --- the sharpest available evidence for the
routing-structure-not-magnitude framing the professor's 2nd criticism
demanded. **Section 5.5 Expert-pair synergy (Exp.\ 3)** and an extended
causal-ablation Results/Verdict (DBRX + Gemma-4-26B, both previously
PENDING, now landed) were also integrated; see below.

---

## 1. Professor's Criticism --- Addressed vs Remaining

| Criticism Point | Status | Evidence |
|---|---|---|
| "H1 supported once GPT-OSS dropped; GPT-OSS excluded for quantization reasons" | **RESOLVED** | GPT-OSS-120B v1 VALID (2000 pairs, H=0.8764, G=0.7274) plus a 5000-pair stability replication ($H=0.8789$, within CI). Paper reports exact permutation p-values on ALL 6 rungs: rho_H=+0.754, p=0.106. Framed honestly: directionally consistent, underpowered (n<=6), and now backed by a Monte Carlo power analysis (Section 6.4) instead of the incorrect floor claim. |
| "Metric measures routing structure, not causal bias magnitude; then Exp1 entropy and Exp5 JS aren't measuring bias concentration" | **RESOLVED, now quantified** | Discussion explicitly resolves: attribution geometry is a routing-structure quantity; caveat applies uniformly to Exp1 entropy AND Exp5 JS. Section 5.4 now adds the quantitative version: Cohen's $d=0.011$ for bias-gap magnitude (dense vs.\ MoE, excl.\ Gemma) vs.\ $d=3.9$--$6.1$ for entropy --- geometry and magnitude are empirically orthogonal. |
| "No CIs, SEs, tests --- analyses must be statistically robust" | **RESOLVED** | MoE ladder: 95% block-bootstrap CIs (5000 draws, seed 42, stratified) for all 6 rungs x 4 metrics (Table 2). Dense baselines have CIs (Table 3 + Appendix A). Split-half shard agreement (Mixtral/DBRX, exact numbers in Appendix A) and a Monte Carlo power simulation (Section 6.4) round out the statistical treatment. |

---

## 2. Study Catalog vs On-Disk Reality

### Experiment 1 --- Sparsity Ladder (RQ1/C1)
| Model | v0 (400) | v1 (5000) | Per-pair phi | Status |
|---|---|---|---|---|
| OLMoE-1B-7B | x | x | x | DONE |
| Phi-3.5-MoE | x | x | x | DONE |
| Mixtral-8x7B | x | x | x | DONE |
| DBRX-132B | x | x | x | DONE |
| GPT-OSS-120B | x | x (2000) + x (5000 replication) | x | **DONE** --- both captures agree ($\Delta H = +0.0024$) |
| Gemma-4-26B | x | x | x | DONE (null-bias flag) |
| **Gemma-4-27B** | - | - | - | **PHANTOM** --- HF 404, model doesn't exist (26B is the real rung); non-issue |

### Experiment 2 --- Dense Baselines (RQ2/C4)
| Model | v0 (400) | v1 per-pair | CI | Status |
|---|---|---|---|---|
| OLMo-7B | x | x (1800 pairs) | x | **DONE**, integrated into Table 3 + Appendix A |
| Phi-3.5-Mini | x | x (4000 pairs) | x | **DONE** |
| Llama-2-7B | x | x (1800 pairs) | x | **DONE** |
| Llama-3.1-8B | x | x (1800 pairs) | x | **DONE** |

### Experiment 3 --- Collectivity Check (C2-lite)
**Integrated into paper this session** as Section 5.5 (Expert-pair synergy)
and an Appendix B paragraph. Landed data (verified non-degenerate) for
4 models: OLMoE-1B-7B (synergy fraction $0.705$/$0.281$, layer0/last),
Phi-3.5-MoE ($0.741$/$0.200$), Mixtral-8x7B ($0.716$/$0.503$, the
2026-07-07 stale capture --- checked, non-degenerate, $n{=}20$ pairs both
layers, used as-is), Gemma-4-26B ($0.747$/$0.183$). DBRX and
GPT-OSS-120B resubmitted this session (jobs **5575799**, **5575791**,
node-pinned around a cluster maintenance-window reservation after the
first attempts (5575743/5575744) were blocked --- see `CLUSTER-STATUS.md`
root-cause section) --- **RUNNING** as of last poll, `--time` reduced to
~1h10m to fit the maintenance-window runway (down from the original 4h/8h).
A redundant Mixtral refresh (job 5575538, from a prior session) was
**cancelled** this session after confirming it requested 1440 GPU-min
against the `coc-ice` 960 GPU-min/job cap and could never run; not
blocking (July capture already verified usable).

### Experiment 4 --- Independent Cross-Check (Robustness)
- **DONE** for 6 models (OLMoE, Mixtral, Phi-3.5-MoE, DBRX, Gemma-4-26B,
  OLMo-7B dense) under Exp6 (Section 5.6/causal check). GPT-OSS-120B
  ladder extension resubmitted this session (job 5575800, node-pinned).

### Experiment 5 --- Demographic Specificity (RQ3/C3)
- **DONE** for OLMoE (5000 pairs, 85 cohorts, JS CI [0.206,0.231]). Exp5 flag
  documented in paper (no Winogender in Exp5 vs ladder).

### Experiment 6 --- Ladder-Wide Causal Ablation (Reviewer-Prioritized)
| Model | Ablation Curve | Status |
|---|---|---|
| OLMoE-1B-7B | x (60 pairs) | DONE, cited in paper |
| Phi-3.5-MoE | x (30 pairs) | DONE, cited in paper |
| Mixtral-8x7B | x (30 pairs) | DONE, cited in paper |
| OLMo-7B (dense) | x (60 pairs) | DONE, cited in paper |
| DBRX | x (60 pairs) | **DONE, integrated this session** --- $\phi$ is the *worst*-performing ranking at the 50% point (reversal), reported honestly |
| Gemma-4-26B | x (60 pairs) | **DONE, integrated this session** --- baseline bias gap $\approx 0$, flagged uninterpretable, numbers reported for completeness only |
| GPT-OSS-120B | resubmitted (job 5575800, node-pinned, 1h/30 pairs) | **RUNNING** as of last poll |

**Gap remaining**: only GPT-OSS-120B's Exp6 ladder rung is still missing;
once it lands, update Section 5.6/Appendix B from "DONE for six models" to
"DONE for all six MoE rungs plus the dense control" and add its headline
numbers.

### Experiment 7 --- Proxy-vs-Exact Shapley Agreement
| Model | Result | Status |
|---|---|---|
| Mixtral-8x7B | mean rho=-0.085/+0.058 | **DONE** (null result), cited in paper |
| OLMoE | mean rho=-0.024/+0.235 | **DONE** (null result), now cited in paper |
| Phi-3.5-MoE | mean rho=-0.084/+0.076 | **DONE** (null result), now cited in paper |

**Fully resolved** --- all 3 tractable models now tested, all null, paper updated.

### Experiment 8 --- Same-Mechanism Comparison (Method-Confound)
| Model | Result | Status |
|---|---|---|
| OLMoE | H_lloo=0.7361, per-pair CI $[0.692,0.921]$ | **DONE with per-pair CI** (job 5575798, node-pinned around the maintenance window; $n=100$, bootstrap $n_{\mathrm{boot}}=5000$, seed 42); integrated into paper Appendix B |
| Phi-3.5-MoE | H_lloo=0.899, per-pair CI $[0.842,0.941]$ | **DONE with per-pair CI** (job 5575536, $n=50$, bootstrap $n_{\mathrm{boot}}=5000$, seed 42); integrated into paper Appendix B |

**Fully resolved** --- both models now have per-pair captures and bootstrap
CIs; the paper's Appendix B and Limitations flag report all 13 model
payloads as landed.

**New this session --- Exp8 ladder extension.** The 2-model result above is
an *ambiguous split* on its own ($n=2$: one point in the dense band, one in
the MoE range --- not yet a trend). Authored 3 new `dense_loo` configs to
extend the same-mechanism check across the sparsity ladder:
`configs/study.mixtral-8x7b.lloo.yaml`, `configs/study.dbrx.lloo.yaml`,
`configs/study.gpt-oss-120b.lloo.yaml` (all $n_{pairs}=30$--$50$, sized to
stay well under the 960 GPU-min/job QOS cap). Doing so surfaced two real
bugs in `discover_dense_ffn_layers`/`compute_dense_layer_contrast`
(`src/moe_bias_shapley/hooks.py`, `shapley.py`) that would have made the
DBRX and GPT-OSS configs crash or silently misbehave:
1. **DBRX uses `.ffn`, not `.mlp`**, and its decoder stack lives at
   `transformer.blocks`, not `model.layers`/`transformer.h`/`gpt_neox.layers`
   (verified against upstream `modeling_dbrx.py`). Fixed by rewriting
   `discover_dense_ffn_layers` to walk `model.named_modules()` generically
   (mirroring `discover_moe_layers`'s existing approach) instead of a
   hardcoded path/attribute list.
2. **GPT-OSS's `GptOssMLP.forward` returns a 2-tuple**
   `(hidden_states, router_scores)`, unlike Mixtral/OLMoE/Phi-3.5-MoE's
   single-tensor return; the LOO ablation's `zero_forward` monkey-patch
   returned a bare zeroed tensor unconditionally, which would crash the
   decoder layer's `hidden_states, _ = self.mlp(...)` unpacking. Fixed by
   probing each layer's real output arity once (cheap, cached per layer)
   and matching it in the zeroed replacement.

Both fixes verified with a synthetic-module smoke test
(`/tmp/smoke_discover.py`, not checked in --- exercises exact attribute
names/container paths/return shapes sourced from upstream transformers
source, not guessed) before shipping the configs; not yet run against the
real 100+GB checkpoints (blocked on cluster access).

**Gemma-4-26B excluded from this extension.** Its decoder layer forks into
*two* parallel FFN branches (`self.shared_expert`, always-on dense; and
`self.moe`, routed experts) that are summed, not a single ablatable module
--- the current single-attribute LOO mechanism would under-ablate (zero one
branch, leave the other active) rather than measure the true per-layer
contribution. Left out rather than shipping a misleading number; a correct
extension would need a compound-ablation code path (zero both branches
together), not attempted this session.

---

## 3. Critical Gaps for Paper Acceptance (TIST/ACM)

### Resolved this session
1. ~~Dense CIs~~ --- **DONE**, integrated into Table 3 + Appendix A.
2. ~~GPT-OSS 5000 pairs~~ --- **DONE**, stability replication confirmed and cited.
3. ~~s04 bootstrap glob fix~~ --- **DONE** (prior session).
4. ~~Exp7 on OLMoE/Phi~~ --- **DONE**, all 3 models now null, paper updated.
5. ~~Exact-permutation floor math error~~ --- **DONE**: corrected $1/12
   \approx 0.083$ (wrong) to the true tie-corrected minima $0.0056$
   ($n{=}6$)/$0.0333$ ($n{=}5$) in all 5 locations; added
   `Section 6.4 Power analysis` (Monte Carlo).
6. ~~Effect sizes~~ --- **DONE**: new `Section 5.4` reports Cohen's $d$
   for entropy/Gini/$t_5$ and the bias-gap-magnitude-parity finding
   ($d=0.011$), plus per-model localizability ratio ($LR$).
7. ~~Exp3 Collectivity (partial)~~ --- **DONE for 4 models**, integrated
   as new `Section 5.5`; mechanistically explains the causal-ablation
   reversals (DBRX, Mixtral).
8. ~~Exp6 ladder extension (DBRX, Gemma-4-26B)~~ --- **DONE**, integrated
   into the causal-check Results/Verdict and Appendix B; the extended
   ladder *weakens* the causal reading (DBRX reversal), reported honestly.
9. ~~Split-half shard exact numbers~~ --- **DONE**, added to Appendix A
   with precise $H$/$G$ per shard for Mixtral and DBRX.

### Still open (GPU-bound; root cause found and worked around this session)
10. **Exp3 Collectivity (DBRX, GPT-OSS-120B)** --- resubmitted node-pinned
    (jobs **5575799**, **5575791**) after discovering the earlier
    attempts (5575743/5575744) were blocked by a real, non-admin-visible
    cluster maintenance-window reservation rather than the QOS cap or a
    scheduler bug (see `CLUSTER-STATUS.md` root-cause section).
    **RUNNING** as of last poll, `--time` reduced to ~1h10m to fit the
    reservation's runway.
11. **Exp6 GPT-OSS-120B** --- resubmitted node-pinned (job **5575800**)
    with the same maintenance-window workaround. **RUNNING** as of last
    poll, `--time=01:00:00`.
12. ~~Exp8 per-pair (both models)~~ --- **DONE**. Phi-3.5-MoE (job
    5575536, $n=50$) integrated earlier this session; OLMoE's per-pair
    capture (job **5575798**, node-pinned) landed with $H=0.7361$,
    Gini$=0.6239$, CI $H \in [0.692, 0.921]$, integrated into Appendix B.
13. **Exp8 ladder extension (Mixtral, DBRX, GPT-OSS-120B)** --- 3 new
    `dense_loo` configs authored + the underlying discovery/ablation code
    bugs fixed and smoke-tested (see Section 2 above); **not yet
    submitted** (blocked on cluster access during the maintenance window).
    Ready to fire via `submit_slurm_study.py --config
    configs/study.{mixtral-8x7b,dbrx,gpt-oss-120b}.lloo.yaml
    --save-per-pair-phi` the moment the cluster reopens.

### Data/Code Hygiene
14. ~~Gemma-4-27B phantom~~ --- documented, non-issue.
15. ~~Kaggle payload manifest~~ --- done (`sghose0/moe-bias-routing-shapley-perpair-phi`).
    Dense v1 per-pair payloads are NOT yet published there --- consider a
    follow-up dataset version once Exp3/6/8 fully land.
16. ~~REPRODUCIBILITY.md~~ --- done.

---

## 4. GPU Resource Plan (ICE Cluster)

See `CLUSTER-STATUS.md` for the live job table (job IDs, states, notes). This
file is not duplicated here to avoid drift between the two docs.

---

## 5. Execution Priority (remaining)

1. **BLOCKED until 2026-08-13 23:59**: PACE ICE is in its scheduled
   quarterly maintenance window (2026-08-11 06:00 -- 2026-08-13 23:59,
   confirmed via the login banner; login itself is refused cluster-wide,
   not just job scheduling). The 3 jobs that were RUNNING when the
   window closed (5575799 DBRX Exp3, 5575791 GPT-OSS Exp3, 5575800
   GPT-OSS Exp6) had unknown-but-healthy-at-last-poll status; OLMoE Exp8
   per-pair (job 5575798) already **COMPLETED** and is integrated. Job
   5575538 (redundant Mixtral refresh) was cancelled: it exceeded the
   960 GPU-min/job QOS cap and could never run. **First action once the
   cluster reopens**: `sacct -j 5575799,5575791,5575800
   --format=JobID,State,ExitCode,Elapsed -X -n`; pull results if
   COMPLETED, resubmit (plain `sbatch`, no node-pinning needed once
   maintenance is over) if TIMEOUT/CANCELLED/NODE_FAIL. See
   `CLUSTER-STATUS.md` for full detail and exact submit lines.
2. **Pull results** into local `results/` as each job completes.
3. **Integrate GPT-OSS-120B's Exp3/Exp6 numbers** into Section 5.5/5.6 and
   Appendix B once landed (currently the only two "PENDING" placeholders
   left in the causal/collectivity evidence chain).
4. **Recompile + vision-check** after that integration pass.
5. **Final commit + push**, then re-audit against every deliverable before
   calling the goal complete.

*Document generated from live repo state, study catalog, cluster inventory
(squeue polled directly this session), and paper draft audit (2x pdflatex
compile + page-image inspection of pages 3, 5, 6, 8, 10, 11).*

---

## 6. Third-pass audit: citations, anonymity, ethics/data-availability (this session)

**Date**: 2026-08-11 (re-poll), cluster still unreachable.

A close read of the front/back matter (not just the numeric claims already
audited in Sections 0-3) found four real gaps missed by the prior two
passes, all now fixed in `moe_bias_report_acm_v2.tex` and pushed:

1. **Missing citations for 3/6 ladder models + 1/3 benchmarks.** Related
   Work named all six MoE models (Mixtral, OLMoE, Gemma, DBRX, Phi-3.5-MoE,
   GPT-OSS) and all three benchmarks (StereoSet, BBQ, Winogender) in prose,
   but only had `\bibitem`s for 3 models and 2 benchmarks -- DBRX,
   Phi-3.5-MoE, GPT-OSS, and Winogender were used throughout the study
   with zero citation. Added 4 bibitems (Databricks DBRX blog post, Abdin
   et al. Phi-3 technical report `arXiv:2404.14219`, OpenAI gpt-oss model
   card `arXiv:2508.10925`, Rudinger et al. Winogender NAACL-HLT 2018) and
   wired `\cite{}` into the listing sentence.
2. **No Data Availability disclosure for the ~360MB per-pair payloads.**
   The `Data \& Code` section pointed only at `results/*/result.json`,
   which does not carry the per-pair phi arrays (gitignored, Kaggle-hosted
   per `CLUSTER-STATUS.md` Sec.\ "Kaggle data release"). Fixed the claim to
   state the payloads exceed the code repo's size limit and are hosted
   externally -- **without** naming the Kaggle owner slug, since the
   dataset owner name would deanonymize the paper (see next point).
3. **No Ethical Considerations section.** Standard expectation for
   bias-evaluation papers; added one scoping the metric as structural
   (not a safety/deployment-risk measure), cross-referencing
   Section~5.6's causal-check finding that phi-ranking is a marginal, not
   exact, proxy, and noting no new human-subjects data or released
   weights/jailbreak artifacts.
4. **Anonymity leak: "our prior work" self-citation.** The paper uses
   `\author{Anonymous}` (double-blind submission) but Related Work said
   "the routing-Shapley decomposition introduced in **our** prior work
   [cite]" -- a known de-anonymization vector (self-citation + first-person
   possessive). Reworded to "introduced in prior work [cite]". Grepped the
   full draft for cluster hostnames, usernames, and institution names
   (`sghose`, `pace.gatech`, `login-ice`, `hice1`, `RONNCC`, `kaggle.com`,
   etc.) -- none present elsewhere.

**Verification**: recompiled 2x pdflatex after each fix (clean, 0 errors,
11 pages, 0 undefined refs each time); vision-checked the related-work
page, the Data\&Code/Ethics page, and the bibliography page via
`pdftocairo` raster + image read. Cross-checked every "landed" claim in
the current draft (5000-pair GPT-OSS replication $H=0.8789$, dense-baseline
per-model CIs, Exp3/6/7/8 model counts) directly against
`stats_analysis/outputs/s04_bootstrap_cis.json` and the `results/*/experiment{3,6,7}/*.json`
files on disk -- all match; nothing in the draft is stale or fabricated.

**Re-polled cluster this session** (`ssh -o BatchMode=yes
login-ice.pace.gatech.edu`): still refused (`Connection timed out during
banner exchange`, i.e.\ no TCP response at all -- consistent with the
confirmed maintenance window closing the login service outright, not just
job scheduling). No further paper- or repo-side gaps found after this
pass; the only remaining open items are the three GPU jobs blocked on the
PACE maintenance window per Section 5 above.
