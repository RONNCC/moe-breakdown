# PhD-Style Gap Analysis: MoE-Bias Study (expt-bias-1)

**Date**: 2026-09-10 (updated after fourth-pass ML-review audit; historical sections retain their original session dates)
**Paper status**: 11-page ACM draft at `moe-expt-bias-2/moe_bias_report_acm_v2.tex` (the prior compile/visual check is recorded below; the new acceptance blockers in Section 8 require resolution before submission)
**Repo**: repository root (this checkout: `/home/user/moe-breakdown`; do not rely on the historical machine-specific path)

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
the MoE range --- not yet a trend). Authored 4 new `dense_loo` configs to
extend the same-mechanism check across the sparsity ladder:
`configs/study.mixtral-8x7b.lloo.yaml`, `configs/study.dbrx.lloo.yaml`,
`configs/study.gpt-oss-120b.lloo.yaml`, `configs/study.gemma4-26b.lloo.yaml`
(all $n_{pairs}=30$--$50$, sized to stay well under the 960 GPU-min/job
QOS cap). Doing so surfaced three real bugs in
`discover_dense_ffn_layers`/`compute_dense_layer_contrast`/
`mean_bias_gap_with_players_ablated` (`src/moe_bias_shapley/hooks.py`,
`shapley.py`) that would have made 3 of the 4 configs crash or silently
misbehave:
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
3. **Gemma-4's decoder layer forks into *two* parallel FFN branches**
   (`self.shared_expert`, always-on dense; `self.moe`, routed experts) that
   are summed before the residual add, not a single ablatable module --- the
   single-attribute LOO mechanism would under-ablate (zero one branch, leave
   the other active), undercounting the layer's true contribution. Fixed by
   adding `_COMPOUND_FFN_ATTR_GROUPS = (("shared_expert", "moe"),)` to
   `discover_dense_ffn_layers`: when a layer matches every attribute in a
   group, `DenseLayerHandle` now carries the extra branch(es) in a new
   `extra_modules` field, and both `compute_dense_layer_contrast` (Exp8) and
   `mean_bias_gap_with_players_ablated` (Exp4) zero every module in the
   group together (probing each branch's own output arity independently,
   since `shared_expert` and `moe` need not share a forward signature).
   Single-branch architectures (Llama/OLMo/Mixtral/DBRX/GPT-OSS) are
   unaffected --- none match the compound group.

All three fixes verified with a synthetic-module smoke test
(`/tmp/smoke_compound_ablation.py`, not checked in --- exercises exact
attribute names/container paths/return shapes already established earlier
this session, not guessed) that (a) confirms discovery finds both Gemma
branches as one player and leaves single-branch layers unchanged, (b) shows
the old single-attribute-only ablation leaves a nonzero residual (the bug
this fixes) while the new compound path zeroes the full contribution, and
(c) runs the real `compute_dense_layer_contrast` function end-to-end (not a
hand-simulation) against a mixed compound + single-branch synthetic ladder,
checking per-pair phi shape, NaN-freedom, and post-run forward restoration
on every branch. Not yet run against the real 240GB+ checkpoints (blocked on
cluster access) --- all 4 configs (including `gemma4-26b`) are ready to fire
the moment the cluster reopens; none are excluded any more.

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
   ($d=0.011$), plus per-model localizability ratio ($LR$). LR now also
   has full uncertainty quantification and tests (offline `s07`): per-rung
   and matched-family bootstrap CIs (all exclude $1$; OLMoE-vs-OLMo-7B
   $0.800$ [$0.763,0.844$], Phi-MoE-vs-Phi-Mini $0.863$ [$0.832,0.884$]),
   an exact model-level permutation test over all C(10,4)=210 labelings
   ($p=0.0048$, complete separation: max dense H $0.7592$ < min MoE H
   $0.7888$), and a 6-of-6 sign test ($p=0.0156$) --- integrated into
   the main-text `Section 5.4` sentence and Appendix A. The bias-gap-
   magnitude-parity claim ("statistically indistinguishable", previously
   test-free) now has model-level tests too (offline `s07`, from
   `result.json` run-level `mean_bias_gap`): exact permutation
   $p=0.992$ over C(9,4)=126 labelings, Welch $p=0.986$, Cohen's
   $d=-0.011$ (4 dense vs 5 non-null MoE); incl.\ Gemma's signed value
   $p=0.66$ over C(10,4)=210 --- still indistinguishable. The only
   parity-level artifact still GPU-bound is the per-pair bias-gap
   bootstrap (gap values aren't recoverable from `per_pair_phi.npy`).
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
13. **Exp8 ladder extension (Mixtral, DBRX, GPT-OSS-120B, Gemma-4-26B)**
    --- 4 new `dense_loo` configs authored + the underlying
    discovery/ablation code bugs fixed and smoke-tested (see Section 2
    above, including the Gemma compound-branch fix); **not yet submitted**
    (blocked on cluster access during the maintenance window). Ready to
    fire via `submit_slurm_study.py --config
    configs/study.{mixtral-8x7b,dbrx,gpt-oss-120b,gemma4-26b}.lloo.yaml
    --save-per-pair-phi` the moment the cluster reopens.

### Data/Code Hygiene
14. ~~Gemma-4-27B phantom~~ --- documented, non-issue.
15. ~~Kaggle payload manifest~~ --- done (`sghose0/moe-bias-routing-shapley-perpair-phi`).
    ~~Dense v1 per-pair payloads~~ --- **RESOLVED (2026-08-11)**: dataset
    version 2 published, 15 files (~565MB) covering all payloads generated
    to date (4 dense v1 baselines, GPT-OSS-120B 5000-pair replication,
    Exp8 LOO x2), verified via `kaggle datasets files`.
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

---

## 7. Sept-10 refresh: cluster-open status and reviewer punchlist (2026-09-10)

**Date**: 2026-09-10. Append-only; Sections 0-6 untouched history.

### 7.1 Cluster-open status and first action

- Maintenance window ended 2026-08-13 23:59. No sacct re-poll on record since 08-11.
- Three jobs UNKNOWN at last poll: 5575799 (DBRX Exp3), 5575791 (GPT-OSS Exp3), 5575800 (GPT-OSS Exp6). OLMoE Exp8 per-pair (5575798) already COMPLETED and integrated.
- Exp8 LLOO ladder extension: 4 configs ready, NOT-SUBMITTED (`configs/study.{mixtral-8x7b,dbrx,gpt-oss-120b,gemma4-26b}.lloo.yaml`).
- First action on cluster reopen (exact):
- `sacct -j 5575799,5575791,5575800 --format=JobID,State,ExitCode,Elapsed -X -n`
- Pull results if COMPLETED; resubmit plain `sbatch` (no node-pinning) if TIMEOUT/CANCELLED/NODE_FAIL.

### 7.2 Sept-10 20-item punchlist (condensed)

Experiment rows:
1. Exp3 DBRX (job 5575799) UNKNOWN; integrate synergy fraction into Sec 5.5/App B on landing.
2. Exp3 GPT-OSS (job 5575791) UNKNOWN; same integration.
3. Exp6 GPT-OSS (job 5575800) UNKNOWN; update Sec 5.6/App B from six models to full ladder plus dense control on landing.
4. Exp8 LLOO x4 NOT-SUBMITTED; fire via `submit_slurm_study.py --config configs/study.{mixtral-8x7b,dbrx,gpt-oss-120b,gemma4-26b}.lloo.yaml --save-per-pair-phi` after sacct triage.
5. GPT-OSS gaps: Exp3/Exp6 placeholders remain only PENDING items in causal/collectivity chain.

Stats blockers:
6. Gemma CI miss: add CIs or flag null-bias uninterpretable everywhere claim appears.
7. t5 d: report Cohen d for top-5-share alongside entropy/Gini in Sec 5.4.
8. Dense comparability: state dense vs MoE comparability caveat (different mechanism, matched families only).
9. Observed power: relabel post-hoc power; report observed power only, no design claim.
10. SUPPORTED label: rename H1 SUPPORTED to qualified wording (directionally consistent, underpowered).
11. Spearman r2: report r-squared alongside rho_H where trend claimed.
12. Multiplicity: disclose multiple-comparison handling or flag uncorrected tests.

Venue items:
13. 8 missing cites: Covert21, Sundararajan17, Lundberg18 interaction, Zhou22, Nangia20, Zhao18, Gallegos24, Frantar23.
14. Fig fixes: no CIs on figs; Fig3 Description; Fig4 N-comparability note; Fig5 n=4 boxplot replace or flag.

Repro items:
15. Kaggle v3: registry shows 8 vs 15 files on disk; publish v3 covering all payloads.
16. Run metadata fields: record seed, config hash, commit, GPU type per run.
17. .aux hygiene: remove or gitignore LaTeX .aux build artifacts.
18-20. Rolled into priority order below: recompile gate, final commit plus push, re-audit every deliverable.

### 7.3 Proposals: new rung and within-model k-sweep

- New rung: add one more MoE rung to grow ladder n=6 toward n~8-9. Power rationale: Monte Carlo Sec 6.4 needs n~12-13 at rho_H=0.754, n~8-9 at rho_H=0.872; each added rung cuts sampling noise.
- Within-model k-sweep: vary top-k or active-expert count inside one model family, same checkpoints, same pairs. Tests routing-structure mechanism without new weights; adds powered within-model points alongside ladder.

### 7.4 Priority order

1. sacct triage jobs 5575799/5575791/5575800.
2. Submit Exp8 LLOO x4.
3. Close GPT-OSS Exp3/Exp6 gaps.
4. Stats fixes (items 6-12).
5. Cites and figs (items 13-14).
6. Kaggle v3 plus registry plus metadata plus aux (items 15-17).
7. Recompile gate: 2x pdflatex clean, 0 undefined refs, vision-check, then commit plus push.

---

## 8. Fourth-pass NeurIPS/ICML-style review: construct validity and acceptance blockers (2026-09-10)

### 8.1 Overall assessment

**Provisional recommendation: major revision / reject in current form.** The project has unusually good run tracking, transparent null results, and several useful robustness checks. However, a source-level review found issues deeper than the outstanding GPU jobs. The central reported object is not a Shapley value, the benchmark adapter does not preserve the native constructs for BBQ or WinoGender, and the compared runs do not share the benchmark mixture claimed in the paper. These are acceptance-blocking construct-validity problems: more models, more pairs, and tighter confidence intervals cannot repair them without re-estimation.

The most defensible paper after repair may be narrower and stronger: **an audit showing that an observational routing-contrast heuristic is diffuse and does not recover causal expert importance**, rather than a paper claiming a routing-Shapley decomposition of social bias. Experiment 7's null proxy/exact agreement and Experiment 6's mixed deletion curves are central results under that framing, not secondary caveats.

### 8.2 Acceptance-blocking findings newly identified

#### A. The primary estimator is not a Shapley estimator (**fatal unless renamed or replaced**)

`compute_routing_contrast` in `src/moe_bias_shapley/shapley.py` computes

```text
(mean_router_weight_stereo - mean_router_weight_anti) * whole-pair_bias_gap
```

and never defines or evaluates a coalition value function `v(S)`. A Shapley value is a weighted average of marginal coalition differences `v(S union {i}) - v(S)`; the routing-contrast score therefore does not inherit Shapley efficiency, symmetry, dummy, or additivity. Calling it “routing-Shapley,” saying it is “computed over coalitions,” or presenting it as a decomposition/partition of the output gap is mathematically unsupported. “RGIS-style” is also not enough: an importance-sampling Shapley approximation must still estimate coalition marginals and document its proposal distribution and weights.

This is not a semantic nit. Experiment 7 reports near-zero agreement between the proxy and exact causal Shapley rankings on all tested models. That evidence directly falsifies the interpretation of the primary score as an approximation to the target estimand within the tested budget.

**Required action:** choose one of two paths.

1. **Rename/reframe path (recommended):** replace “Shapley” for Exp1/Exp5 with “routing-contrast heuristic/score”; remove all Shapley-axiom and payoff-partition claims; retitle the paper; make proxy invalidation the headline; reserve “Shapley” for the actual coalition intervention experiments. Report fidelity to causal effects rather than implying it.
2. **Estimator replacement path:** define `v(S)` precisely (including router renormalization, shared experts, token/layer scope, and absent-expert semantics), estimate genuine interventional Shapley values with uncertainty on a tractable preregistered subset, and rerun the primary analysis. Validate local accuracy/efficiency numerically.

Reference context: Lundberg and Lee define SHAP through coalition-conditioned marginal contributions and local accuracy; Covert, Lundberg, and Lee stress that a removal explanation requires explicit choices of removal operator, behavior, and summary ([NeurIPS 2017](https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf); [JMLR 2021](https://jmlr.org/papers/volume22/20-1316/20-1316.pdf)).

#### B. BBQ conversion does not identify the stereotyped answer (**rerun required**)

In `benchmarks.load_bbq`, `label` is correctly identified as the unknown answer for ambiguous items, but `biased_ans` is then set to the **first arbitrary non-unknown answer**. BBQ has two non-unknown group answers; which one aligns with the stereotype depends on `target_loc`/answer metadata and `question_polarity`. The loader ignores both. Consequently, an unknown fraction of “stereo” prompts are non-target answers, and positive/negative question variants can have reversed interpretation.

**Required action:** reconstruct the target answer using the official answer metadata (`target_loc`, or an equivalent derivation from `answer_info`) and question polarity; unit-test all answer permutations; reproduce the official BBQ ambiguous-context bias score on frozen model logits before using the adapter for attribution. Retain both target and non-target contrasts rather than selecting an arbitrary distractor. All Exp1, Exp2, Exp5, and causal runs containing BBQ must then be rerun or explicitly excluded.

The official BBQ repository states that `target_loc` is the answer-option index used to compute bias score and that ambiguous and disambiguated conditions answer different questions ([NYU BBQ repository](https://github.com/nyu-mll/BBQ)).

#### C. WinoGender labels male as stereotypical unconditionally (**rerun or remove**)

`load_winogender` always writes the male-pronoun sentence to `stereo` and the female-pronoun sentence to `anti_stereo`. Male is not intrinsically the stereotyped member for every occupation, and the source benchmark evaluates occupational gender correlations rather than this fixed direction. The code comment acknowledges that the implementation is not native WinoGender scoring but does not solve the sign problem.

**Required action:** derive stereotype direction from the released occupation statistics/metadata, or treat the pair as an unsigned counterfactual sensitivity test and never call its sign a stereotype gap. Add neutral-pronoun and participant/occupation referent controls. Report WinoGender separately before pooling.

#### D. The paper's “common prompt battery” does not exist in the saved runs (**re-estimation required**)

Inspection of tracked `pair_meta.json` files gives:

- Five 5000-pair MoE v1 captures: 2,106 StereoSet + 2,894 BBQ + **0 WinoGender**.
- GPT-OSS v1 (2,000 pairs): **2,000 StereoSet only**.
- Dense v1 captures: OLMo/Llama-2/Llama-3.1 each **1,800 StereoSet only**; Phi-3.5-Mini 2,106 StereoSet + 1,894 BBQ.

The cause is deterministic concatenation followed by `pairs[:max_items]` in `load_benchmarks`; the configured `seed` is recorded but never used for benchmark sampling. Thus model comparisons conflate architecture with benchmark composition, and claims that all runs use StereoSet/BBQ/WinoGender are false. The signed aggregate can also change with arbitrary dataset order and cap.

**Required action:** create a frozen item manifest shared by every model, balanced or explicitly weighted by benchmark/category, with stable IDs and a documented sampling rule. Run all models on the same items. At minimum, recompute existing comparisons on their exact common intersection (currently StereoSet-only) and report per-benchmark estimates and heterogeneity. Do not describe the current first-`N` selection as seeded sampling.

#### E. The advertised stratified bootstrap is actually one stratum (**recompute CIs**)

`s04_bootstrap_cis.py::load_pair_meta` evaluates `e.get("group", e.get("benchmark", "unknown"))`. Every saved entry has a present but null `group`, so the benchmark fallback is never taken; `str(None)` assigns every pair to one stratum. This was verified on all tracked v1 manifests. Therefore the paper's repeated “stratified by prompt group” statement is inaccurate, and benchmark/category composition uncertainty is not represented.

**Required action:** treat null/empty group as missing and fall back to benchmark, preferably benchmark × bias category; add a test with null groups; recompute every CI. Because prompts generated from the same StereoSet context or BBQ template are related, consider a cluster bootstrap at source-item/template level rather than an IID pair bootstrap. Compare percentile with BCa/studentized intervals or explain the choice.

#### F. The bias payoff is not benchmark-native and mixes incomparable quantities

`_sequence_logprob` scores mean teacher-forced log probability over each **entire completed string**. For BBQ this includes context, question, and an answer of variable token length, diluting answer evidence and introducing lexical/length effects. StereoSet's native evaluation also includes an unrelated option and reports language-model and stereotype components; the loader discards the unrelated sentence. The final pooled signed mean mixes sentence association, QA unknown-vs-group preference, and a non-native pronoun contrast.

**Required action:** score conditional completion log probability only (same prefix, answer tokens only; length convention preregistered), reproduce each benchmark's native aggregate metrics, and keep constructs separate in the main analysis. Add prompt-format/tokenization sensitivity and an unrelated/meaningfulness control for StereoSet. Report the pooled result only as a prespecified meta-analysis with weights and heterogeneity, not as a raw mean. StereoSet's LMS/SS/ICAT design explicitly uses the unrelated option to separate meaningful language modeling from stereotype preference ([StereoSet paper](https://arxiv.org/pdf/2004.09456)).

### 8.3 Major statistical and experimental gaps

#### G. Cross-model sparsity is observational and heavily confounded

The six points differ simultaneously in total experts, active experts, layers, expert width, shared-expert design, model scale, training corpus, load-balancing objective, instruction tuning, precision, and model family. `k/N` also repeats at 0.25 and is partly constructed by summing slots over layers. Spearman correlation across six non-exchangeable architectures cannot identify a causal effect of sparsity. Model-level permutation assumes exchangeability that is not credible across related model families and heterogeneous measurement protocols.

**Required action:** phrase H1 as an observational cross-architecture association. The decisive experiment is a controlled top-`k` study on checkpoints trained with multiple `k` values (ideally several training seeds), not merely changing `k` at inference on one fixed checkpoint. An inference-time `k` sweep is still useful as a mechanism stress test, but it changes both compute and distribution relative to training and must not be presented as equivalent to trained sparsity. Factorial controls should separately vary total `N`, active `k`, expert capacity, and load-balancing.

Routing design itself can alter load balance and specialization, so `k/N` is not a sufficient architecture descriptor; Expert Choice explicitly documents under/over-specialization and load-balancing effects ([NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf)).

#### H. Player cardinality/partition makes entropy and top-5 comparisons non-identifiable

Dividing entropy by `log(N)` bounds it in `[0,1]`; it does **not** make expert-level and layer-level partitions equivalent. Splitting one causal component into many correlated players can increase normalized entropy without changing the function. Fixed top-5 share is especially mechanical (5/32 dense layers versus 5/256–4608 MoE experts), as the draft partially acknowledges. Zero padding to `max_experts` and cross-layer flattening add further architecture dependence.

**Required action:** do not infer an architectural localizability gap from unlike partitions. Complete Exp8 for all models and compare the same intervention unit; add matched-cardinality aggregation/subsampling, per-layer expert entropy followed by a hierarchical summary, effective support size `exp(H_raw)`, and top-`q` mass at common fractions. Include synthetic split/merge controls demonstrating how metrics move when the represented function is unchanged. The dense/MoE result is currently a measurement-granularity result, not evidence of physical localization.

#### I. “Bias magnitude parity” is not established

The reported parity test uses one signed `mean_bias_gap` per model. Signed averaging can cancel positive and negative categories, and the runs have different benchmark mixtures (Section 8.2D). Calling this “absolute contrastive bias-gap magnitude” conflicts with the source field, which is `np.mean(gaps)`, not `np.mean(abs(gaps))`. Failure to reject with 4 versus 5 heterogeneous model summaries is not equivalence.

**Required action:** retract “statistically indistinguishable/parity” until rerun on a common item set. Report signed and absolute item-level effects by benchmark/category, uncertainty, and a smallest effect size of interest. If parity is a claim, use an equivalence test or interval against that margin; do not interpret a large null-hypothesis p-value as evidence of equality.

#### J. Multiplicity and post-selection are not controlled

The draft tests H and Gini across multiple overlapping subsets, reports top fractions, localizability ratios, bias magnitude, two matched families, layers, cohorts, and several ablation fractions. “Every defensible subset” is post hoc and the subsets are strongly dependent. No family of confirmatory hypotheses or multiplicity policy is declared.

**Required action:** designate one primary endpoint, one primary model set, and one exclusion policy before the next run. Mark all other analyses exploratory; report all tests in a machine-readable table and control FDR/FWER where claims depend on them. Prefer effect intervals to “supported/rejected” labels. Do not report `rho^2` as variance explained for a six-point Spearman rank statistic; it has no useful regression interpretation here.

#### K. Post-hoc power is circular

Power simulated at the observed `rho` is optimistic and unstable at `n=6`; the stronger no-GPT effect is selected after inspecting the data. The statement that 8–13 rungs “would be needed” is therefore not a defensible prospective design calculation.

**Required action:** label it a sensitivity analysis only. Plan future sample size using a smallest scientifically meaningful effect and uncertainty over effect size, or simulate the proposed hierarchical/controlled design. Increasing pair count improves within-model precision but does not increase the number of independent architectures.

### 8.4 Causal/interpretability gaps

#### L. Zero ablation is an out-of-distribution intervention and “capability” is under-measured

Zeroing 10–50% of expert outputs creates states not seen during training, can cause generic residual-stream disruption, and does not model how a deployed system might renormalize or reroute around unavailable experts. Perplexity measured on the same bias prompts is not “general language capability.” A single random ranking is also a weak null, and normalized disparity drop becomes unstable when the baseline signed gap is small.

**Required action:** add many size-matched random sets with confidence bands; compare zeroing, router masking with weight renormalization, mean-output replacement, and matched-magnitude noise; measure held-out general capability (at least a clean LM corpus plus several task suites); report deletion AUC with paired uncertainty and absolute gap changes, not only ratios. Repeat on independent prompt samples and seeds. Interpret ablation as a specific intervention, not literal removal of stored social bias.

Removal-based explanation literature requires the missingness operator to be explicit, and warns that different removal choices answer different questions ([Covert et al., JMLR 2021](https://jmlr.org/papers/volume22/20-1316/20-1316.pdf)). Recent preprint evidence also directly reports that population routing statistics need not predict token-level interventional importance; this supports the project's reframed null but should be cited as concurrent, non-peer-reviewed work ([Engmann et al., arXiv:2606.10703](https://arxiv.org/abs/2606.10703)).

#### M. The “synergy fraction” needs validation and uncertainty

The ratio `sum(abs(Phi_ij)) / (sum(abs(phi_i)) + sum(abs(Phi_ij)))` is not automatically a conserved fraction of output attribution. Depending on the interaction-index convention, main and pairwise effects can overlap or interactions can be counted twice. Results use 20 pairs, two cherry-prone depth locations, no CIs, and four models, yet the prose says “universal” and uses Gemma's null-bias run as mechanistic evidence. DBRX's ablation reversal cannot be explained by DBRX synergy because that run has not landed.

**Required action:** specify the exact interaction index and normalization; verify efficiency on synthetic additive and interacting games; clarify diagonal/main-effect allocation and double counting; bootstrap prompts; sample multiple layers selected a priori; correlate interaction mass with causal-ranking fidelity across model/layer cells. Replace “universal” and causal explanations with descriptive language until replicated. Shapley-Taylor is one principled interaction decomposition to compare against ([Sundararajan, Najmi, ICML 2020](https://arxiv.org/abs/1902.05622)).

#### N. Exp5's null does not test demographic specialization

The expert-index permutation destroys coordinate alignment and naturally raises JS distance; observing a lower cohort-to-pooled distance is not evidence against demographic structure. Each cohort is also included in its pooled comparator, biasing distance downward. The correct label-permutation null is acknowledged as unavailable. Exp5 uses one model, no WinoGender, uneven and sometimes tiny cohort counts, and routing-contrast rather than validated causal attribution.

**Required action:** treat the current Exp5 result as inconclusive. Persist item-level labels, use leave-one-cohort-out pooled references, permute cohort labels within benchmark/template strata, enforce minimum cohort sizes, and report shrinkage estimates with multiplicity control. Replicate across models and evaluate stability on held-out items. Since the estimator is not causal, call this routing-profile heterogeneity unless causal interventions validate it.

### 8.5 Scope, literature, and reproducibility gaps

#### O. Fairness construct and harm model are underspecified

The paper moves from likelihood preference on U.S.-centric intrinsic benchmarks to claims about “social bias,” “fairness,” “alignment,” and viable mitigation. It does not specify a deployment context, affected population, or downstream harm, and Gemma's near-zero signed score is called proof that post-training alignment is a “robust defense.” That conclusion is not supported by three adapted intrinsic benchmarks.

**Required action:** scope claims to stereotype-association behavior on the evaluated English datasets. State who/what the measurement is intended to protect and which harms it does not measure. Remove claims of general safety/alignment or mitigation success. Add disaggregated category results and limitations for binary gender, U.S. cultural specificity, prompt toxicity, and instruction/base model comparability. Blodgett et al. explicitly recommend connecting “bias” measures to harms, affected groups, and normative reasoning ([ACL 2020](https://aclanthology.org/2020.acl-main.485/)).

#### P. Related work misses the closest methodological competitors

The bibliography covers basic MoE and benchmark papers but does not engage deeply with removal-based explanation validity, attribution sanity checks, interaction indices, expert pruning/causal audits, or benchmark construct critiques. Add at least:

- Covert, Lundberg, and Lee, *Explaining by Removing* (JMLR 2021).
- Sundararajan and Najmi, *The Many Shapley Values for Model Explanation* / Sundararajan et al., *Shapley-Taylor* (clarify the exact value and interaction game used).
- Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018), adapted as parameter/label randomization controls.
- Hooker et al., *ROAR* (NeurIPS 2019), for removal-induced distribution shift.
- Zhou et al., *Expert Choice Routing* (NeurIPS 2022), for routing/load-balance confounds.
- Blodgett et al. (ACL 2020) and benchmark-validity work, for construct scope.
- Concurrent MoE interpretability papers should be labeled preprints and contrasted rather than used as settled motivation. For example, MoE-X argues that a specially trained architecture can be intrinsically interpretable, which does not imply that off-the-shelf experts are localizable ([arXiv:2503.07639](https://arxiv.org/abs/2503.07639)); Engmann et al. report observational-routing/causal-importance disagreement ([arXiv:2606.10703](https://arxiv.org/abs/2606.10703)).

#### Q. Reproducibility is not yet submission-grade

There is no automated test suite for benchmark semantics or Shapley axioms; model/dataset revisions are not pinned by immutable hashes; the runner records a seed it does not use; configs and result files do not consistently record code commit, config hash, package lock hash, GPU, precision, model revision, dataset revision, and item-manifest hash. The paper's self-reference `MoE-Bias-Research-Group (2026)` is not an independently retrievable scholarly source. Generated LaTeX files are tracked, and data availability lacks a durable anonymous URL/checksum manifest.

**Required action:** add unit/integration tests for every loader and metric; save an immutable item manifest; record all provenance fields; provide a one-command CPU smoke reproduction plus analysis-only reproduction; publish checksums and an anonymous artifact link; remove or replace the placeholder self-citation. Keep only the source PDF if the venue requires it and ignore `.aux/.log/.out` artifacts.

### 8.6 Revised experiment priority (supersedes Section 7.4 for scientific validity)

1. **Freeze claims and rename the primary estimator** unless a genuine coalition estimator will replace it.
2. **Repair and test benchmark adapters** (BBQ target/polarity; WinoGender direction; conditional completion scoring; StereoSet meaningfulness control).
3. **Create one common, stratified item manifest** and rerun/recompute the common-intersection baseline before spending compute on new rungs.
4. **Fix bootstrap stratification** and rerun all uncertainty analyses; add benchmark/category heterogeneity.
5. **Run same-unit causal attribution (Exp8) and robust ablation controls** before extending the observational ladder.
6. **Only then** triage/complete Exp3/Exp6 GPU jobs and consider new models or trained-`k` controls.
7. Rewrite title/abstract/method/results around the estimator actually computed; run a claim-to-artifact audit and double-blind reproducibility check.

### 8.7 Minimum viable resubmission package

A credible revision should include, at minimum:

- a mathematically correct estimator name and explicit estimand;
- verified native benchmark scoring plus a shared item manifest;
- per-benchmark/category results and corrected cluster-bootstrap CIs;
- primary analysis on comparable units/items, with one prespecified endpoint;
- causal fidelity with robust random/intervention controls and real capability evaluation;
- toned-down fairness and causal language;
- tests and immutable provenance sufficient for independent reproduction.

Until A–F are resolved, the open GPU items in Sections 2–7 are **not the critical path**: they would add precision or breadth to a mis-specified measurement pipeline.
