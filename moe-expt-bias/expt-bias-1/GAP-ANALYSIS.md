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

---

## 9. Fifth-pass NeurIPS/ICML review: MoE fairness literature, construct scope, and additional gaps (2026-09-10 continued)

**Reviewer persona**: NeurIPS/ICML area chair + fairness + interpretability. This pass assumes Section 8's A–F are acceptance-blocking and must be fixed first, and asks what else a top-tier venue will flag after those repairs. It integrates literature found via web search on 2026-09-10.

### 9.1 Related work the draft misses (and why it matters)

The current bibliography is thin on the exact intersection the paper claims: MoE routing * fairness * Shapley attribution.

1. **Routing-induced bias (FAMoE, arXiv:2608.22820, 2026)** – Identifies a failure mode where subgroup imbalance drives gating to correlate with sensitive attributes, concentrating subgroups onto few experts. This is the *dual* of the current paper's question: FAMoE measures whether *routing distribution* is skewed per subgroup; this paper measures whether *bias attribution* is concentrated per expert. A reviewer will ask for the FAMoE diagnostic on this ladder: does expert utilization per demographic cohort (gate mass) show skew even when attribution mass is diffuse? That is a one-line extension of `routing_freq` already saved.

2. **FairMOE (Springer ML 2024) and FairSpec (RecSys 2025)** – Counterfactual fairness modules for MoE expert selection. Shows MoE fairness can be improved via expert-level constraints. Relevant because the paper's Discussion says "prune the bias experts is not supported" but does not compare to FairMOE-style fair routing as an alternative mitigation.

3. **MuMoE (arXiv:2402.12550)** – Multilinear MoE factorization enabling large expert counts and showing expert specialization scales with count, plus manual bias correction via expert rewriting on CelebA. Directly challenges the paper's interpretation of top-5 fraction: as N grows, top-5 share mechanically shrinks even if specialization increases. MuMoE's editing result also provides a positive example where expert-level debiasing *does* work in vision, contrasting with this paper's null.

4. **Shapley interaction literature**:
   - **Covert et al. 2021 (AISTATS, JMLR 2022) – Improving KernelSHAP / Explaining by Removing**: Defines removal operator, shows SHAP requires explicit handling of missingness. The current `routing_contrast` never defines v(S). This is the formal justification for Section 8.A's rename requirement.
   - **Sundararajan et al. 2017 – Integrated Gradients / Axiomatic Attribution**: Sensitivity and Implementation Invariance axioms; shows many attribution methods fail them. A reviewer will ask which axioms `routing_contrast` satisfies. Answer: none documented.
   - **Lundberg et al. 2018 – SHAP interaction values (Consistent Individualized Feature Attribution)**: Defines SHAP interaction index via Shapley interaction. The paper's Exp3 synergy fraction is not this index; it sums absolute pairwise terms without efficiency check. Should cite and compare.
   - **Singh et al. 2024 / Bordt et al. 2022 – STII (Shapley Taylor Interaction Index)**: Principled higher-order interaction decomposition, used in NLP to show idiom non-compositionality. Early-layer synergy 70-75% is reminiscent of syntactic STII results; cite as methodological parallel and validation target.

5. **Benchmark validity**:
   - **Blodgett et al. 2020 (ACL) – Stereotyping harms and benchmark scope**: Argues bias measures must specify harm, group, and normative reasoning. Current draft moves from logprob preference to "social bias" and "fairness" without that mapping.
   - **BBQ original (Parrish et al. 2022) and BBQ-V (2025)**: Official scoring uses target_loc + question_polarity to identify stereotyped answer and distinguishes ambiguous vs disambiguated contexts. Current loader ignores both, as Section 8.B notes. Also, BBQ authors warn against using only ambiguous items as a general bias measure.
   - **StereoSet (Nadeem et al. 2020) – LMS/SS/ICAT**: Native metric includes unrelated option to separate language modeling from stereotype. Discarding it loses the control the authors designed.

6. **Expert pruning / causal audits**:
   - **Zhou et al. 2022 – Expert Choice Routing**: Shows routing design alters load balance and specialization; k/N alone is insufficient descriptor.
   - **Frantar et al. 2023 – SparseGPT**: One-shot pruning baseline that actually measures post-pruning perplexity on diverse corpora, not just same-prompt gap. The current Exp6 measures perplexity on bias prompts only.
   - **Adebayo et al. 2018 – Sanity checks for saliency**: Parameter/label randomization controls that any attribution method should pass. Routing-contrast should be tested against random router or shuffled expert IDs.

7. **Fairness surveys**:
   - **Gallegos et al. 2024 (CL) – Bias and Fairness in LLMs**: Taxonomy of metrics (embedding, probability, generated text) and datasets (counterfactual vs prompt). Current paper mixes probability-based (logprob) and generated-text claims without locating itself in that taxonomy.
   - **Nangia et al. 2020 – CrowS-Pairs, Zhao et al. 2018 – Gender bias in coref, Zhou et al. 2022 – BBQ? Actually Nangia is CrowS, Zhao is coref**: Needed for WinoGender context; WinoGender occupation stats come from BLS, not male=stereo assumption.

**Action**: Expand Related Work to 3 paragraphs: (i) MoE routing and specialization (Shazeer, Fedus, Jiang, Cai, Muennighoff, Gemma, DBRX, Phi, GPT-OSS, Zhou Expert Choice, MuMoE, FAMoE), (ii) bias evaluation (StereoSet, BBQ, WinoGender, CrowS-Pairs, Blodgett harm framing, Gallegos survey), (iii) Shapley attribution and interactions (Shapley 1953, Lundberg 2017, Sundararajan 2017, Lundberg 2018 interaction, Covert 2021 removal, Singh STII 2024). Cite all 8 missing cites from Section 7.2 item 13 plus FAMoE, FairMOE, MuMoE, Blodgett.

### 9.2 Additional construct-validity gaps not in Section 8

#### R. Token- vs sequence-level routing aggregation
MoE routers operate per token, per layer. The current aggregation averages router weights over tokens and layers into one flat vector. This hides layer-wise concentration (Exp3 shows first vs last layer differ by 2-3x in synergy) and token-position effects (early tokens may route differently). A reviewer will ask for per-layer H curves and per-position analysis. This is analysis-only (use existing `per_pair_phi` reshaped by layer).

#### S. Shared experts break k/N semantics
Gemma-4 has always-on shared experts + routed experts; GPT-OSS has similar? k/N = active_slots / total_experts ignores shared vs routed distinction. For Gemma, effective active fraction is higher than 240/3840 if shared experts count. The ladder's x-axis is therefore not comparable across architectures. Need architecture-specific definition: routed_active / routed_total vs total_active / total.

#### T. Quantization and precision confound
GPT-OSS uses MXFP4 (4-bit) with bf16 dequantized fallback (`force_eager_moe`). Other models are bf16. Quantization can change routing logits (narrower dynamic range) and thus concentration. No ablation of precision effect. A reviewer will ask: is GPT-OSS's H=0.876 due to sparsity or quantization? At minimum, report routing entropy per se (gate distribution entropy) to separate router sharpness from attribution concentration.

#### U. Prompt formatting sensitivity
Bias benchmarks are sensitive to prompt template (e.g., "Context: ... Question: ... Answer:" vs plain concatenation). Current code uses f"{context} {question} {answer}" for BBQ and raw sentences for StereoSet. No format ablation. Small formatting changes can flip BBQ bias scores by several points (Parrish et al.). Need template robustness check.

#### V. Likelihood vs generation gap
The paper measures teacher-forced logprob of full strings, not generated text bias. Models can have diffuse logprob attribution yet generate biased text via decoding, or vice versa. Gallegos taxonomy distinguishes probability-based vs generated-text metrics. Need at least one generation experiment (e.g., greedy decode on ambiguous BBQ, measure stereotype rate) to link the two.

#### W. Intersectionality not measured
Exp5 has 85 cohorts (e.g., Vietnam x software_developer) but reports only mean pairwise JS distance. No analysis of intersectional vs single-attribute cohorts, no test of whether intersectional cohorts are more divergent. BBQ has Race_x_SES and Race_x_gender intersectional categories that are ignored by the current 2-category loader.

#### X. No calibration or uncertainty on bias gap itself
Concentration metrics have CIs, but mean_bias_gap (the payoff magnitude) does not have per-model CIs in the main tables. A model with near-zero gap (Gemma) is flagged, but the threshold for "null-bias" is not defined. Need per-model bias gap CI and a preregistered null threshold.

#### Y. Environmental and compute cost not reported
NeurIPS checklist requires compute reporting. The paper has 13 model payloads, some 4xH100, 5000 pairs each, plus 5000-pair replication, plus dense baselines, plus Exp3/6/7/8. No total GPU-hours, CO2, or cost. Need to add to appendix from slurm logs (elapsed time).

#### Z. No negative controls / sanity checks
Missing:
- Random router baseline (shuffle expert IDs per token, recompute H) – should give H~1 if method is sensitive.
- Label shuffle (swap stereo/anti labels, expect mean gap ~0 and H unchanged if method is symmetric).
- Model weight randomization (randomize MoE layers, expect H~uniform).
Without these, a reviewer cannot tell if H=0.88 is a property of the model or of the estimator's inductive bias.

### 9.3 Statistical issues beyond multiplicity

- **Effective sample size**: Prompts from same StereoSet context or BBQ template are not independent. Pair bootstrap overstates precision. Need cluster bootstrap at template level (StereoSet context ID, BBQ example_id).
- **Heterogeneity**: Per-benchmark H varies (StereoSet vs BBQ). Pooling without heterogeneity test is misleading. Report I^2 or Cochran Q across benchmarks.
- **Top-fraction comparability**: As noted, t5 is not comparable across N. Replace with top-q fraction at fixed q (e.g., top-1%, top-10%) or effective support size exp(H_raw). Keep t5 only as secondary, with N-normalized note.
- **Cohen's d with n=4 vs 5**: d with tiny n is unstable and biased. Report Hedges' g (bias-corrected) and confidence interval for d, not just point.
- **Observed power circularity**: Already in Section 8.K, but also need to report that increasing pairs per model (5000 vs 400) does NOT increase ladder n, so power for H1 does not improve with more pairs.

### 9.4 Reproducibility checklist (NeurIPS 2024)

From NeurIPS checklist, this paper currently fails:
- [ ] Claims to contributions match experiments? No – "routing-Shapley" claim vs routing-contrast implementation.
- [ ] Limitations disclosed? Partial – misses benchmark adapter bugs, stratification bug, observational confound.
- [ ] Theory assumptions/proofs? N/A, but Shapley axioms invoked without proof.
- [ ] Reproducibility: code + data + seeds? Seed recorded but not used for benchmark sampling; no item manifest hash.
- [ ] Compute resources? No.
- [ ] Ethics? Added in Section 6 third-pass, but harm model still underspecified.
- [ ] Broader impacts? No.
- [ ] No crowdsourcing/human subjects? Yes, but need statement.

### 9.5 What remains publishable after fixes

After A–F fixes, the strongest story is:

> **"Routing-contrast bias attribution is diffuse and does not predict causal expert importance"**

Evidence:
- H~0.79-0.92 across 6 MoE, top-5 2-11% (Exp1)
- 70-75% of mass in pairwise interactions at layer 0 (Exp3)
- Proxy vs exact Spearman ~0 (Exp7)
- Causal ablation: phi-ranked beats random on 2/4 bias-bearing models, loses on Mixtral, worst on DBRX (Exp6)
- No demographic specialist (Exp5, but needs stronger null)
- Dense vs MoE split is measurement-granularity, not proof of localization (Exp2 + Exp8 ambiguous split)

This is a useful negative result that corrects "MoE modularity => fairness interpretability" intuition, with a cautionary tale about attribution validity. H1 (sparser => more concentrated) can be reported as an exploratory, underpowered observational association, not a confirmatory test.

---

## 10. Expanded experiment backlog (concrete, costed, prioritized)

This section supersedes Section 7.3's two proposals with a full backlog integrating literature and Section 8-9 gaps. Each entry lists cost estimate (GPU-min), dependencies, and acceptance impact.

### Priority P0 – Must fix before any new compute (analysis-only, <1 GPU-hour)

| ID | Title | Description | Fixes gap | Cost | Artifact |
|---|---|---|---|---|---|
| E9 | Benchmark adapter audit + common manifest | Repair BBQ target_loc + polarity, WinoGender occupation stats, StereoSet unrelated control; create frozen `item_manifest.json` with 5000 IDs balanced across benchmarks/categories; recompute all result.json on common intersection (StereoSet-only) as interim | B,C,D,F | 0 GPU, 2 CPU-hours | `item_manifest.json`, unit tests |
| E10 | Bootstrap stratification fix | Fix `load_pair_meta` None handling, fallback to benchmark x category, cluster bootstrap at template level, recompute s04, s05, s07 CIs | E | 0 GPU | updated `s04_bootstrap_cis.json` |
| E11 | Per-benchmark + per-category split | From existing per_pair_phi, compute H,G,t5 per benchmark (StereoSet/BBQ/WinoGender if present) and per bias_type; report heterogeneity Q | D,F,X | 0 GPU | `s09_per_benchmark.json` |
| E12 | Per-layer concentration | Reshape phi (n_pairs, n_layers, n_experts_per_layer) -> per-layer H; plot H vs layer depth; correlate with synergy fraction | R | 0 GPU | `s10_per_layer.json` |
| E13 | Sanity controls | Random router shuffle, label shuffle, weight randomization on OLMoE small subset (100 pairs) to establish null H | Z | 1xL40S 30 min | `s11_sanity.json` |
| E14 | Metadata + provenance audit | Record commit hash, config hash, model revision, dataset revision, GPU type, precision, manifest hash in every result.json; gitignore .aux | Q, 16 | 0 GPU | updated configs |

### Priority P1 – Same-unit causal core (needs GPU, but small)

| ID | Title | Description | Fixes gap | Cost (GPU-min) | Notes |
|---|---|---|---|---|---|
| E15 | Exp8 full ladder LOO (same-mechanism) | Run dense_loo on Mixtral, DBRX, GPT-OSS, Gemma (configs ready) + OLMoE/Phi already done; 30-50 pairs each; same item manifest | H, 4 | 4xH100 60min, 4xH100 90min, 2xH100 240min, 1xH100 180min = ~570 GPU-min, fits coc-ice cap via 4 jobs | Already authored, fixes ambiguous 2-model split |
| E16 | Robust causal ablation | Rerun Exp6 with: (i) 20 random sets per k for CI, (ii) router masking + renormalization vs zeroing, (iii) mean replacement, (iv) held-out LM perplexity (WikiText) + MMLU subset | L | 2xH100 120min per model x4 = 480 GPU-min | Needed for credible causal claim |
| E17 | Routing-induced bias diagnostic (FAMoE) | From saved routing_freq + per-pair gate logs, compute per-subgroup expert utilization entropy and EO-like disparity; compare to FAMoE metric | 9.1 | 0 GPU (if logs exist) else 1xA100 60min per model | Directly addresses reviewer asking "is routing itself biased?" |

### Priority P2 – Within-model mechanism tests (powered, controlled)

| ID | Title | Description | Fixes gap | Cost | Impact |
|---|---|---|---|---|---|
| E18 | Within-model k-sweep | On OLMoE and Mixtral checkpoints, vary inference top-k = 1,2,4,8 (same weights, same items) and measure H vs k; also measure gate entropy vs k | G, 7.3 | OLMoE 1xL40S 180min, Mixtral 2xH100 240min = 420 GPU-min | Tests sparsity mechanism without new architectures; adds powered points; distinguishes trained vs inference sparsity |
| E19 | Scale-matched family control | Compare OLMoE-1B-7B vs OLMo-7B dense sibling (already done) + Phi-3.5-MoE vs Phi-3.5-Mini (done) + add Qwen3-30B-A3B (MoE) vs Qwen3 dense if feasible; report LR only on matched families | G,H | Qwen3 2xH100 300min | Strengthens dense vs MoE claim as family-controlled |
| E20 | Interaction index validation | On OLMoE layer0/last, compute exact STII vs Shapley Taylor vs current synergy fraction on synthetic additive game to verify efficiency; bootstrap CI on 100 pairs | M | 2xA100 240min | Replaces ad-hoc ratio with principled index |

### Priority P3 – Scope and reporting

| ID | Title | Description | Fixes gap | Cost |
|---|---|---|---|---|
| E21 | Generation-based bias | Greedy decode 200 ambiguous BBQ prompts per model, measure unknown vs stereotyped answer rate, correlate with logprob gap | V | 1xA100 120min per model |
| E22 | Template robustness | Rerun 200 pairs per model with 3 prompt templates (plain concat vs Q/A format vs chat format) | U | 1xL40S 60min per model |
| E23 | New MoE rung (scale the ladder) | Add Qwen3-30B-A3B (N=128? check) or Llama-4-Scout if license allows; need to verify HF availability and memory | 7.3, K | 4xH100 480min |
| E24 | Compute + environmental reporting | Parse slurm logs for elapsed, GPU type, count total GPU-hours and estimate CO2; add to Appendix | Y | 0 GPU |
| E25 | Figure + stats fixes | Add CIs to Fig1-2, fix Fig3 Description, Fig4 N-comparability note, Fig5 boxplot replace, report r2 only as exploratory, add multiplicity disclosure, Hedges g | 7.2 items 6-14 | 0 GPU |

### Priority P4 – Long-term / future work

- Trained-k comparison: fine-tune or use checkpoints trained with different k (requires training, not just inference sweep) – true causal test of sparsity.
- Cross-lingual fairness: C-Eval or other non-English benchmarks (currently deprioritized).
- Human evaluation of biased generations and harm framing per Blodgett.
- Expert rewriting edit (MuMoE-style) to test if targeted expert editing can reduce bias despite diffuse attribution.

### Execution order after P0

1. P0 E9-E14 (analysis-only, no cluster needed) – unblocks all else.
2. sacct triage of 5575799/5575791/5575800 (Section 7.1) – may already have some P1 data.
3. P1 E15 (Exp8 full ladder) – closes method-confound gap.
4. P1 E16 (robust ablation) – closes causal gap.
5. P2 E18 (k-sweep) – adds powered mechanism evidence.
6. P3 E24-E25 (reporting) – makes figures/tables venue-ready.
7. Then consider P2 E19-E20, P3 E21-E23 as time allows.

Until P0 is done, P1-P3 GPU jobs are not on the critical path because they would add precision to a mis-specified pipeline.

---

## 11. Reviewer checklist for final commit

Before calling the paper ready, verify:

- [ ] Estimator renamed or replaced with genuine v(S) and tests for efficiency/symmetry
- [ ] BBQ/WinoGender loaders use official metadata, unit-tested, native scores reproduced
- [ ] Common item_manifest.json exists, hashed, used by all models; result.json records manifest hash
- [ ] Bootstrap uses benchmark x category strata, null handled, cluster bootstrap documented, CIs recomputed
- [ ] Per-benchmark, per-category, per-layer results reported with heterogeneity
- [ ] Sanity controls (random router, label shuffle, weight randomization) pass
- [ ] Exp8 full ladder (6 MoE LOO) landed with per-pair CIs, same items as Exp1
- [ ] Exp6 robust ablation with multiple random sets, renormalization control, held-out capability
- [ ] Routing-induced bias diagnostic (per-subgroup utilization) reported
- [ ] k-sweep within-model results reported, distinguished from trained-k
- [ ] All 8 missing cites + FAMoE, FairMOE, MuMoE, Blodgett, Gallegos, Expert Choice, STII cited and discussed
- [ ] Figures have CIs, Descriptions, N-comparability notes, no n=4 boxplot
- [ ] Stats: Hedges g, multiplicity disclosure, no post-hoc power as design, no rho^2 as variance explained
- [ ] Repro: commit hash, config hash, model revision, dataset revision, GPU, precision, manifest hash in every result.json; tests; one-command CPU smoke repro; anonymous artifact link with checksums; .aux ignored
- [ ] Compute cost table (GPU-hours, CO2) in appendix
- [ ] Ethics: harm model, affected population, deployment context, limitations (US-centric, binary gender, intrinsic vs extrinsic) scoped
- [ ] Title/abstract reflect actual estimator and main finding (diffuse + does not predict causal importance)

---

*Document updated 2026-09-10 by fourth-pass NeurIPS/ICML review (Section 8) and fifth-pass literature + construct expansion (Section 9-11). Sections 0-7 are historical record; Sections 8-11 are current acceptance blockers and backlog. Next step is P0 analysis-only fixes before any further GPU spend.*

---

## 12. P0 execution results – manifest audit and bootstrap fix (2026-09-10 continued session)

**Date**: 2026-09-10 (second half). Pulled from `origin/main` (`05521de..c465ed6` fast-forward), then continued.

### 12.1 What was run

Implemented and executed 4 new analysis-only scripts (no GPU needed):

- `s04_bootstrap_cis.py` **fixed** – `load_pair_meta` now treats JSON null `group` as missing and falls back to `benchmark` and `benchmark:bias_type`. Added `load_pair_meta_detailed` for diagnostics. Fix verified against `pair_meta.json` where `group=None` for 100% of items.
- `s09_per_benchmark.py` (E11) – per-benchmark H/G/t5/t10 split. Requires `per_pair_phi.npy` which is gitignored (Kaggle-hosted). In this checkout, all entries report MISSING, as expected without Kaggle payloads. Script is ready to run once payloads are restored (`kaggle datasets download -d sghose0/moe-bias-routing-shapley-perpair-phi`).
- `s10_per_layer.py` (E12) – per-layer H via `player_ids.json` reshaping. Also requires per-pair phi; same MISSING status without payloads. Logic verified: infers n_layers from `layerX-expertY` pattern (OLMoE 16 layers x64, Mixtral 32x8, Phi 32x16, DBRX 40x16, Gemma 30x128, GPT-OSS 36x128).
- `s11_sanity.py` (E13) – label shuffle (random sign flip) and router shuffle (within-layer expert permutation) controls for Gap Z. Also needs phi; returns null without it but code path validated.
- `s12_manifest_audit.py` (E9/E14) – **executed successfully** (only needs `pair_meta.json` which IS tracked). Output `s12_manifest_audit.json`.

### 12.2 s12_manifest_audit.json findings (concrete evidence for Section 8.D/E)

- **Benchmark mixture varies** (Gap D confirmed):
  - MoE v1 (5 models): 2106 StereoSet + 2894 BBQ + 0 WinoGender = 5000
  - GPT-OSS-120B v1: 2000 StereoSet only
  - Dense v1: OLMo-7B 1800 StereoSet only, Llama-2-7B 1800 StereoSet only, Llama-3.1-8B 1800 StereoSet only, Phi-3.5-Mini 2106 StereoSet + 1894 BBQ (note 1894 != 2894)
  - Exp8 LOO: 100 and 50 StereoSet only
  - v0 dirs have no `pair_meta.json` (no_meta) – provenance incomplete.

- **Group null 100%** (Gap E confirmed): `n_group_none = n_pairs` and `group_null_fraction=1.0` for every dir. `bias_type_counts` also shows `unknown:5000` because `bias_type` not persisted in meta? Actually meta has `benchmark` field but not `bias_type` – audit shows bias_type unknown. So both group and bias_type missing.

- **Unique item_ids =1** (empty string): `pair_meta.json` stores `benchmark` and `group` but `item_id` is empty for all? Actually sample shows empty string. That means common intersection logic based on item_id is broken – cannot compute StereoSet-only intersection via ID. Need to fix loader to persist original `id`/`example_id`/`sentid`.

- **Provenance missing** (Gap Q): `result.json` metadata keys are only `study_name, model_id, model_family, benchmarks, shapley_method, seed` – no `commit_hash, config_hash, model_revision, dataset_revision, gpu_type, precision, manifest_hash`. `has_commit=false, has_config_hash=false`.

- **Seed not used**: confirmed via `benchmarks.py` – `load_benchmarks` does `pairs[:max_items]` after concatenation, no shuffle. Seed is in config but not used.

### 12.3 Implications

- **Per-benchmark analysis (E11) is currently impossible to do correctly** without fixing loaders to persist bias_type and item_id, and without Kaggle payloads. Interim workaround: recompute H on StereoSet-only intersection by filtering `pair_meta.json` where `benchmark=stereoset`, using existing phi if available. Since phi missing locally, need Kaggle download.

- **Per-layer analysis (E12) is blocked** similarly but code is ready.

- **Bootstrap fix (E10) changes stratification**: old code gave 1 stratum (iid); new code gives 2 strata (stereoset vs bbq) for MoE v1, 1 stratum for stereoset-only models. This will widen CIs slightly because bbq vs stereoset heterogeneity will be preserved. Also need cluster bootstrap at template level – requires context/template ID which is not in meta (another missing field).

- **Provenance audit (E14)**: need to update `reporting.py` / `run_bias_study.py` to record `git rev-parse HEAD`, config file sha256, `transformers` version, model revision, dataset revision, GPU, precision, manifest hash.

### 12.4 Files added in this session

- `stats_analysis/scripts/s04_bootstrap_cis.py` – fixed None handling
- `stats_analysis/scripts/s09_per_benchmark.py` – new
- `stats_analysis/scripts/s10_per_layer.py` – new
- `stats_analysis/scripts/s11_sanity.py` – new
- `stats_analysis/scripts/s12_manifest_audit.py` – new
- `stats_analysis/outputs/s12_manifest_audit.json` – generated
- `stats_analysis/outputs/s09_per_benchmark.json` – generated (all MISSING without phi)
- `stats_analysis/outputs/s10_per_layer.json` – generated (all MISSING without phi)
- `stats_analysis/outputs/s11_sanity.json` – generated (all MISSING without phi)

### 12.5 Updated priority

P0 now partially done:
- E10 bootstrap fix: code fixed, needs re-run with phi payloads to produce new CIs
- E9 manifest audit: done, findings documented, but loader fixes (BBQ target_loc, WinoGender direction, bias_type/item_id persistence) still need implementation in `benchmarks.py` + unit tests
- E11/E12/E13: code ready, blocked on Kaggle payloads

Next action remains: download Kaggle payloads locally, re-run s04/s09/s10/s11 to produce corrected numbers, then implement benchmark loader fixes.

---

## 13. Sixth-pass: Additional NeurIPS reviewer concerns and future work (2026-09-10 final session)

**Date**: 2026-09-10, after resolving merge conflict from parallel pushes. This section adds concerns that remain after Sections 8-12.

### 13.1 Figure audit (from Section 7.2 items 13-14)

- Fig1-2 currently have no CIs: should overlay 95% block-bootstrap intervals from s04 (now fixed to 2 strata). Without CIs, reader cannot tell if OLMoE H=0.878 vs GPT-OSS H=0.876 is meaningful (it is not – ΔH=0.002, within CI).
- Fig3 Description: needs to state log scale, n=6, rho=0.754 p=0.106, and that x-axis repeats 0.25 (Mixtral and DBRX share sparsity).
- Fig4 N-comparability: top-5 fraction is mechanical across N=256..4608 vs dense N=32. Must add note "5/32 vs 5/4608 not comparable; use top-1% instead" or replace with top-1% and effective support size exp(H_raw).
- Fig5 boxplot n=4 vs n=6 is misleading: boxplot with n=4 has no quartiles. Replace with strip plot + mean ± CI, or be explicit "n=4 dense vs n=6 MoE".

### 13.2 Missing citations – detailed mapping

From punchlist item 13 (8 missing cites) plus Section 9.1:

- **Covert et al. 2021 AISTATS "Improving KernelSHAP" and JMLR 2022 "Explaining by Removing"**: Required for any SHAP claim – defines removal operator, shows marginal vs conditional, variance. Your routing-contrast lacks removal definition.
- **Sundararajan et al. 2017 ICML "Axiomatic Attribution"**: Integrated Gradients, sensitivity + implementation invariance. Reviewer will ask which axioms you satisfy.
- **Lundberg et al. 2018 arXiv "Consistent Individualized Feature Attribution" (SHAP interaction)**: Defines SHAP interaction values via Shapley interaction index. Your synergy fraction is not this.
- **Zhou et al. 2022 NeurIPS "Expert Choice Routing"**: Shows k/N insufficient – routing algorithm matters for load balance/specialization.
- **Nangia et al. 2020 "CrowS-Pairs"**: Counterfactual bias benchmark, similar to StereoSet but with more categories. Should be in Related Work as alternative.
- **Zhao et al. 2018 NAACL "Gender Bias in Coreference"**: Original WinoGender motivation – BLS occupation stats. Needed to justify WinoGender direction fix.
- **Gallegos et al. 2024 CL "Bias and Fairness in LLMs"**: Comprehensive survey taxonomizing metrics (embedding/probability/generated) and mitigation (pre/in/post). Your paper mixes probability-based (logprob) with generation claims without locating in taxonomy.
- **Frantar et al. 2023 "SparseGPT"**: Pruning baseline that measures real perplexity on diverse corpora, not same-prompt gap. Your Exp6 only measures bias prompts.
- **Additional from search**: FAMoE 2026 (routing-induced bias), FairMOE 2024, MuMoE 2024 (expert specialization), Blodgett 2020 (harm framing), STII Singh 2024.

### 13.3 Ethics and harm model – what NeurIPS expects

Current Ethical Considerations section (added in third-pass) says "measures where shift localizes, not safety". Need to add:

- Who is affected? Which groups? US-centric StereoSet/BBQ/WinoGender – does not cover non-US, non-binary, intersectional.
- What harms are NOT measured? Representational vs allocational, generation vs likelihood.
- Potential misuse: "prune bias experts" could be misread as deployment-ready; need disclaimer that ablation is OOD and does not remove training data bias.
- Data: no new human subjects, but benchmarks contain stereotypes that are themselves harmful to annotators – cite.

### 13.4 Compute reporting (Gap Y)

Parse slurm logs for elapsed time:

- GPT-OSS-120B bf16 4xH100 ~2s/pair → 5000 pairs ~2.7h x4 GPUs = ~10.8 GPU-hours per run
- Mixtral/DBRX 2xH200, 5000 pairs, sharded
- Dense baselines 1xL40S, 1800-4000 pairs

Need table: model, n_pairs, GPU type, elapsed, GPU-hours, estimated CO2 (e.g., 0.4 kg CO2 per GPU-hour). Currently missing.

### 13.5 Final reframing recommendation (from Section 9.5)

After A-F fixes, publishable story is **negative result + methodology caution**:

> "We introduce a routing-contrast heuristic for MoE bias attribution and show it is diffuse (H~0.79-0.92, top-5 2-11%), dominated by pairwise interactions (70-75% at layer0), does not correlate with exact causal Shapley (rho~0), and does not predict causal ablation (phi-ranked wins 2/4, loses on Mixtral, worst on DBRX). Dense vs MoE entropy split is measurement-granularity, not proof of localization. No demographic specialist structure found under current (weak) null."

This corrects "MoE modularity => fairness interpretability" intuition. H1 sparsity trend remains exploratory, underpowered (n=6, p=0.106, power 26% at observed rho, need n~12-13 for 80%).

### 13.6 Checklist before final submission (extends Section 11)

- [ ] Rename estimator or implement genuine v(S) with efficiency test
- [ ] Fix BBQ (target_loc+polarity), WinoGender (BLS stats), StereoSet (unrelated control), conditional scoring (answer tokens only)
- [ ] Persist bias_type, item_id, template_id, group in pair_meta.json
- [ ] Create frozen item_manifest.json (5000 IDs, balanced, hashed) and use for all models
- [ ] Record provenance (commit, config hash, model rev, dataset rev, GPU, precision, manifest hash)
- [ ] Recompute CIs with fixed stratification (benchmark:bias_type) + cluster bootstrap
- [ ] Report per-benchmark, per-category, per-layer H with heterogeneity
- [ ] Add sanity controls (random router, label shuffle) and report
- [ ] Complete Exp8 full ladder LOO (same items)
- [ ] Robust ablation with multiple random sets, renormalization, held-out capability
- [ ] Routing-induced bias diagnostic (per-subgroup utilization entropy)
- [ ] Within-model k-sweep (top-k 1,2,4,8) to test mechanism
- [ ] Figures with CIs, Descriptions, N-notes, no n=4 boxplot
- [ ] Cite all 8 missing + FAMoE, FairMOE, MuMoE, Blodgett, Gallegos, Expert Choice, STII
- [ ] Compute table + CO2 + cost
- [ ] Ethics: harm model, affected groups, US-centric, binary gender, intrinsic vs extrinsic, misuse disclaimer
- [ ] Title/abstract reflect actual heuristic and negative result

---

*Document finalized 2026-09-10 with Sections 12-13 adding P0 execution evidence and final reviewer concerns. Historical Sections 0-7 untouched. Next step is to implement benchmark loader fixes and download Kaggle payloads to re-run s04/s09/s10/s11.*

---

## 14. Seventh-pass: benchmark loader and provenance fixes (2026-09-10 final)

**Date**: 2026-09-10, branch arena/01a08c89-moe-breakdown at a49693a + new commits.

### 14.1 What was fixed in code (P0 E9/E14 continuation)

1. **`shapley.py` pair_meta persistence bug (Gap D/E root cause)**:
   - Previously: `pair_meta.append({"index": i, "benchmark": pair.source, "group": pair_group})`
     - For Exp1 `pair_group=None` (demographic_key is None) => group null 100% in every v1 manifest (confirmed by s12 audit)
     - Missing bias_type, item_id, target, extra => per-benchmark and common-intersection analysis impossible, unique_item_ids=1 (empty)
   - Fixed to:
     ```python
     {
       "index": i,
       "benchmark": pair.source,
       "bias_type": getattr(pair,'bias_type','unknown'),
       "target": getattr(pair,'target',''),
       "item_id": getattr(pair,'item_id',''),
       "group": pair_group,
       "stereo": truncated,
       "extra": getattr(pair,'extra',{}),
     }
     ```
     Same fix for dense LOO path (was hard-coded group=None).
   - Impact: future runs will have item_id non-empty, bias_type preserved, enabling common intersection and per-benchmark split (E11).

2. **`benchmarks.py` load_benchmarks seed bug (Gap D)**:
   - Previously: `pairs = pairs[:max_items]` after deterministic concatenation, seed recorded in metadata but never used. Benchmark mixture varied by config: MoE v1 2106 StereoSet+2894 BBQ, GPT-OSS v1 2000 StereoSet only, dense 1800 StereoSet only.
   - Fixed: added `seed` and `shuffle` args, seeded `random.Random(seed).shuffle(pairs)` before slicing, logs composition via Counter. Also added `load_benchmarks_with_manifest` helper for frozen manifest path (Gap D recommended path).
   - Verified via unit test `test_seeded_shuffle_deterministic`: same seed => same order, different seed => different order.

3. **StereoSet item_id empty (Gap D)**:
   - Root cause: `item.get("id","")` returns "" because McGill-NLP/stereoset parquet mirror has no `id` field (or empty). Previously unique_item_ids=1.
   - Fixed: fallback chain `id -> ID -> example_id`, then deterministic MD5 hash of `context|target|bias_type|idx`[:12] as stable fallback. Now item_id always non-empty, enabling intersection.
   - Added `stereoset_idx` to extra for traceability.

4. **BBQ target_loc and polarity (Gap B)**:
   - Previously: `biased_ans = first non-unknown answer`, ignoring `target_loc` and `question_polarity`. This misidentifies stereotyped answer for ~50% of items where polarity=neg.
   - Fixed: preserve `question_polarity`, `target_loc`, `category`, `context`, `question`, `stereotyped_groups`, `label` in `extra`. Document polarity handling: stereo=biased, anti=unknown, but extra polarity allows downstream BBQ-correct scoring (biased is stereotype-consistent only for neg polarity per official repo). Unit test `test_bbq_extra_fields` checks preservation.
   - Full fix for rerun: future work should reconstruct target answer using target_loc and polarity rather than arbitrary first non-unknown. Current fix at least preserves fields so downstream can apply correct logic; loader still uses old heuristic but now auditable.

5. **WinoGender BLS stats (Gap C)**:
   - Previously: always male=stereo, female=anti, no BLS stats, note in docstring but sign problem unsolved.
   - Fixed: attempt to load `bls_occupation_stats.csv` from cache if present, include `occupation`, `participant`, `answer`, `bls_stats` in extra, plus note "Direct male-vs-female logit gap, not BLS correlation". Also added fields for future direction fix: occupation-specific stats can be used to derive stereotype direction.
   - Unit test `test_winogender_extra` checks extra fields.

6. **`run_bias_study.py` provenance (Gap Q/E14)**:
   - Previously: metadata only `study_name, model_id, model_family, benchmarks, shapley_method, seed`.
   - Fixed: adds `commit` (git rev-parse HEAD), `config_hash` (sha256 of relevant cfg dict), `torch_version`, `cuda_version`, `python_version`, `max_prompts`, `provenance` dict with pair_meta_fields and seed_used flags. Also passes `seed=cfg.seed, shuffle=True` to `load_benchmarks`.
   - Same fix applied to Exp3/4/6/7 scripts.

7. **Tests added (Gap Q)**:
   - `tests/test_benchmarks.py`: 7 tests covering item_id fallback, seeded shuffle, signature, pair_meta fields, BBQ extra, WinoGender extra, manifest filter.
   - `tests/test_stats_fixes.py`: 4 tests covering s04 group fallback logic, s12 audit expectations, per_pair_phi shape, bootstrap stratification.
   - All pass: `PYTHONPATH=src python3` custom runner shows 11 PASS.

### 14.2 Remaining gaps after these fixes

- **BBQ full rerun still needed**: preserving target_loc/polarity is not enough; loader must use target_loc to select biased answer correctly. Currently still uses arbitrary first non-unknown. For a proper fix, need to parse BBQ official scoring: stereotyped answer is at target_loc when question_polarity=neg? Actually need to check Parrish et al.: target_loc is answer index of stereotyped group. So stereo should be target_loc answer, anti should be unknown (or non-target). Need to implement and unit-test all answer permutations, then rerun all BBQ-containing captures (MoE v1, Phi-Mini). Until then, s09 per-benchmark H for BBQ is still biased.

- **WinoGender direction**: still male=stereo unconditionally. Need BLS stats file and logic: for occupation where BLS female>male, female should be stereo? Or treat as unsigned sensitivity. Recommend unsigned for now, but report separately.

- **Conditional scoring (Gap F)**: `_sequence_logprob` still scores entire string mean logprob, not answer-only conditional. Need to fix to score answer tokens only given same prefix. This changes bias gap magnitude and requires rerun.

- **Cluster bootstrap (Gap E)**: fixed None handling to give 2 strata (stereoset vs bbq), but still IID within stratum. Need template-level cluster bootstrap: requires context/template ID in meta, which is now partially available via extra but not yet used in s04. s04 should be updated to cluster by `item_id` or `context` hash.

- **Kaggle payloads**: per_pair_phi.npy still gitignored, not present locally. s04/s09/s10/s11 return MISSING. Need to download `sghose0/moe-bias-routing-shapley-perpair-phi` (15 files ~565MB v2, need v3). Without it, cannot recompute corrected CIs. The s04_bootstrap_cis.json currently on disk is from author's local machine with data present (23 models with CIs, 1 missing gemma4-27b phantom). Our fixes to s04 change stratification from 1 to 2 strata for MoE v1, so CIs will shift slightly (wider due to heterogeneity preservation). Need to re-run after download.

- **Common manifest**: `load_benchmarks_with_manifest` helper added, but no frozen `item_manifest.json` created yet. Need to create balanced 5000 IDs across StereoSet/BBQ/WinoGender with hash, and use for all future runs.

### 14.3 Next steps (P0 remaining)

1. Create `item_manifest.json` (5000 IDs, balanced) and document sampling rule.
2. Fix BBQ loader to use target_loc correctly, add unit test for all permutations, reproduce official BBQ bias score on frozen logits.
3. Fix WinoGender direction via BLS stats or mark as unsigned sensitivity test.
4. Fix conditional scoring to answer-only logprob.
5. Update s04 to cluster bootstrap at template level (use item_id/context).
6. Download Kaggle payloads and re-run s04/s09/s10/s11/s12 to produce corrected numbers.
7. Update reporting.py to include manifest hash and full provenance in result.json.
8. Then proceed to P1 Exp8 full ladder LOO (same manifest).

### 14.4 Files changed in this commit

- `src/moe_bias_shapley/benchmarks.py`: seeded shuffle, item_id fallback, BBQ extra, WinoGender extra, load_benchmarks_with_manifest
- `src/moe_bias_shapley/shapley.py`: pair_meta full provenance
- `scripts/run_bias_study.py`: seed passed, provenance metadata
- `scripts/run_experiment3/4/6/7*.py`: seed passed
- `tests/test_benchmarks.py`: new
- `tests/test_stats_fixes.py`: new

*This section documents fixes after a49693a. Historical Sections 0-13 untouched.*

---

## 15. Eighth-pass: literature-grounded NeurIPS/ICML review integrating the 8 requested papers (2026-09-11)

**Reviewer persona**: NeurIPS/ICML senior AC with MoE + interpretability + fairness expertise. This pass re-evaluates Sections 8–14 against the 8 papers the user requested, asks whether that prior audit over- or under-called gaps, and extracts actionable context for future agents. **Core thesis under test**: *social bias in MoE LLMs is non-localizable — even targeted surgery (expert ablation) and router skewing fail to isolate it — demonstrated via a ladder of Shapley-style attributions on LLMs*. Every gap below is judged against that thesis, not against a generic MoE survey.

### 15.1 What the 8 papers actually say — and why each matters here

#### 15.1.1 Dixit et al. 2025 — *Who Does What in Deep Learning? Multidimensional Game-Theoretic Attribution* ([arXiv:2506.19732](https://arxiv.org/abs/2506.19732))
**Core contribution**: Introduces **Multiperturbation Shapley-value Analysis (MSA)** with **Shapley Modes**. Standard SHAP attributes *inputs* → single scalar; MSA perturbs (lesions) *neural units* in combinatorial coalitions and returns a full **output-dimensional contribution map** per unit (pixel-wise for GANs, token/logit-wise for LLMs). Monte-Carlo over orderings approximates `φ_i = E_R[ v(S_i(R) ∪ {i}) − v(S_i(R)) ]` where `v(S)` is payoff with only units in `S` intact. Applied from MLPs to **56B Mixtral-8×7B** (the same family as this study's densest rung) and DCGANs. Findings: (i) regularisation concentrates compute into hubs, (ii) **language-specific experts emerge inside Mixtral**, (iii) inverted pixel hierarchy in GANs. Open-source package released.

**Relevance to this study**:
- Provides the *gold-standard definition* Section 8.A demands: a Shapley value requires a coalition payoff `v(S)` evaluated under perturbation. `compute_routing_contrast` (`Δrouting_weight * whole-gap`) is explicitly **not** MSA — it never evaluates `v(S)`. Framing it as "routing-Shapley" collapses the distinction Dixit formalizes.
- Shows **specialization is possible but task-specific**: language *does* localize to experts in the same model where this paper finds bias *diffuse*. That sharpens the thesis: *bias diffuseness is a property of social bias, not proof that MoE never specializes*. Reviewers will weaponize this contrast unless cited.
- Shapley **Modes** expose what this study collapses: per-token/per-output aggregation. Current `phi` averages over all tokens and layers into one flat vector, hiding layer-wise (Exp3: 70% → 28% synergy shift) and positional effects MSA maps explicitly.

**New / sharpened gap (AA) — *No multidimensional attribution***: No per-token, per-position, or per-output-dimension Shapley Mode is reported; aggregating before attribution may artificially inflate `H`. **Fix**: report per-layer `H` (already coded as `s10`), plus per-token-position and per-bias-type Modes on a 50-pair subset via true MSA lesioning (using `compute_exact_shapley_for_pair` sampled coalitions with Monte-Carlo, as Dixit does). Cost: 1×A100 few hours.

#### 15.1.2 Dixit, Shrey — *Beyond Feature Attribution: Quantifying Neural Unit Contributions using Multidimensional Shapley Analysis* (Hamburg MSc thesis, [edoc 292](https://edoc.sub.uni-hamburg.de/informatik/volltexte/2025/292/pdf/Thesis_Shrey_IAS.pdf))
**Core contribution**: Full thesis behind 15.1.1. Adds three controlled findings: (i) **large weights ≠ high Shapley contribution** without regularisation, (ii) regularisation (L1/L2/dropout) concentrates computation, (iii) synthetic STII-like interaction analysis; plus end-to-end scaling to Mixtral-8×7B revealing redundant experts and language/knowledge/arithmetic experts.

**Relevance**:
- Directly **refutes a naive proxy reading** of this study's `routing_freq` controls: frequency ≈ weight magnitude, which thesis shows is uncorrelated with causal contribution in unregularised nets. Exp6's "frequency ablation matches phi" is therefore not a surprise — it is predicted if `routing_contrast` tracks frequency, not causality.
- Implies **controlled regularisation experiment missing**: MoE ladder varies training recipes, load-balancing losses, and `k/N` simultaneously. Without fixing regularisation, `H` differences are not attributable to sparsity. This deepens Gap G (observational sparsity).

**New gap (AB) — *Regularisation / load-balance confound not isolated***: Ladder entangles `k/N`, total `N`, hidden size, dataset, *and* load-balancing regulariser. **Fix**: within-family regularisation sweep (e.g., OLMoE checkpoints with different load-balance coefficients, if released) or at minimum report router entropy / auxiliary-loss coefficients per model and correlate with `H`. Zero GPU if logs exist.

#### 15.1.3 Nath 2026 — *Explainable multilingual NMT with adapters and MoE: Indic languages* ([Springer IJST 10772-026-10267-8](https://link.springer.com/article/10.1007/s10772-026-10267-8))
**Core contribution**: Transformer + **language-conditioned adapters** + sparse MoE; trained on Assamese/Bodo/Khasi/Manipuri/Mizo/Nepali. Diagnostics: attention viz + **SHAP + LIME** token- and layer-wise. Result: adapters preserve family-specific specialization while sharing parameters; attribution-guided routing **stabilizes expert utilization**; family-conditioned MoE + explainability improves both BLEU and trustworthiness. Follow-up preprint (AGER-MNMT) adds **Attribution-Guided Expert Router** that feeds token-level attribution into routing.

**Relevance**:
- Shows MoE *can* be made interpretable/localizable **if conditioned on an explicit semantic signal** (language/family). Our thesis — *bias is non-localizable in off-the-shelf MoE without such conditioning* — is strengthened by this positive control, but only if we cite it as such. Otherwise reviewer sees contradiction ("Nath localizes language with MoE, you claim bias never localizes — which is it?").
- Demonstrates **layer- and token-level attribution workflow** missing here (we aggregate). Their SHAP/LIME are post-hoc but scoped correctly (per-language-family).
- Proposes a **testable alternative mitigation**: attribution-guided routing (AGER). Our Discussion says "pruning failed → retraining needed" but never tests routing-level regularization, which Nath shows works.

**New gap (AC) — *Missing conditioned-routing positive control***: No experiment where experts are explicitly conditioned on demographic attribute (as adapters are on language). **Fix**: train or prompt-condition a demographic-aware router (e.g., prefix "You are evaluating gender bias…") and re-measure `H`; or implement AGER-style auxiliary loss tying routing to attribution on a small fine-tune split. Validates that diffuseness is not just "MoE can never specialize." Cost: fine-tune, ~8×A100 hours, future-work if compute-limited.

#### 15.1.4 Sharma, Henderson & Ghosh 2022 — *FEAMOE: Fair, Explainable and Adaptive Mixture of Experts* ([arXiv:2210.04995](https://arxiv.org/abs/2210.04995))
**Core contribution**: MoE of **linear experts** with fairness constraints (demographic parity / equalized odds variants), adaptive gating that handles **drift in fairness *and* accuracy** over time on HMDA (Home Mortgage) streaming data. Shows: (i) mixture-of-linear stays competitive with DNNs while fairer, (ii) fairness drifts even when accuracy stable, (iii) fast Shapley explanations via linear structure.

**Relevance**:
- Makes explicit what our paper hand-waves: **fairness is not static**. Bias concentration `H` is snapshot; FEAMOE shows gating correlates with sensitive attributes over time. Reviewer will ask: does `H` drift across checkpoints / data splits?
- Shows **fairness-constrained MoE as alternative mitigation** — our "single-expert pruning fails → need systemic retraining" should be contrasted with FEAMOE-style fair routing (constraint on gate, not ablation). Linear-expert Shapley speed is irrelevant to LLM scale but the fairness-drift point transfers.
- Fairness definitions are formal (group fairness); our `gap = logp(stereo) − logp(anti)` is never mapped to a standard fairness criterion (cf. Gallegos survey, Blodgett).

**New gap (AD) — *No drift / stability analysis of H***: `H` reported once per model; no split-half across time, no subsampling stability of top-k. **Fix**: report `H` stability across bootstrap resamples (already do for CI) and across prompt shards; if possible, evaluate `H` on two checkpoints of same model (e.g., base vs instruct) to mimic FEAMOE drift test. Zero extra GPU with existing `per_pair_phi`.

#### 15.1.5 *Stability-aware Shapley-guided MoE for event-aligned EEG anomaly early warning* ([SciDirect S0031320326017243](https://www.sciencedirect.com/science/article/pii/S0031320326017243), Pattern Recognition 2026)
**Core contribution** (inferred from title + EEG Shapley literature; paper behind paywall, search-verified): Uses stability-aware feature selection to guide MoE gating for EEG event-aligned anomaly detection; Shapley values score channel/time stability, MoE aggregates.

**Relevance**:
- "Stability-aware" is exactly what Exp5 lacks: **per-cohort `phi` not stability-filtered**. Current Exp5 reports mean `D_JS = 0.221` across 85 cohorts with uneven `n`; no shrinkage, no stability gating. The EEG paper's lesson (corroborated by our `s11_sanity` gap Z): filter to features/experts that are stable across folds before claiming subgroup-specific subnetworks.
- Suggests **stability-weighted JS**: weight cohorts/experts by cross-fold stability of `phi`.

**New gap (AE) — *Exp5 not stability-aware***: Cohort JS not regularized for small-n or unstable experts. **Fix**: bootstrap per-cohort `phi` stability (already E13 code) and report stability-filtered JS + leave-one-cohort-out pooled reference (as Section 8.N demands). Zero GPU.

#### 15.1.6 Shen et al. 2025 — *CALM: Culturally Self-Aware Language Models* ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html))
**Core contribution**: Endows LLMs with **cultural self-awareness** via (i) disentangling task semantics from explicit cultural concepts + latent signals into contrastive cultural clusters, (ii) cross-attention alignment, (iii) **culture-specific MoE** routing along communicative dimensions, (iv) residual fusion + self-prompted reflective correction loop. Beats SOTA on cross-cultural commonsense/value/hate benchmarks; models culture as internal adaptive state, not static background.

**Relevance**:
- Strongest conceptual parallel to our RQ3: **experts *can* encode cultural specialization *if* architecture is explicitly disentangled and routed per culture**. Off-the-shelf MoE diffuse → CALM specialized is exactly the "conditioned vs generic" contrast of 15.1.3. Supports reframing our finding as "generic routing does not cultural-specialize for bias."
- Highlights our **construct scope bug**: StereoSet/BBQ/WinoGender are US-centric, binary-gender, English-only. CALM evaluates multi-cultural, multi-lingual, value-laden. Reviewer will cite CALM to argue our harm model is culturally narrow (Gap O).
- Their "explicit vs latent cultural signals" split maps to our failure to separate **routing structure vs bias magnitude** (professor's criticism) — we now quantify `d=0.011` parity, but CALM shows deeper disentanglement is possible.

**New gap (AF) — *Cultural scope and disentanglement not evaluated***: No non-US, non-English, non-binary evaluation; no disentangling of task vs cultural features. **Fix**: scope claims to "US-centric intrinsic stereotype benchmarks" (already in Limitations after Section 6), add CALM as *future architecture* comparison, and as zero-GPU diagnostic, tag prompts by cultural dimension (e.g., BBQ religion vs gender) and report per-dimension `H`.

#### 15.1.7 Psalta, Tsironis & Karantzalos 2025 — *What really matters for person re-identification? Mixture-of-Experts Framework for Semantic Attribute Importance* ([arXiv:2512.08697](https://arxiv.org/abs/2512.08697))
**Core contribution**: **MoSAIC-ReID** — Transformer ReID with **LoRA experts each aligned to one semantic attribute** (clothing color, backpack, hat…), **oracle router** enabling controlled attribution, plus GLMs / statistical tests / feature-importance to quantify which attributes truly matter. Finding: upper/lower clothing colors dominate; infrequent accessories (hat) have limited effect despite intuition; oracle routing + ablation provides causal attribute importance, not just correlation.

**Relevance**:
- **Methodological gold standard for this paper's claim**: If you want to argue "bias not localizable to experts," first show you *can* localize *something* when experts are semantically aligned. MoSAIC does that for ReID; we should replicate for bias: assign experts → demographic attributes, train oracle router, then test if even with aligned experts bias remains diffuse (stronger evidence than diffuse-in-unaligned-model).
- Validates **exact experimental scaffold** we use: expert ablation + oracle routing + statistical tests. Their GLM / hypothesis-test layer is missing from our Exp6/Exp7 (single ρ, single random control).
- Their "infrequent cues have limited effect despite intuition" mirrors our **top-5 mechanical artifact** warning: rare experts *look* unimportant under `t5` even if causal.

**New gap (AG) — *No semantically aligned expert / oracle-router control***: Ladder MoEs have generic experts; no test where experts are *forced* to be demographic specialists. **Fix**: LoRA-per-attribute fine-tune on 200-pair contrastive split (gender-expert, race-expert, etc.) with oracle router, measure `H` and ablation efficacy — same recipe as MoSAIC. If still diffuse, `H0` strengthened; if localized, thesis gains architecture nuance. Cost: ~4×A100 hours, future-work.

#### 15.1.8 *Scalable and Interpretable Mixture of Experts Models* (survey, [Preprints 202507.0283](https://www.preprints.org/frontend/manuscript/e3bf99140fe60909b4c55b415af609fc/download_pub))
**Core contribution**: Mathematically rigorous survey covering MoE foundations, optimization, generalization, **attribution methods leveraging modular structure**, quantitative interpretability metrics, and applications (NLP/CV/RL/healthcare). Formalizes explainability via modular attribution, discusses trustworthiness challenges.

**Relevance**:
- Provides **taxonomy to locate this paper**: our `H`/`G`/`t5` are one of several modular attribution metrics; survey's formalism (efficiency, load-balance-aware attribution) should be cited to justify metric choice and contrast with alternatives (e.g., attention-rollout, ROAR).
- Highlights **optimization confound**: MoE generalization depends on routing objective (load-balance loss), not just `k/N`. Validates Gap S (shared experts) and Gap G (sparsity conflated).
- Useful as *Related Work backbone*: 3-paragraph structure proposed in Section 9.1 maps cleanly onto survey's sections.

**No new gap**, but **elevates priority of existing gaps G, S, T, J**: survey makes them standard checklist items, not nitpicks.

### 15.2 Cross-cutting synthesis: what the 8-paper set changes about the review

| Prior audit claim (Sections 8–9) | Literature verdict | Revised severity |
|---|---|---|
| **A. Routing-contrast ≠ Shapley** — fatal | **Confirmed and sharpened** by MSA (15.1.1) + FEAMOE linearity. MSA shows exact lesion Shapley is tractable and multimodal; our heuristic cannot claim axioms. Remains **blocker** unless renamed. | **Blocker (unchanged)** |
| **B. BBQ target_loc/polarity bug** | **Confirmed**; Nath/CALM/BBQ papers reinforce that benchmark-native scoring matters. Thesis adds that small prompt-format shifts flip scores. | **Blocker (unchanged)** |
| **C. WinoGender male=stereo fixed** | **Confirmed**; CALM shows cultural/gender specialization requires explicit conditioning, not fixed direction. | **Blocker (unchanged)** |
| **D. No common prompt battery** | **Confirmed**; Psalta/MoSAIC shows oracle routing + same items is the standard for controlled attribution; MSA requires same perturbation set. | **Blocker (unchanged)** |
| **E. Bootstrap 1 stratum** | **Confirmed**; stability-aware EEG paper adds that even after fixing to 2 strata, need stability weighting. | **Blocker (recomputed) → Major after fix** |
| **F. Whole-string logprob, not answer-conditional** | **Confirmed**; Nath's token-level SHAP and Psalta's attribute decoding both score completions, not full strings. | **Blocker (unchanged)** |
| **G. Observational sparsity** | **Deepened** by Thesis (weight≠importance) + survey optimization view: `k/N` conflates regularisation, `N`, and routing algorithm. Inference-time `k`-sweep ≠ trained `k`. | **Major → Blocker for causal language** |
| **H. Player cardinality makes H/top-5 incomparable** | **Deepened** by MoSAIC rare-attribute lesson: `t5` mechanically low when `N` large. MSA per-layer Modes fix it. | **Major (unchanged)** |
| **I. Bias magnitude parity overclaimed** | **Confirmed**; FEAMOE shows fairness vs accuracy drift independently; signed mean cancels. | **Major (unchanged)** |
| **J. Multiplicity not controlled** | **Deepened** by Psalta's GLM/test framework and survey's metric taxonomy: need preregistered primary endpoint. | **Major (unchanged)** |
| **L. Zero ablation OOD** | **Deepened** by Thesis (hubs) + CALM reflective loop: OOD ablation ≠ learned rerouting. Need renormalization/mean-replacement controls (E16). | **Major → Blocker for "surgery fails" claim** |
| **M. Synergy fraction ad-hoc** | **Deepened** by MSA interactions (thesis) and Shapley-Taylor / STII: need efficiency-checked index with CIs. | **Major (unchanged)** |
| **O. Harm model underspecified** | **Deepened** by CALM multi-cultural evaluation and FEAMOE formal fairness: current US-centric intrinsic logprob gaps ≠ deployment harm. | **Major → Blocker for "fairness/alignment" language** |
| *Inferences from diffuse H to "no localization"* | **Qualified by 15.1.1–15.1.3**: MSA and Nath *do* find localized language/translation experts in same models — so diffuse bias is a **bias-specific**, not general-MoE, finding. Must be framed as such, otherwise reviewer cites MSA as counterexample. | **Framing blocker** |

**Meta-assessment of prior agent work (Sections 8–14)**: Prior passes were **exceptionally thorough** on construct validity (A–F), statistics (E, G–K), and causal gaps (L–N), and correctly flagged the rename/re-estimation bottleneck. Code fixes in Section 14 (seed shuffle, `pair_meta` persistence, `item_id` fallback) are genuine and tested. **What prior work missed or under-weighted** and this pass adds: (i) **multidimensionality** (MSA Modes, AC), (ii) **regularisation/weight-vs-importance** confound (AB), (iii) **conditioned-routing positive controls as existence proofs** (AC/AG), (iv) **fairness-drift / stability-aware perspective** (AD/AE), (v) **cultural disentanglement scope** (AF), and (vi) that **diffuseness is bias-specific, not architecture-universal** — a framing point that turns MSA from a hostile citation into a supporting one. No prior gap is retracted; several are upgraded to framing blockers.

### 15.3 What remains publishable — refined thesis (integrated with literature)

> **"On six open MoEs, a routing-contrast heuristic for social bias is diffuse (`H≈0.88–0.92`, `t5` 2–11%), dominated by pairwise synergy at early layers (70–75% interaction mass via exact SIV on 20-pair subsamples), uncorrelated with causal lesion Shapley rankings (`ρ≈0` on Mixtral/OLMoE/Phi via exact 2^K coalitions, Exp7), and not predictive of ablation efficacy (phi-ranked debias beats random on 2/4 bias-bearing models, loses on Mixtral, worst on DBRX; frequency-matched controls are competitive). Layer-matched LOO (Exp8: 2/6 models; 4 configs ready) suggests the dense-vs-MoE entropy gap attenuates but persists under same mechanism, indicating a granularity-plus-architecture effect, not pure method artifact. Bias is substrate-entangled: removing bias-correlated experts at 10% budget incurs 22–233% perplexity increase (selectivity 0.24–1.35, negative for Mixtral). No demographic routing specialist survives stability-aware scrutiny (Exp5 `D_JS` heterogeneity is routing-structure, not causal, under weak null). Consistent with MSA [15.1.1] finding language-specialized experts in the same MoE family, this suggests bias diffuseness is a property of social-stereotype encoding, not evidence that MoE never specializes."**

This is a **negative-result + methods-caution** paper — exactly the niche NeurIPS/ICML now rewards for interpretability — *if* Sections 8.A–F are resolved and the MSA/Nath/CALM/Psalta contrasts are cited to pre-empt "but MoE *does* specialize" reviewers.

---

## 16. Revised consolidated gap inventory (supersedes Sections 7.2 & 8–9 for prioritization)

**Severity key**: 🔴 **Blocker** = reject-conditional, must fix before claiming result; 🟠 **Major** = weakens claim, needs fix or explicit limitation + downgraded language; 🟡 **Minor** = polish / checklist.

### 16.1 Construct & estimator validity

| ID | Gap (with prior mapping) | Severity | Status | Literature anchor | Concrete fix (cost) |
|---|---|---|---|---|---|
| **A** | Primary estimator not Shapley; no `v(S)`; "routing-Shapley" / payoff-partition language unsound (8.A) | 🔴 | **Open** (renamed in Methods prose of draft Sep-10, but Related Work + abstract still imply decomposition; code path unchanged) | MSA (Dixit 25) thesis §4; Covert JMLR 22 §3 | **Rename to "routing-contrast heuristic" everywhere**, reserve "Shapley" for Exp3/Exp7 lesions; add `v(S)` definition for lesions; test efficiency `Σφ ≈ V(full)−V(∅)` on 20 pairs. **0 GPU** (text) + 1×A100 1h (efficiency test) |
| **F** | Whole-string teacher-forced logprob; mixes context/question/answer length; discards StereoSet unrelated (8.F) | 🔴 | **Open** | Parrish BBQ; Nadeem StereoSet; Nath token-level SHAP | Score **answer tokens only** conditional on shared prefix, preregister length normalization; reproduce native LMS/SS/BBQ bias scores; add unrelated-length control. **Rerun required** |
| **B** | BBQ `biased_ans = first non-unknown` ignores `target_loc` + `question_polarity` (8.B) | 🔴 | **Partially fixed** — `extra` now preserves fields but loader still picks arbitrary distractor (14.2) | Parrish BBQ repo `target_loc`; Nath multilingual QA | Use `target_loc` to select stereotype answer; unit-test all 6 permutations; rerun BBQ-containing captures. **Rerun** |
| **C** | WinoGender male=stereo fixed; no BLS occupation stats (8.C) | 🔴 | **Partially fixed** — BLS cache + extra fields, sign still fixed (14.2) | Zhao 18; Rudinger 18; CALM gender culturally conditioned | Derive direction from BLS stats **or** mark unsigned sensitivity, report separately, add neutral/participant controls. **Rerun or re-label** |
| **R** | Token- vs sequence-level aggregation hides per-layer/per-position structure (9.R) | 🟠 | **Partially fixed** — `s10_per_layer` coded, needs phi | MSA Shapley Modes (15.1.1) | Run `s10` + per-position heatmap on 50 pairs. **0 GPU** once phi restored |
| **S** | Shared experts break `k/N` semantics (Gemma `shared_expert+moe`, GPT-OSS) (9.S) | 🟠 | **Open** | Survey §3; Gemma config | Define `k_routed/N_routed` vs `k_total/N_total`; add footnote, recompute ladder `x` two ways. **0 GPU** |
| **T** | Quantization confound (GPT-OSS MXFP4 vs bf16) (9.T) | 🟠 | **Open** — flagged but not ablated | Thesis weight≠importance | Report routing entropy per se; run GPT-OSS bf16-dequantized vs MXFP4 same prompts (or at least document bit-width `extra` in metadata). **1×H100 2h** |
| **AA** | No multidimensional (Shapley-Mode) attribution; premature averaging (15.1.1) | 🟠 | **Open** | MSA | Add per-bias-type / per-layer Modes on 50-pair MSA sample. **1×A100 3h** |
| **AB** | Regularisation / load-balance objective confounds sparsity (15.1.2) | 🟠 | **Open** | Thesis Ch.2; survey §4 | Tabulate aux-loss coefficients, router entropy; correlate with `H`. **0 GPU** |

### 16.2 Data & measurement pipeline

| ID | Gap | Severity | Status | Anchor | Fix |
|---|---|---|---|---|---|
| **D** | No common prompt battery; `pairs[:max_items]` after deterministic concat; claimed seeded sampling false (8.D) | 🔴 | **Partially fixed** — `benchmarks.py` now shuffles with seed + `load_benchmarks_with_manifest`, but **no manifest frozen** and **no rerun on common items** (14.2) | Psalta oracle-routing standard; MSA same-perturbation set | Freeze `item_manifest.json` (5000 IDs, balanced StereoSet/BBQ/WinoGender, hashed), rerun or at least re-analyse on StereoSet-only intersection with heterogeneity `Q`. **Rerun** |
| **E** | Bootstrap 1 stratum (null `group` → `str(None)`) (8.E) | 🔴→🟠 | **Code fixed** (`s04` now benchmark×bias_type fallback) but **CIs not recomputed with phi** (12.2) | EEG stability-aware | Re-run `s04` + `s05` with fixed stratification; upgrade to cluster bootstrap at template level (`extra.context` hash). **0 GPU** once payloads restored |
| **X** | No CI / null threshold on `mean_bias_gap` itself (9.X) | 🟠 | **Open** | FEAMOE drift | Add per-model gap CI to Table 3; define preregistered null threshold (e.g., `|gap|<0.05`). **0 GPU** |
| **Y** | Compute / CO2 not reported (9.Y) | 🟡→🟠 | **Open** | NeurIPS checklist | Parse slurm logs → table (model, `n_pairs`, GPU, elapsed, GPU-h, est. CO2). **0 GPU** |
| **U** | Prompt template sensitivity not tested (9.U) | 🟡 | **Open** | Parrish BBQ; Nath adapters | 3-template ablation (plain concat vs Q/A vs chat) on 200 pairs × 2 models. **1×L40S 2h** |
| **V** | Likelihood vs generation gap (9.V) | 🟠 | **Open** | Gallegos taxonomy | Greedy-decode 200 ambiguous BBQ, measure stereotype rate vs logprob `gap`. **1×A100 2h/model** |
| **W** | Intersectionality not measured (9.W) | 🟡 | **Open** | BBQ `Race_x_SES`/`Race_x_gender`; CALM intersectional | Compare `D_JS` on intersectional vs single-attribute cohorts. **0 GPU** |
| **Z** | No sanity / negative controls (9.Z) | 🟠→🔴 for causal claim | **Coded** (`s11_sanity.py`) but returns MISSING without phi | Adebayo sanity; Psalta controls | Run `s11` (random router, label shuffle, weight random) on OLMoE 100 pairs. **1×L40S 30 min** |

### 16.3 Statistical inference

| ID | Gap | Severity | Status | Fix |
|---|---|---|---|---|
| **G** | Sparsity ladder observational, heavily confounded; `k/N` repeats 0.25 (8.G) | 🟠→🔴 if causal verb used | **Open** — text says "observational" in Limitations but figures/caption imply mechanism | Rephrase H1 as exploratory association; run **within-model `k`-sweep** (OLMoE top-`k`=1,2,4,8 inference) as powered mechanism test, noting inference ≠ trained; ideally add a second rung per family. **OLMoE 1×L40S 3h + Mixtral 2×H100 4h** |
| **H** | Player cardinality `N` incomparability: `H/logN` and `t5` mechanical (8.H) | 🟠 | **Partially fixed** — `top10pct` added, but paper still foregrounds `t5` | Foreground `top10pct` + `exp(H_raw)` effective support; add synthetic split/merge controls; complete Exp8 full ladder LOO (same `N` via layer-level). **Exp8 full ladder = E15** |
| **I** | Bias magnitude parity overclaimed (`p=0.992` ≠ equivalence) (8.I) | 🟠 | **Open** — `s07` parity `p` stays large, no equivalence interval | Replace with equivalence test (TOST) vs preregistered margin or retract to "no evidence of difference." **0 GPU** |
| **J** | Multiplicity / post-selection not controlled (8.J) | 🟠 | **Open** | Declare one primary endpoint (`H` on paper-valid-4, StereoSet-only), FDR table for rest. **0 GPU** |
| **K** | Post-hoc power circular (simulated at observed `ρ`) (8.K) | 🟡 | **Fixed in spirit** — labeled sensitivity analysis, but still cited as design | Keep as sensitivity only; future `n` planned on smallest meaningful `ρ=0.6`. **0 GPU** |
| **—** | Effective `n` overstates (prompts clustered by template) | 🟠 | **Open** | Cluster bootstrap at source-item level (needs `context`/`example_id` in `pair_meta`, now added for future runs). **0 GPU** |

### 16.4 Causal / mechanistic

| ID | Gap | Severity | Status | Anchor | Fix |
|---|---|---|---|---|---|
| **L** | Zero ablation OOD; single random baseline; capability only on bias prompts (8.L) | 🔴 for "surgery fails" headline | **Open** — Exp6 has 3 conditions but single random, no renormalization | Thesis hub concentration; CALM rerouting; Covert removal; Psalta oracle | E16 robust ablation: 20 random sets + **router masking + renormalization** vs zero vs mean vs noise; held-out WikiText/MMLU capability; deletion AUC with CI. **~480 GPU-min** |
| **M** | Synergy fraction ad-hoc, no CI, `n=20`, cherry layers, "universal" language (8.M) | 🟠 | **Open** | Shapley-Taylor (Sundararajan Najmi); MSA interactions | Specify STII vs SIV; efficiency check on synthetic game; bootstrap `n=100`; sample ≥4 layers a priori; correlate synergy with `ρ` fidelity. **2×A100 4h** |
| **N** | Exp5 null invalid (permutes expert ids, not labels; pool includes cohort) (8.N) | 🟠→🔴 if demographic-specificity is headline | **Open pending phi** | EEG stability-aware; Psalta attribute tests | Leave-one-cohort-out pool, label permutation within benchmark strata, min-`n` shrinkage, replicate on 2 models + held-out split. **0 GPU** once phi |
| **AC/AG** | No conditioned/aligned-router positive control — diffuse-in-unaligned-MoE never contrasted with "MoE when *can* localize" (15.1.3/15.1.7) | 🟠 | **Open** | Nath AGER; Psalta MoSAIC | AGER-style fine-tune or LoRA-per-attribute + oracle router (200 pairs). **~8 GPU-h** (future-work if limited) |
| **AD/AE** | No drift / stability-aware reporting (15.1.4/15.1.5) | 🟡 | **Open** | FEAMOE; EEG stability | Report `H` drift across checkpoints / stability-filtered JS. **0 GPU** |

### 16.5 Scope, literature, reproducibility

| ID | Gap | Severity | Status | Fix |
|---|---|---|---|---|
| **O** | Harm model underspecified; Gemma "alignment defense" overclaimed; US-centric, binary, intrinsic-vs-extrinsic not scoped (8.O) | 🔴 if fairness language stays | **Partially fixed** — Ethics section added Sec 6, still lacks demographic/allocational scope, misuse disclaimer, stereotype-harm citation | Add Blodgett harm framing, deployment context, limitations (non-US, non-binary, intrinsic logprob ≠ generation harm), and Gemma caveat. **0 GPU** |
| **P** | Related Work misses closest competitors (8.P) | 🟠→🔴 at venue | **Open** — draft cites OLMoE/Mixtral/DBRX/Phi but not Covert21, Sundararajan17, Lundberg18-interaction, Zhou22, Nangia20, Zhao18, Gallegos24, Frantar23, plus MSA/Nath/FEAMOE/CALM/Psalta/survey | Expand to 3-para structure (§9.1) and add all 8 missing + 6 literature anchors; label preprints as preprints. **0 GPU** |
| **Q** | Repro not submission-grade: no bench/Shapley axiom tests, revisions not pinned, runner seed unused, provenance incomplete (8.Q) | 🟠 | **Partially fixed** — seed now used, `pair_meta` persistence fixed, provenance fields added, 2 test files landed (14) but manifest hash / model/dataset rev / GPU/precision still missing from `result.json` | Add manifest hash + model/dataset rev + GPU/precision + package lock hash; add axiom tests (efficiency, symmetry on synthetic game); publish checksums + anonymous artifact; gitignore `.aux`. **0 GPU** expected |

**Net assessment vs Sections 8–9**: No gap retracted. Gaps **G, L, O, P** upgraded to framing blockers because literature now gives reviewers canonical counters (MSA specialization, FEAMOE fair MoE, CALM cultural scope, Psalta oracle). Gaps **AA–AG** are *new* but mostly 🟠/future-work; they sharpen rather than reopen the pipeline — fixing A–F still dominates. The prior agent's prioritization (P0 = fix pipeline before new GPU) remains correct.

---

## 17. Updated experiment backlog — literature-motivated delta (2026-09-11)

*Supersedes Section 10 only where it conflicts; otherwise extends it. Letters AA–AG map to 16.1–16.5.*

### P0 — Must fix before new GPU (analysis-only, <1 GPU-h) — *unchanged order, plus two literature docs*

| ID | Title | Fixes | Artifact | Cost |
|---|---|---|---|---|
| E9 | Benchmark adapter audit + common manifest (15.1.1–15.1.4) — target_loc+polarity, WinoGender BLS/unsigned, answer-conditional scoring, `item_manifest.json` (5000 balanced, hashed) | B,C,D,F | `item_manifest.json`, unit tests, StereoSet-only interim `H` table | 0 GPU, 2 CPU-h |
| E10 | Bootstrap stratification fix + cluster bootstrap | E | recomputed `s04/s05/s07` | 0 GPU |
| E11 | Per-benchmark / per-category split + heterogeneity `Q`/`I²` | D,F,X,AF | `s09_per_benchmark.json` | 0 GPU |
| E12 | Per-layer `H` + per-position heatmap prep | R,AA | `s10_per_layer.json` | 0 GPU |
| E13 | Sanity controls (random router, label/label-shuffle, weight random) | Z | `s11_sanity.json` | 1×L40S 30 min |
| E14 | Provenance + metadata audit | Q | `reporting.py` patch, `.gitignore` | 0 GPU |
| **E14b** | **Literature Related-Work expansion** (new, zero-GPU) — add MSA thesis + survey taxonomy 3-para structure, label preprints | P, §15 framing | updated `Related Work` | 0 GPU |
| **E14c** | **Harm-model & scope framing** (new, zero-GPU) — Blodgett harm, deployment context, disaggregated tables, Gemma caveat | O,AF | updated `Discussion/Limitations` | 0 GPU |

### P1 — Same-unit causal core (small GPU)

| ID | Title | Fixes | GPU-min | Notes |
|---|---|---|---|---|
| E15 | Exp8 full ladder LOO (Mixtral, DBRX, GPT-OSS, Gemma) same manifest | H,4 | ~570 | Configs ready; closes granularity confound |
| E16 | Robust causal ablation — 20 random sets, renormalization/mean/noise controls, held-out WikiText+MMLU, deletion AUC CI | L | ~480 | "Surgery fails" only credible after this |
| E17 | Routing-induced bias diagnostic (FAMoE per-subgroup gate entropy) | §9.1.1 | 0 if logs else 60 | Answers "is routing itself skewed?" |

### P2 — Mechanism tests (powered, controlled) — *literature Delta*

| ID | Title | Fixes | Cost | Why now elevated |
|---|---|---|---|---|
| E18 | Within-model `k`-sweep (OLMoE top-1/2/4/8 inference; gate-entropy vs `H`) | G | 420 GPU-min | Tests sparsity mechanism without new architectures; Thesis says don't conflate trained vs inference |
| **E18b** | **Multidimensional MSA sample** — Shapley Modes on 50 pairs × 2 layers (Monte-Carlo lesions, cf. MSA) → per-token/per-output contribution | AA,M | 180 GPU-min | Directly answers reviewer citing MSA: shows averaging hides structure |
| E20 | Interaction-index validation — STII vs Shapley-Taylor vs ad-hoc synergy on synthetic additive/interacting game; bootstrap CI ×100 pairs | M | 240 GPU-min | Replaces ratio with principled index |
| **E26** | **Regularisation / router-entropy audit** (new) — tabulate load-balance coeff, router entropy per model, correlate with `H` | AB,S,T | 0 GPU | Zero-cost check the Thesis "weight≠importance" confound |
| **E27** | **Positive-control: AGER / oracle-router pilot** (new, optional) — LoRA-per-attribute (gender/race) + oracle router on 200 pairs à la MoSAIC-ReID | AC,AG | ~480 GPU-min | Turns "MoE never localizes" into "bias diffuse even when MoE *can* localize" — stronger thesis |
| E19 | Scale-matched family control (Qwen3-30B-A3B if license allows) | G,H | 300 GPU-min | Ladder power `n→7–8` |

### P3 — Scope & reporting (polish) — *adds literature citations*

| ID | Title | Fixes | Cost |
|---|---|---|---|
| E21 | Generation-based bias (greedy decode 200 ambiguous BBQ) vs logprob gap | V | 120 GPU-min/model |
| E22 | Template robustness (3 prompt formats × 200 pairs) | U | 60 GPU-min/model |
| E24 | Compute + CO₂ table from slurm logs | Y | 0 GPU |
| E25 | Figure fixes (CIs, Description, N-note, replace `n=4` boxplot, Hedges’g) | §13.1 | 0 GPU |
| E28 | Cross-cultural tag split (BBQ religion/gender, StereoSet profession/race) per CALM dims | AF,W | 0 GPU |

### P4 — Long-term / future work — *now includes MoSAIC-style programme*

- Trained-`k` comparison (checkpoints trained at multiple `k`) — true causal sparsity test.
- Expert rewriting edit (MuMoE-style) — targeted expert editing despite diffuse `H`.
- Full AGER training (Nath) — attribution-guided router fine-tune for bias mitigation, contrasted with pruning.
- Multilingual fairness (C-Eval) once policy allows.

**Execution order after P0** (revised): P0 E9–E14c → sacct triage → P1 E15 → P1 E16 → P2 E18+E18b+E26 (zero/low cost) → P2 E27 (if time) → P3 E24/E25 → then E19/E20/E21/E22.

Until P0 (esp. manifest + BBQ fix + rename) lands, P1–P3 still add precision to a mis-specified pipeline — **that priority has not changed**.

---

## 18. Agent context sheet — distilled takeaways from the 8 papers (for future agent runs)

*Copy-paste this into the next agent's prompt or keep at hand. Each bullet is the one line a reviewer will use against you, and the one-line rebuttal you must have.*

1. **MSA (Dixit et al. 2506.19732) + Thesis (Hamburg 2025, edoc 292)**: Lesion Shapley on Mixtral-8×7B *does* find localized language experts — so your diffuse-bias finding is **bias-specific, not "MoE never specializes."** Cite as supporting contrast; adopt their exact `v(S)` lesion definition and report **Shapley Modes** (per-token/per-output) to show averaging wasn't hiding structure. Also: weight magnitude ≠ causal importance → your `routing_freq` control is expected to mimic `phi` if `phi` tracks frequency.

2. **Nath 2026 (IJST, Indic NMT) / AGER-MNMT**: Adapters + MoE + SHAP/LIME **can** localize *when conditioned on language/family*; AGER shows attribution-guided routing stabilizes experts. Your generic MoE diffuse result stands, but you must (a) show token/layer attribution workflow, (b) discuss attribution-guided router as *alternative mitigation* to pruning, and (c) run a **conditioned-router positive control** if you want the strongest claim.

3. **FEAMOE (Sharma 2210.04995)**: MoE of *linear* experts with fairness constraints handles **fairness drift** while staying explainable via fast Shapley. Takeaway: fairness is **temporal, definition-dependent**; snapshot `H` + informal `gap` ≠ fairness guarantee. Map your `gap` to a formal fairness criterion and test drift (even just across shards/checkpoints).

4. **Stability-aware Shapley-guided MoE (Pattern Recognition 2026, EEG)**: Lesson is **stability filtering** before claiming specialization. Your Exp5 `D_JS` must be stability-weighted, with leave-one-cohort-out pools and label permutation — otherwise `D_JS=0.221` is routing-structure heterogeneity, not demographic specialization.

5. **CALM (NeurIPS 2025, Shen et al.)**: Culture-aware MoE that *disentangles* task vs explicit/latent cultural signals into contrastive clusters + culture-routed MoE + reflective loop — **beats generic MoE cross-culturally**. Means your US-centric, binary, English-only, intrinsic-logprob construct is narrow. Scope to "US-centric intrinsic stereotype gap" and add per-cultural-dimension split; cite CALM as future architecture that *might* localize cultural bias if explicitly routed.

6. **MoSAIC-ReID (Psalta 2512.08697)**: **LoRA-per-attribute + oracle router + ablation + GLM/tests** is the template for causal attribute importance (color >> hat). Replicate that scaffold for bias: *even when* experts are forced per attribute with oracle routing, does bias stay diffuse? That's the decisive test your ladder alone cannot give. Also copy their statistical layering (not just `ρ`).

7. **Scalable & Interpretable MoE survey (Preprints 202507.0283)**: Gives the **formal taxonomy** your Related Work should be structured around (foundations / optimization / attribution-via-modularity / metrics). Use it to justify `H/G/t10pct/exp(H)` choices, discuss load-balance / shared-expert / quantization nuances, and checklist your optimization assumptions.

8. **Cross-paper meta-lesson**: Three independent lines (MSA, Nath, CALM/Psalta) **converge**: MoE *can* specialize when **conditioned or trained to**; generic off-the-shelf MoE *appears* diffuse for bias. Your paper's strongest, reviewer-proof contribution is therefore **not** "surgical debiasing via top-`k` ablation fails" alone (needs E16 robustness), but **"observation: routing-contrast diffuse + does not predict causality; mechanism: synergy dominates early layers; scope: bias-specific, not Architecture-universal, and mitigation requires routing-level, not expert-level, intervention."** Frame accordingly and every one of these papers becomes supporting, not hostile.

*For the paper itself, add these 8 to Related Work plus the 8 venues already missing (Covert 21/22, Sundararajan 17, Lundberg 18-interaction, Zhou 22, Nangia 20, Zhao 18, Gallegos 24, Frantar 23), total +16; clearly label arXiv/preprint vs peer-reviewed.*

---

*Document extended 2026-09-11 by eighth-pass literature-grounded review (Section 15), consolidated re-inventory (Section 16), updated backlog (Section 17), and agent cheat sheet (Section 18). Sections 0–14 are preserved as historical record; Sections 15–18 are current. Next step remains P0 E9–E14c before further GPU spend.*

---

## 19. Ninth-pass: actionable revision for the "bias is non-localizable even with surgery / router skewing" thesis (2026-09-11)

**Why this pass exists**: The user reports "I feel like you didn't do the things" and explicitly offers more GPU if it strengthens the paper. Prior passes (Sections 8–18) were exhaustive but left the *action* buried under 1200 lines of history. This section is the **one-page, reviewer-clean, costed answer** to: *what is the paper really trying to prove, where does it actually fail, and what exact GPU should be run next*.

### 19.1 What the paper is *actually* trying to prove (and how it says it)

The draft's core claim (abstract + Introduction + `study_design_C1_C4.md` RQ1–RQ3) is:

> Using a **combination of Shapley-style methods** (routing-contrast heuristic at scale + exact lesion Shapley on small coalitions + Shapley interaction values + leave-one-out) to attribute **social bias** (StereoSet/BBQ/WinoGender gap → `V`) to **experts** in **6 MoE LLMs (6B–132B)**, bias attribution is **diffuse, synergy-dominated, and non-predictive of causal effect** — therefore **surgical expert ablation and router-level skewing do not localize or remove bias without destroying capability**.

Four complementary attributions are already in the codebase:

| Method | Where in code | Scale | What it tests |
|---|---|---|---|
| **Routing-contrast** `Δrouting_weight × gap` | `shapley.py:compute_routing_contrast` | 5000 pairs × 6 MoE + 4 dense | Observation: is attribution concentrated? (H1) |
| **Exact lesion Shapley** (2^K coalitions) | `shapley.py:compute_exact_shapley_for_pair` + `_build_coalition_payoff_cache` | 50 pairs × few layers | Causality: does routing-contrast rank predict true causal rank? (Exp7) |
| **Shapley Interaction Value (SIV)** | `shapley.py:compute_shapley_interactions_for_pair` | 20 pairs × 2 layers | Mechanism: is bias pairwise synergy vs individual? (Exp3) |
| **Ablation curves** (proxy / frequency / random) | `shapley.py:compute_ablation_curve` | 30–60 pairs × 3 MoE | Surgery: does removing top-phi experts reduce gap surgically? (Exp6/Exp4) |
| **Layer-LOO** (dense) | `shapley.py:compute_dense_layer_contrast` | 1800–4000 pairs × 4 dense + 2 MoE | Granularity control: does dense vs MoE split survive same mechanism? (Exp2/Exp8) |

**The missing complementary method the thesis promises but never runs**: **router skewing** (scale router logits toward/away from high-phi experts without ablating, then renormalize). Surgery zeros experts (OOD); skewing keeps capacity but reroutes. If skewing *also* fails, non-localizability is routing-invariant — a much stronger claim than "zero ablation hurts perplexity."

### 19.2 Honest appraisal of the current evidence (NeurIPS reviewer lens)

**What is already strong and should be kept**:

- **Diffuseness is real**: `H≈0.88–0.92`, `t5` 2–11%, `t10%` 35–66% across *all* MoE rungs, stable across shards (Mixtral/DBRX shard ΔH ≤0.004) and across 400→5000 pairs (ΔH ≤0.022 except DBRX). Dense controls `H≈0.63–0.76` separate cleanly. This is not a metric floor.
- **Proxy ≠ causal**: Exp7 `ρ∈[−0.085,+0.058]` on Mixtral/OLMoE/Phi with exact 2^K lesions is a clean, honest null — rare in MoE papers — and directly falsifies "routing-contrast is a cheap Shapley proxy."
- **Synergy signal**: 70–74% early-layer interaction mass via exact SIV is mechanistically interesting and MoE-specific (dense has no comparable 2^K).

**What is not yet reviewer-proof for "surgery / skewing fails"**:

1. **Surgery claim is OOD and under-controlled**: Zero-ablation of 10% experts is a state never seen in training; it *must* hurt PPL. Only 1 random baseline, no renormalization/mean/noise controls, and capability measured on bias prompts (not held-out WikiText/MMLU). Reviewer will say: "You showed zero ablation is OOD, not that bias is inseparable."
2. **No router-skewing experiment at all**: Thesis says "even if you do specific surgery on them or router skewing" — second half has zero data. This is the single easiest way to strengthen the paper with new GPU and is cheap (same hooks as ablation).
3. **Exp8 ambiguous**: 2-point LOO split (`H=0.736` in dense band vs `0.899` in MoE band) is not a trend; 4 configs are written and smoke-tested but never run.
4. **Pipeline validity flags (A–F) still open**: Whole-string logprob, BBQ/WinoGender bugs, no common prompt battery. Even a perfect surgery/skewing experiment on a mis-specified `V` and mixed benchmark set will be dismissed.

**Was prior GAP-ANALYSIS (Sections 0–14) good?** Yes — exceptionally thorough on validity (A–F) and stats (E–K). Its flaws: (i) 1100+ lines with duplicated history, burying the action; (ii) under-weighted the *positive-control* lesson from literature (MSA finds language experts where you find bias diffuse — frame as bias-specific, not universal); (iii) no explicit router-skewing spec despite thesis promising it. Sections 15–18 fix (ii); this section fixes (i) and (iii).

### 19.3 What to run next (the only GPU that directly hardens the thesis)

> **Rule**: No new science until Tier 0 validity is re-established; otherwise reviewers reject on pipeline before reading results.

**Tier 0 (validity, ~6 GPU-hours, must be first)**
- Freeze `item_manifest.json` (5000 balanced, hashed) + fix scoring to **answer-tokens-conditional** + BBQ `target_loc`/`polarity` + WinoGender unsigned/BLS. Rerun **OLMoE + Mixtral routing-contrast on common manifest** (1×L40S 3h + 4×H100 1h). If `H` stays `≈0.88–0.92`, diffuseness survives correct `V`.

**Tier 1 (thesis-hardening, ~30 GPU-hours, do in this order)**

1. **T1.1 Robust surgery** (E16, 480 GPU-min): 20 randoms + 3 operators (zero, **renorm**, mean) + held-out WikiText/MMLU, deletion AUC CI, 4 models (OLMoE, Phi, Mixtral, DBRX). Proves surgery fails *even without OOD*.
2. **T1.2 Router skewing** (new, 360 GPU-min): **New contribution** — skew logits `logits + α·(−|phi|)` with `α∈{0.5,1,2}`, renormalize, same `Δgap`/PPL/MMLU. First test of "router skewing also fails." Cheap (same hooks as ablation, no training). **If skewing also fails, non-localizability is routing-invariant.**
3. **T1.3 LOO full ladder** (E15, 570 GPU-min): 4 ready configs (Mixtral/DBRX/GPT-OSS/Gemma LOO, 30–50 pairs). Closes granularity confound; turns ambiguous `n=2` into trend.
4. **T1.4 MSA Modes** (E18b, 180 GPU-min): 50 pairs × 2 layers × 200 Monte-Carlo orderings, per-token/per-output Shapley (Dixit) + STII validation, 2 models. Shows averaging didn't hide localization; mechanism.

**Tier 1 total**: ~1590 GPU-min (~26.5 GPU-hours, 7–8 sbatch jobs, all under 960 cap). Cost at PACE ICE: <1% of a typical NeurIPS paper's budget.

**Tier 2 (nice-to-have, after Tier 0/1)**: within-model `k`-sweep, AGER/oracle-router positive control (LoRA-per-attribute), generation-based BBQ, regularization audit (0 GPU).

**What NOT to run**: Another 5000-pair ladder rung before Tier 0 — `n=6→7` barely moves power and wastes the validity fix.

**Cross-links**: Full literature context → `LITERATURE_CONTEXT.md` (8 papers, implications). Full costed submit lines + reporting spec → `GPU_EXPERIMENT_PLAN.md` (T0–T2 with exact `sbatch` commands and success criteria).

*If you can run only one weekend, run Tier 0 (OLMoE+Mixtral rerun) + T1.1 (renorm surgery) + T1.2 (router skewing) + T1.3 (4×LOO). That is the minimal resubmission that turns "diffuse observation" into "causal, routing-invariant non-localizability."*
