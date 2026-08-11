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
GPT-OSS-120B resubmitted this session (jobs **5575743**, **5575744**,
4h/8h walltime after the config time-limit fix below) --- **PENDING**,
blocked by a cluster-side `Reserved for maintenance` scheduler reason
despite idle H100 capacity (see `CLUSTER-STATUS.md` live-poll section).
A redundant Mixtral refresh (job 5575538, from a prior session) was
**cancelled** this session after confirming it requested 1440 GPU-min
against the `coc-ice` 960 GPU-min/job cap and could never run; not
blocking (July capture already verified usable).

### Experiment 4 --- Independent Cross-Check (Robustness)
- **DONE** for 6 models (OLMoE, Mixtral, Phi-3.5-MoE, DBRX, Gemma-4-26B,
  OLMo-7B dense) under Exp6 (Section 5.6/causal check). GPT-OSS-120B
  ladder extension resubmitted this session (job 5575745).

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
| GPT-OSS-120B | resubmitted (job 5575745, 8h/30 pairs) | **PENDING** in queue at last poll |

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
| OLMoE | H_lloo=0.758 (summary only) | DONE at summary level; per-pair capture resubmitted with `--save-per-pair-phi` as job **5575752** (queued) after discovering job 5575255 only persisted summary $\phi$ |
| Phi-3.5-MoE | H_lloo=0.899, per-pair CI $[0.842,0.941]$ | **DONE with per-pair CI** (job 5575536, $n=50$, bootstrap $n_{\mathrm{boot}}=5000$, seed 42); integrated into paper Appendix B |

**Gap remaining**: OLMoE's per-pair payload is still not on disk (job
5575752 queued); Phi-3.5-MoE's landed this session and is fully
integrated. Paper's Appendix B now reports this split accurately.

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

### Still open (GPU-bound, queue-saturated at last poll)
10. **Exp3 Collectivity (DBRX, GPT-OSS-120B)** --- resubmitted (5575743,
    5575744) with QOS-max walltime (4h/8h) after discovering the prior
    submissions' 3h time limit was too short for the observed
    ~9.6min/pair $\times$ 2-layer cost; reduced to 10 pairs/model to fit.
    **PENDING** in queue.
11. **Exp6 GPT-OSS-120B** --- resubmitted (5575745) with 8h walltime
    (was 3h) and reduced pairs (30) + reduced routing-freq sample (60,
    was 200). **PENDING** in queue.
12. ~~Exp8 per-pair (Phi-3.5-MoE)~~ --- **DONE**, job 5575536 landed
    ($n=50$, bootstrap CI integrated into Appendix B). OLMoE's per-pair
    capture resubmitted with `--save-per-pair-phi` as job **5575752**,
    still **PENDING** in queue.

### Data/Code Hygiene
13. ~~Gemma-4-27B phantom~~ --- documented, non-issue.
14. ~~Kaggle payload manifest~~ --- done (`sghose0/moe-bias-routing-shapley-perpair-phi`).
    Dense v1 per-pair payloads are NOT yet published there --- consider a
    follow-up dataset version once Exp3/6/8 fully land.
15. ~~REPRODUCIBILITY.md~~ --- done.

---

## 4. GPU Resource Plan (ICE Cluster)

See `CLUSTER-STATUS.md` for the live job table (job IDs, states, notes). This
file is not duplicated here to avoid drift between the two docs.

---

## 5. Execution Priority (remaining)

1. **Poll cluster periodically**: `ssh login-ice.pace.gatech.edu 'squeue -u sghose7'`
   (VPN must be off). 4 jobs (5575743 DBRX Exp3, 5575744 GPT-OSS Exp3,
   5575745 GPT-OSS Exp6, 5575752 OLMoE Exp8 per-pair) are **PENDING**,
   all blocked by a `Reserved for maintenance` scheduler reason despite
   idle H100 capacity (see `CLUSTER-STATUS.md` live-poll section) --- not
   a bug on our side, may need a PACE support ticket if it persists. Job
   5575538 (redundant Mixtral refresh) was cancelled: it exceeded the
   960 GPU-min/job QOS cap and could never run.
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
