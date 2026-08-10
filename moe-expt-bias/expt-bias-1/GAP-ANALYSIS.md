# PhD-Style Gap Analysis: MoE-Bias Study (expt-bias-1)

**Date**: 2026-08-10  
**Paper status**: 9-page ACM draft at `moe-expt-bias-2/moe_bias_report_acm_v2.tex` (compiles clean)  
**Repo**: `/Users/ronnie.ghose/src/priv/gatech/research-projs/moe-breakdown`

---

## 1. Professor's Criticism — Addressed vs Remaining

| Criticism Point | Status | Evidence |
|---|---|---|
| "H1 supported once GPT-OSS dropped; GPT-OSS excluded for quantization reasons" | **RESOLVED** | GPT-OSS-120B v1 now VALID (2000 pairs, H=0.8764, G=0.7274). Paper reports exact permutation p-values on ALL 6 rungs: ρH=+0.754, p=0.106. Framed honestly: directionally consistent, underpowered (n≤6). |
| "Metric measures routing structure, not causal bias magnitude; then Exp1 entropy and Exp5 JS aren't measuring bias concentration" | **RESOLVED in draft** | Discussion §1 explicitly resolves: attribution geometry is a routing-structure quantity; caveat applies uniformly to Exp1 entropy AND Exp5 JS. Framed as localization geometry, not bias magnitude. |
| "No CIs, SEs, tests — analyses must be statistically robust" | **MOSTLY RESOLVED* | MoE ladder: 95% block-bootstrap CIs (5000 draws, seed 42, stratified) for all 6 rungs × 4 metrics (Table 2). Dense baselines: NO CIs (no per-pair φ payloads exist) — flagged CI:pending. **Action**: Dense per-pair jobs submitted (8 shards RUNNING, jobs 5575082-5575089). |

---

## 2. Study Catalog vs On-Disk Reality

### Experiment 1 — Sparsity Ladder (RQ1/C1)
| Model | v0 (400) | v1 (5000) | Per-pair φ | Status |
|---|---|---|---|---|
| OLMoE-1B-7B | ✓ | ✓ | ✓ | DONE |
| Phi-3.5-MoE | ✓ | ✓ | ✓ | DONE |
| Mixtral-8x7B | ✓ | ✓ | ✓ | DONE |
| DBRX-132B | ✓ | ✓ | ✓ | DONE |
| GPT-OSS-120B | ✓ | ✓ (2000) | ✓ | **v2 RUNNING** (job 5575070, 5000 pairs, ETA ~2.7h) |
| Gemma-4-26B | ✓ | ✓ | ✓ | DONE (null-bias flag) |
| **Gemma-4-27B** | ✗ | ✗ | ✗ | **PHANTOM** — HF 404, model doesn't exist (26B is real rung) |

### Experiment 2 — Dense Baselines (RQ2/C4)
| Model | v0 (400) | v1 (1800/4000) | Per-pair φ | Status |
|---|---|---|---|---|
| OLMo-7B | ✓ | ✓ (1800) | **SUBMITTED** (jobs 5575082-83) |
| Phi-3.5-Mini | ✓ | ✓ (4000) | **SUBMITTED** (jobs 5575084-85) |
| Llama-2-7B | ✓ | ✓ (1800) | **SUBMITTED** (jobs 5575086-87) |
| Llama-3.1-8B | ✓ | ✓ (1800) | **SUBMITTED** (jobs 5575088-89) |

### Experiment 3 — Collectivity Check (C2-lite)
- **Not run at all** — no configs, no results. Requires `shapley_method: exact` on small ablation subsets per catalog. Zero artifacts on disk.

### Experiment 4 — Independent Cross-Check (Robustness)
- **Partial**: ablation curves exist for 4 models (OLMoE, Mixtral, Phi-3.5-MoE, OLMo-7B dense) under Exp6. Missing for DBRX, GPT-OSS-120B, Gemma-4-26B.

### Experiment 5 — Demographic Specificity (RQ3/C3)
- **DONE** for OLMoE (5000 pairs, 85 cohorts, JS CI [0.206,0.231]). Exp5 flag added to paper (no Winogender in Exp5 vs ladder).

### Experiment 6 — Ladder-Wide Causal Ablation (Reviewer-Prioritized)
| Model | Ablation Curve | Status |
|---|---|---|
| OLMoE-1B-7B | ✓ (60 pairs) | DONE, cited in paper |
| Phi-3.5-MoE | ✓ (30 pairs) | DONE, cited in paper |
| Mixtral-8x7B | ✓ (30 pairs) | DONE, cited in paper |
| OLMo-7B (dense) | ✓ (60 pairs) | DONE, cited in paper |
| **DBRX** | ✗ | **MISSING** |
| **GPT-OSS-120B** | ✗ | **MISSING** |
| **Gemma-4-26B** | ✗ | **MISSING** |

### Experiment 7 — Proxy-vs-Exact Shapley Agreement
| Model | Result | Status |
|---|---|---|
| Mixtral-8x7B | ✓ (mean ρ=-0.085/+0.058) | **DONE** (null result), cited in paper |
| OLMoE | ✗ | **MISSING** |
| Phi-3.5-MoE | ✗ | **MISSING** |

### Experiment 8 — Same-Mechanism Comparison (Method-Confound)
| Model | Result | Status |
|---|---|---|
| OLMoE | H_lloo=0.751 (summary) | **DONE** at summary level, cited |
| Phi-3.5-MoE | H_lloo=0.883 (summary) | **DONE** at summary level, cited |
| Per-pair payloads | ✗ | **MISSING** — no CIs/bootstrap for Exp8 |

---

## 3. Critical Gaps for Paper Acceptance (TIST/ACM)

### Must-Fix Before Submission
1. **Dense CIs** — 8 shards RUNNING (5575082-5575089). Need to verify completion + extend s04 glob to `exp2-dense-*`.
2. **GPT-OSS 5000 pairs** — job 5575070 RUNNING, ETA ~2.7h. Ladder uniformity (all 5000) strengthens H1 power story.
3. **Exp6 ladder completion** — DBRX, GPT-OSS, Gemma causal ablation missing. These 3 models sit at distinct sparsity points; without them the "mixed" causal verdict is incomplete.
4. **s04 bootstrap glob fix** — currently only `exp1-concentration-*`; must include `exp2-dense-*` after dense per-pair lands.

### High-Value Additions (Reviewer-Prioritized)
5. **Exp3 Collectivity** — Shapley interaction values (marginal vs synergy). Zero cost to design; need small GPU runs per model. Directly speaks to "collective routing" vs independent experts.
6. **Exp7 on OLMoE/Phi** — proxy agreement null on Mixtral; needs replication on other models. Low-cost (small n, exact Shapley tractable on Mixtral, need routing_contrast vs exact on others).
7. **Exp8 per-pair** — same-mechanism LOO on MoE. Per-pair would give CIs/bootstrap, making the "H_lloo drops to dense band" claim statistically robust.

### Data/Code Hygiene
8. **Gemma-4-27B phantom** — document clearly as nonexistent HF model; 26B is the rung.
9. **Kaggle payload manifest** — already done (`sghose0/moe-bias-routing-shapley-perpair-phi`).
10. **REPRODUCIBILITY.md** — done.

---

## 4. GPU Resource Plan (ICE Cluster)

### Current Queue (as of 2026-08-10)
- **5575070**: GPT-OSS 5000 pairs (4×H100, 04:00:00, ~2.7h remaining)
- **5575082-5575089**: 8 dense per-pair shards (4 models × 2, 1×L40S each, 05:00:00, just submitted)
- **Idle GPU capacity**: 4 H200 nodes (64 GPUs free), 3 H100 nodes (22 GPUs free), 4 L40S nodes (28 GPUs free), A100/A40/V100/RTX/A40/MI210

### Immediate Submissions (Phase 1 — this session)
| Job | Model | Config | GPUs | Time | QOS | Notes |
|---|---|---|---|---|---|---|
| Exp6-DBRX | DBRX | study.dbrx.concentration | 2×H200 | 04:00:00 | 360 min | Reuse Exp1 v1 config; submit via submit_slurm_experiment4.py |
| Exp6-GPT-OSS | GPT-OSS-120B | study.gpt-oss-120b.concentration | 4×H100 | 04:00:00 | 960 min | Same as GPT-OSS 5000 config, just add --max-pairs 100 |
| Exp6-Gemma | Gemma-4-26B | study.gemma4-26b.concentration.v1.yaml | 1×A100/L40S | 04:00:00 | 240 min | Model exists (26B); causal check only |
| Exp3-OLMoE | OLMoE | study.olmoe.concentration | 1×L40S | 03:00:00 | 180 min | Exact Shapley on top layers only |
| Exp3-Mixtral | Mixtral | study.mixtral-8x7b.concentration | 1×L40S | 03:00:00 | 180 min | |
| Exp3-Phi | Phi-3.5-MoE | study.phi3.5-moe.concentration | 1×L40S | 03:00:00 | 180 min | |

### Phase 2 (after Phase 1 completes)
| Job | Model | Config | GPUs | Time | QOS |
|---|---|---|---|---|---|
| Exp7-OLMoE | OLMoE | study.olmoe.concentration | 1×L40S | 02:00:00 | 120 min |
| Exp7-Phi | Phi-3.5-MoE | study.phi3.5-moe.concentration | 1×L40S | 02:00:00 | 120 min |
| Exp8-OLMoE per-pair | OLMoE | study.olmoe.lloo | 1×L40S | 04:00:00 | 240 min |
| Exp8-Phi per-pair | Phi-3.5-MoE | study.phi3.5-moe.lloo | 1×L40S | 04:00:00 | 240 min |

**Total GPU-min estimate**: Phase 1 ≈ 2160 min (within coc-ice, staggered across nodes); Phase 2 ≈ 720 min.

---

## 5. Execution Priority

1. **Submit Phase 1 jobs NOW** (parallel, disjoint GPU types — H200 for DBRX, H100 for GPT-OSS, A100/L40S for Gemma, L40S for Exp3). Cluster has capacity.
2. **Monitor 5575070 + dense shards** — when complete, pull results, rerun s03/s04, extend glob, update paper.
3. **Submit Phase 2** once Phase 1 lands.
4. **Paper integration loop**: each batch of results → update draft → compile → vision check.

---

## 6. Next Actions (This Session)

- [ ] Submit Exp6 DBRX, GPT-OSS, Gemma + Exp3 all 6 models (Phase 1)
- [ ] Monitor 5575070 (GPT-OSS 5000) completion
- [ ] Monitor dense per-pair shards (5575082-89) completion
- [ ] When complete: pull, extend s04 glob, rerun bootstrap, regenerate figures, update draft
- [ ] Submit Phase 2 jobs

*Document generated from live repo state, study catalog, cluster inventory, and paper draft audit.*