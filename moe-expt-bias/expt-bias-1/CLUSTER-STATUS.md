# ICE Cluster Status & Experiment Tracking (expt-bias-1)

Last updated: 2026-08-11 (cluster now fully inaccessible: PACE quarterly
maintenance began at the exact boundary predicted below).

## Root cause found and worked around, 2026-08-11 ~04:30-05:00 cluster time

The `ReqNodeNotAvail, Reserved for maintenance` reason blocking jobs
5575743/5575744/5575745/5575752 (DBRX Exp3, GPT-OSS-120B Exp3, GPT-OSS-120B
Exp6, OLMoE Exp8 per-pair) is **not** a scheduler bug and **not** a stale
per-job cache (both were ruled out earlier this session). It is a real,
non-admin-visible upcoming maintenance reservation: `scontrol show
reservation` never lists it (reservations aren't shown to non-privileged
users on this cluster), but the scheduler still refuses to start any job
on a specific node if the job's requested `--time` would run past the
reservation's start. **Confirmed empirically**: a 2-minute probe job and a
10-minute probe job pinned to an idle H100 node (`--nodelist=...`) both
went straight to RUNNING with reason `(None)`; the same node rejected a
4-hour request with `Reserved for maintenance`. Binary-searching the
boundary on 3 different nodes independently (atl1-1-03-012-28-0,
-013-3-0, -013-8-0, atl1-1-03-004-21-0) all converged on the same
**~1h10m-1h22m runway from ~04:30-04:45 EDT probe time**, i.e. the
maintenance window starts approximately **05:50-06:00 EDT on 2026-08-11**
cluster-wide.

**Workaround applied**: cancelled all 4 stuck jobs and resubmitted each
pinned to a specific currently-idle node (`--nodelist=...`) with `--time`
reduced to the largest value confirmed safe under the maintenance
boundary (session-wide across parallel probes; not 960 GPU-min QOS-cap
limited except where noted):

| New job ID | Was (stuck ID) | Study | Node | `--time` used | Outcome |
|---|---|---|---|---|---|
| **5575799** | 5575743 | Exp3 collectivity, DBRX-132B | atl1-1-03-012-28-0 | 01:10:00 | RUNNING (verified healthy) |
| **5575791** | 5575744 | Exp3 collectivity, GPT-OSS-120B | atl1-1-03-013-3-0 | 01:10:00 | RUNNING (verified healthy) |
| **5575800** | 5575745 | Exp6 ladder extension, GPT-OSS-120B | atl1-1-03-013-8-0 | 01:00:00 | RUNNING (verified healthy) |
| **5575798** | 5575752 | Exp8 per-pair LOO, OLMoE-1B-7B | atl1-1-03-004-21-0 | 01:10:00 | **COMPLETED** ($H=0.7361$, Gini$=0.6239$, per-pair CI landed; integrated into paper) |

Because the maintenance-window runway (~1h) is much shorter than the
original 4h/8h requests, the still-RUNNING Exp3/Exp6 jobs may hit
`TIMEOUT` before finishing their full pair/layer schedule; re-poll after
the maintenance window passes (`squeue -u sghose7`) and resubmit any
`TIMEOUT`'d job with the same node-pinning trick once new idle nodes are
available, using a fresh probe to find the next safe `--time` window.

## CONFIRMED: PACE quarterly maintenance, 2026-08-11 06:00 -- 2026-08-13 23:59

The maintenance-window hypothesis above is now **directly confirmed**: SSH
to `login-ice.pace.gatech.edu` is refused cluster-wide with the login
banner "PACE is performing our quarterly maintenance period scheduled to
begin at 6:00am on Tuesday, August 11, 2026, and conclude by 11:59pm on
Thursday, August 13, 2026. Access to the cluster is restricted." The
empirically-derived boundary (~05:50-06:00 EDT) matches the announced
06:00 start almost exactly. This is a **hard, scheduled, non-actionable
external block** for the full window -- no further node-pinning,
resubmission, or polling is possible until it lifts; login itself is
closed, not just job scheduling.

**Fate of the 3 jobs that were RUNNING when the window closed**
(5575799 DBRX Exp3, 5575791 GPT-OSS-120B Exp3, 5575800 GPT-OSS-120B Exp6):
unknown until the cluster reopens. They were healthy and progressing at
last poll (before 06:00); PACE maintenance typically drains/requeues or
kills running jobs on affected nodes. **First action once the cluster
reopens (after 2026-08-13 23:59)**: `sacct -j
5575799,5575791,5575800 --format=JobID,State,ExitCode,Elapsed -X -n`. If
any show `COMPLETED`, pull results immediately. If `TIMEOUT`/`CANCELLED`/
`NODE_FAIL`, resubmit with a fresh `sinfo -p ice-gpu -N` idle check (the
maintenance is now over, so plain `sbatch` without node-pinning should work
again) using the exact submit lines recorded above.

**Cancelled this session**: job **5575538** (redundant Mixtral-8x7B Exp3
refresh) --- confirmed mis-submitted: requested `gres/gpu=4` x `06:00:00`
= 1440 GPU-min, exceeding the `coc-ice` 960 GPU-min/job cap, so it could
never run (scheduler's projected start was 2026-08-16, a stale backfill
estimate). Non-blocking: the July capture it would have refreshed is
already verified non-degenerate and used in the paper.

## GPU Inventory (sinfo -p ice-gpu, 2026-08-10)

Partition `ice-gpu`, ALL state `up`, TIMELIMIT `16:00:00`.

| GPU type | nodes | GPUs/node | notes |
|---|---|---|---|
| H100 | 6 | 8 | biggest pool; headroom after study jobs |
| H200 | 6 | 8 | used by DBRX (2xH200 fits 132B bf16) |
| L40S | 4 | 8 | OLMoE-class runs |
| A100 | 4 | 2 | max 2/node (160GB) --- DBRX/GPT-OSS need H200 instead |
| A40 | 2 | 2 | --- |
| RTX_6000 | 2 | 4 | --- |
| MI210 | 2 | 2 | --- |
| V100 | 11 | 2 | slowest; use only for tiny runs |

QOS `coc-ice` GPU-minute cap: MaxTRESMinsPerJob = gres/gpu=960 (i.e. 2 GPUs x 8h
or 4 x 4h). Submissions above that are rejected (observed: 24h/12h requests
rejected; 4xH100 4h accepted at ~2s/pair).

**Time-limit bug found and fixed this session**: `submit_slurm_experiment3.py`
passes `--time={slurm.time}` straight from the study config's `slurm.time`
field (overriding the sbatch template's 4h default), and both
`study.dbrx.concentration.yaml` and `study.gpt-oss-120b.concentration.yaml`
had `time: 03:00:00` --- too short for Exp3's ~9.6min/pair x 2-layer cost
(DBRX's active-expert sets run 11-14 experts, exceeding the `max_k=10`
truncation warning but still costing exact-Shapley time) and for Exp6's
full-ladder ablation schedule on GPT-OSS-120B. Both jobs TIMEOUT'd at
exactly 3:00:0x (visible in slurm logs). Fixed by bumping `slurm.time` to
the QOS-max for each GPU count (DBRX 4xH100 -> `04:00:00`; GPT-OSS-120B
2xH100 -> `08:00:00`) and reducing pair counts to fit
(`--max-pairs 10` for Exp3 on both; `--max-pairs 30 --routing-freq-pairs 60`
for Exp6 on GPT-OSS-120B). Resubmitted as jobs **5575743** (DBRX Exp3),
**5575744** (GPT-OSS-120B Exp3), **5575745** (GPT-OSS-120B Exp6).

## Queue (polled directly via `squeue -u sghose7`, this session)

| Job ID | Study | State (last poll) | Notes |
|---|---|---|---|
| 5575070 | exp1-concentration-gpt-oss-120b-5000 | **COMPLETE** | Pulled to `results/`; s04 confirms $H=0.8789$ vs.\ certified 2000-pair $H=0.8765$ ($\Delta H=+0.0024$, within CI). Integrated into paper (Table 1, Appendix A). |
| 5575082--5575089 | exp2-dense-*-v1 (8 shards, 4 models x2) | **COMPLETE** | Per-pair phi landed for all 4 dense models. Integrated into paper (Table 3, Appendix A). |
| 5575242/5575243/5575244 (Exp6 DBRX/GPT-OSS/Gemma, superseded IDs) | Exp6 ladder extension | superseded by 5575543/5575545/5575544 | Original submission IDs; see below for the runs that actually completed. |
| 5575537--5575542 | Exp3 collectivity, all 6 ladder models (first WORKDIR-bug attempt) | superseded | WORKDIR path-doubling bug (fixed in `submit_slurm_experiment3.py`); these failed in 1-2s. |
| 5575543 (DBRX), 5575544 (Gemma-4-26B) | Exp6 ladder-completion | **COMPLETE** | Both landed on disk this session; integrated into paper (Section 5.6 causal check, Appendix B). DBRX shows a $\phi$-ranking reversal (worst of 3 orders at 50%); Gemma flagged uninterpretable (baseline bias gap $\approx 0$). |
| 5575545 (GPT-OSS-120B, Exp6 first attempt) | Exp6 ladder extension | TIMEOUT at 3:00:0x | Root cause: `slurm.time: 03:00:00` in the study config, too short. Fixed (see above); resubmitted as **5575745**. |
| 5575251, 5575252 | Exp7 proxy agreement (OLMoE, Phi-3.5-MoE) | **COMPLETE** | OLMoE $\rho=-0.024$ (layer0), $+0.235$ (layer15); Phi-3.5-MoE $\rho=-0.084$ (layer0), $+0.076$ (layer31); both null. Integrated into paper (Appendix B). |
| 5575255->5575752->5575767->**5575798**, 5575536 | Exp8 per-pair LOO (OLMoE, Phi-3.5-MoE) | **COMPLETE (both)** | Phi-3.5-MoE per-pair phi landed (job 5575536, `per_pair_phi.npy`, $n_{pairs}=50$); bootstrap CI computed and integrated (Appendix B). OLMoE's original job (5575255) only persisted summary `result.json`; after 3 resubmission attempts blocked by the maintenance-window scheduling issue (see root-cause section above), **job 5575798** (node-pinned, `--time=01:10:00`) completed cleanly: $H=0.7361$, Gini$=0.6239$, $n_{pairs}=100$, per-pair phi saved; CI computed and integrated into Appendix B ($H \in [0.692, 0.921]$). |
| 5575538 | Exp3 collectivity, Mixtral-8x7B refresh (redundant; stale July data already usable) | **CANCELLED** (mis-submitted, exceeded QOS cap) | Prior-session resubmission; not blocking (July capture already verified non-degenerate and used in paper). |
| 5575743->**5575799** | Exp3 collectivity, DBRX-132B (time-limit fix: 4h, 10 pairs) | **RUNNING** (node-pinned, 01:10:00) | Resubmitted this session; see root-cause section above for the node-pinning workaround. |
| 5575744->**5575791** | Exp3 collectivity, GPT-OSS-120B (time-limit fix: 8h, 10 pairs) | **RUNNING** (node-pinned, 01:10:00) | Resubmitted this session; same workaround. |
| 5575745->**5575800** | Exp6 ladder extension, GPT-OSS-120B (time-limit fix: 8h, 30 pairs, 60 routing-freq pairs) | **RUNNING** (node-pinned, 01:00:00) | Resubmitted this session; same workaround; this is the only remaining Exp6 gap. |

The 3 still-RUNNING jobs above were verified progressing cleanly (model
loaded, computation started, no tracebacks) as of last poll; their
`--time` is well under the original 4h/8h request due to the maintenance
window, so a `TIMEOUT` before completing the full pair/layer schedule is
possible --- re-poll with `ssh login-ice.pace.gatech.edu 'squeue -u
sghose7'` (VPN must be off) and resubmit any `TIMEOUT`'d job with a fresh
node-pin probe if so.

## Experimental status

### Captures complete & synced locally (results/), integrated into paper

All of these have result.json + per_pair_phi.npy locally (bootstrap CIs computable via `s04_bootstrap_cis.py`):

| exp dir | model | pairs | per-pair phis | CI available |
|---|---|---|---|---|
| exp1-concentration-olmoe-1b-7b-v1 | OLMoE-1B-7B | 5000 | yes | yes |
| exp1-concentration-phi3.5-moe-v1 | Phi-3.5-MoE | 5000 | yes | yes |
| exp1-concentration-mixtral-8x7b-v1 | Mixtral-8x7B | 5000 | yes | yes |
| exp1-concentration-dbrx-v1 | DBRX-132B | 5000 | yes | yes |
| exp1-concentration-gemma4-26b-v1 | Gemma-4-26B | 5000 | yes | yes |
| exp1-concentration-gpt-oss-120b-v1 | GPT-OSS-120B | 2000 | yes | yes --- VALID (old broken zero-phi capture superseded) |
| exp1-concentration-gpt-oss-120b-5000 | GPT-OSS-120B (stability replication) | 5000 | yes | yes --- confirms v1 ($\Delta H = +0.0024$) |
| exp1-concentration-gemma4-26b | Gemma (v0) | 400 | yes | yes |
| exp5-demographic-specificity-olmoe-1b-7b-v1 | OLMoE demographics | 5000 | yes | yes |
| exp2-dense-baseline-olmo-7b-v1 | OLMo-7B dense | 1800 | yes | yes (newly landed) |
| exp2-dense-baseline-phi3.5-mini-v1 | Phi-3.5-Mini dense | 4000 | yes | yes (newly landed) |
| exp2-dense-crosscheck-llama-2-7b-v1 | Llama-2-7B dense | 1800 | yes | yes (newly landed) |
| exp2-dense-crosscheck-llama-3.1-8b-v1 | Llama-3.1-8B dense | 1800 | yes | yes (newly landed) |

### Landed since last update, integrated into paper

- **Exp7 proxy-vs-exact-Shapley agreement**: DONE for 3 models (Mixtral,
  OLMoE, Phi-3.5-MoE), all null (`|\rho| <= 0.24`).
- **Dense v1 per-pair CIs**: all 4 dense models have bootstrap CIs
  (Table 3 + Appendix A).
- **GPT-OSS-120B 5000-pair stability replication**: confirms the certified
  2000-pair v1 capture.
- **Exp3 collectivity** (4 models: OLMoE, Phi-3.5-MoE, Mixtral, Gemma-4-26B):
  integrated as paper Section 5.5 + Appendix B paragraph this session.
- **Exp6 ladder extension** (DBRX, Gemma-4-26B): landed and integrated into
  Section 5.6 (causal check) + Appendix B this session. DBRX shows a
  $\phi$-ranking reversal at the 50% ablation point (worst of the 3 orders);
  Gemma-4-26B is flagged uninterpretable (baseline bias gap $\approx 0$).
- **Statistical audit**: found and fixed a math error in the exact-permutation
  floor claim (was $1/12\approx0.083$ at $n=6$; true tie-corrected minimum is
  $0.0056$), added a Monte Carlo power analysis, Cohen's-$d$ effect sizes,
  and split-half shard exact numbers.

### Still pending GPU completion (re-verify once queue clears)

- **Exp3 collectivity, DBRX + GPT-OSS-120B**: RUNNING as jobs
  **5575799**/**5575791** (node-pinned around the maintenance window; see
  root-cause section above), `--time` 01:10:00 each (down from the
  original 4h/8h request). Re-poll for completion or `TIMEOUT`.
- **Exp6 ladder extension, GPT-OSS-120B**: RUNNING as job **5575800**
  (node-pinned, `--time=01:00:00`, down from 8h). This is the only
  remaining gap in the causal-ablation evidence chain (Section 5.6
  currently reports "DONE for six models"). Re-poll for completion or
  `TIMEOUT`.
- **Exp8 per-pair LOO**: **DONE for both models.** OLMoE's per-pair
  capture landed as job **5575798** ($H=0.7361$, Gini$=0.6239$,
  $n_{pairs}=100$; CI $H \in [0.692, 0.921]$) after 3 earlier attempts
  were blocked by the maintenance-window scheduling issue. Phi-3.5-MoE's
  per-pair payload landed earlier this session (job 5575536,
  `per_pair_phi.npy`, $n=50$; CI $H \in [0.842, 0.941]$). Both integrated
  into Appendix B; all 13 model payloads referenced in the Limitations
  bootstrap-CI flag are now on disk.
- **Exp8 ladder extension (new this session)**: 3 new `dense_loo` configs
  authored to turn the ambiguous 2-model OLMoE/Phi-3.5-MoE split into a
  real ladder trend --- `study.mixtral-8x7b.lloo.yaml`, `study.dbrx.lloo.yaml`,
  `study.gpt-oss-120b.lloo.yaml`. Authoring these surfaced and fixed two
  real bugs in `discover_dense_ffn_layers`/`compute_dense_layer_contrast`
  (DBRX's `.ffn`/`transformer.blocks` naming, GPT-OSS's tuple-returning
  `.mlp`); both fixes smoke-tested against synthetic modules matching the
  real architectures exactly (see GAP-ANALYSIS.md Sec. 2). **Not yet
  submitted** --- blocked on cluster access. Gemma-4-26B deliberately
  excluded: its dual-branch FFN (`shared_expert` + `moe`, summed) can't be
  ablated by the current single-attribute LOO mechanism without silently
  under-ablating; documented as a known limitation rather than shipped
  broken.

### Missing / non-issues

- **gemma4-27b**: result dir exists but no result.json / no per-pair phi.
  Resolved as non-issue: the Gemma-4-26B row uses google/gemma-4-26B-A4B-it
  (30 layers x 8 experts = N_A=240). No action needed unless a second Gemma
  rung is wanted.

## How to submit a new run

```bash
cd moe-expt-bias/expt-bias-1
python3 scripts/submit_slurm_study.py --config configs/<study>.yaml   # from a login node
python3 scripts/submit_slurm_experiment3.py --config configs/<study>.yaml  # Exp3 collectivity (fixed WORKDIR sourcing)
python3 scripts/submit_slurm_experiment6.py --config configs/<study>.yaml  # Exp6 ladder-wide ablation
```

Resource facts: GPT-OSS-120B bf16 needs 4xH100 (~2s/pair; 2xH100 CPU-offloads
at ~25s/pair --- too slow). Large-MoE bf16 fits best in H200x2. QOS cap
960 GPU-min/job. Phi-3.5-MoE LOO (Exp8) needs 256G RAM to avoid OOM (96G is
insufficient for per-pair phi retention across 32 LOO passes).

## Kaggle data release

Per-pair phi payloads (totalling ~360MB, 8 captures) are published as Kaggle
dataset `sghose0/moe-bias-routing-shapley-perpair-phi` (see repo
README/data note). GitHub rejects >100MB files, so these payloads ship via
Kaggle; the repo carries result.json + pair_meta.json + phi.npy aggregates
instead. Dense v1 per-pair payloads (newly landed) are NOT yet on Kaggle ---
consider a follow-up dataset version once Exp3/6/8 fully land, to publish a
complete single snapshot rather than multiple incremental versions.
