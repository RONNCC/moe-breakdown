# ICE Cluster Status & Experiment Tracking (expt-bias-1)

Last updated: 2026-08-11 (VPN toggled off mid-session; cluster reachable,
queue re-polled directly via `squeue`/`scontrol`).

## Live poll, 2026-08-11 ~04:00 cluster time

`squeue -u sghose7` after VPN was disabled: **all 4 remaining jobs PENDING**
with reason `ReqNodeNotAvail, Reserved for maintenance`
(5575743 DBRX Exp3, 5575744 GPT-OSS-120B Exp3, 5575745 GPT-OSS-120B Exp6,
5575748 OLMoE Exp8 per-pair). This is **not** a QOS/resource problem: all
four fit the `coc-ice` 960 GPU-min/job cap exactly or under it, and
`sinfo -p ice-gpu -N` shows 4+ fully **idle** H100 nodes (`atl1-1-03-012-28-0`,
`-013-3-0`, `-013-8-0`, `-013-13-0`, 8xH100 each) plus 3 idle L40S nodes
throughout a 4-minute repoll window --- yet `scontrol show reservation`
reports **no reservations in the system**, and the target nodes' own
`scontrol show node` shows `State=IDLE` with no reservation flag. This
looks like a stale/cached backfill-scheduler reason string tied to a
maintenance event around node reboot time (`BootTime=2026-08-10T09:58`)
rather than a real live block, but it did not clear within the polling
window. **Not actionable by this agent** (no admin/PACE-support access);
if it persists past a few hours, file a PACE ICE support ticket citing
these job IDs and the `Reserved for maintenance` reason with no matching
`scontrol show reservation` entry.

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
| 5575255->5575748, 5575536 | Exp8 per-pair LOO (OLMoE, Phi-3.5-MoE) | Phi-3.5-MoE DONE, OLMoE pending | Phi-3.5-MoE per-pair phi landed (job 5575536, `per_pair_phi.npy`, $n_{pairs}=50$); bootstrap CI computed and integrated (Appendix B). OLMoE's original job (5575255) only persisted summary `result.json`; resubmitted with `--save-per-pair-phi` as **5575748** (queued). |
| 5575538 | Exp3 collectivity, Mixtral-8x7B refresh (redundant; stale July data already usable) | **PENDING** at last poll | Prior-session resubmission; not blocking (July capture already verified non-degenerate and used in paper). |
| **5575743** | Exp3 collectivity, DBRX-132B (time-limit fix: 4h, 10 pairs) | **PENDING** at last poll | Resubmitted this session. |
| **5575744** | Exp3 collectivity, GPT-OSS-120B (time-limit fix: 8h, 10 pairs) | **PENDING** at last poll | Resubmitted this session. |
| **5575745** | Exp6 ladder extension, GPT-OSS-120B (time-limit fix: 8h, 30 pairs, 60 routing-freq pairs) | **PENDING** at last poll | Resubmitted this session; this is the only remaining Exp6 gap. |

All 4 currently-PENDING jobs are queued (not running) due to a cluster-wide
GPU-minute backlog at last poll --- not a submission error. Re-poll with
`ssh login-ice.pace.gatech.edu 'squeue -u sghose7'` (VPN must be off).

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

- **Exp3 collectivity, DBRX + GPT-OSS-120B**: resubmitted this session as
  jobs 5575743/5575744 after fixing the `slurm.time` config bug (was 3h,
  too short; now 4h/8h) and reducing to 10 pairs/model. **PENDING** in
  queue at last poll (cluster-wide GPU-minute backlog, not a job error).
- **Exp6 ladder extension, GPT-OSS-120B**: resubmitted as job 5575745
  (8h walltime, 30 pairs, 60 routing-freq pairs). **PENDING** in queue.
  This is the only remaining gap in the causal-ablation evidence chain
  (Section 5.6 currently reports "DONE for six models").
- **Exp8 per-pair LOO** (OLMoE): no per-pair payload on disk (only summary
  `result.json`, job 5575255). Resubmitted with `--save-per-pair-phi` as
  job **5575748** (queued). Phi-3.5-MoE's per-pair payload landed this
  session (job 5575536, `per_pair_phi.npy`, $n=50$); bootstrap CI computed
  and integrated into Appendix B ($H \in [0.842, 0.941]$).

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
