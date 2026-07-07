# Experiment Catalog

## expt1 — Fused MoE Kernel Latency Characterization (Allgather + DeepEP)

**Goal:** Characterize fused MoE kernel latency as a function of routing distribution, token count, TP/EP parallelism, and all2all backend by calling `FusedMoEModularKernel.apply()` directly with synthetic inputs, bypassing the vLLM serving stack.

**Hardware:** PACE ICE H100 SXM5 (NVLink + InfiniBand HDR), single-node jobs

**Backends measured:**
- `allgather_reducescatter` — standard NCCL allgather dispatch (tp1-ep1, tp1-ep4, tp2-ep2)
- `deepep_low_latency` — DeepEP one-sided RDMA dispatch (tp1-ep1 only; multi-GPU blocked by NVSHMEM IBRC on PACE ICE)

**Routing distributions:** uniform, zipfian, random, skewed-2x, skewed-4x, worst-case

**Token range:** 64–65536

**Key finding:** Even over NVSwitch (best-case intra-node interconnect), allgather dispatch dominates latency at realistic token counts (>4096). At 65536 tokens, tp1-ep4 is ~16× slower than tp1-ep1.

**Results:** `expt1/all_runs.zip` — 33 result CSVs across all studies

**Code:** `expt1/src/fused_moe_kernel_study/`

**Configs:** `expt1/configs/`

**Cluster storage:** `/storage/ice1/0/2/sghose7/moe-breakdown-runs/`

---

## expt2 — NCCL Transport Condition Degradation Study

**Goal:** Quantify how much fused-MoE latency depends on fast intra-node GPU P2P communication (NVSwitch/NVLink) vs. PCIe fallback, by intentionally degrading NCCL transport and measuring latency across decode-like and prefill-like token regimes.

**Hardware:** PACE ICE H200 SXM5, single-node, 4 GPUs/node (nodes atl1-1-03-017/018; originally targeted H100 but all H100 nodes had bad multi-GPU NCCL, switched to H200)

**Shape:** Qwen3-30B-A3B (hidden=2048, inter=768, E=128, topk=8, bf16)

**Transport conditions (3 NCCL configs):**
- `nvlink_default` — NVLink/NVSwitch at full bandwidth (~900 GB/s), baseline
- `no_nvls_no_p2p` — `NCCL_NVLS_ENABLE=0 NCCL_P2P_DISABLE=1` (PCIe fallback)
- `no_nvls_no_p2p_1ch` — above + `NCCL_MAX_NCHANNELS=1` (single-channel PCIe)

**Parallel points:** tp1-ep1 (control, no inter-GPU comm), tp1-ep2, tp1-ep4, tp2-ep2

**Token range:** 1–8192 (11 values spanning decode and prefill regimes)

**Routing:** uniform (fixed for isolation)

**Total jobs:** 12 SLURM jobs (3 transport × 4 parallel points)

**Key question:** Does the slowdown ratio peak in the moderate-token regime (where allgather volume is significant but GEMM hasn't gone compute-bound) and return to ~1.0 at large tokens?

**Results:** `expt2/all_runs.zip` — 12 CSVs, jobs 5440143–5440154 (132 benchmark conditions)

**Code:** `expt2/src/fused_moe_kernel_study/`

**Configs:** `expt2/configs/study.transport-conditions.qwen3.yaml`

**Cluster storage:** `/storage/ice1/0/2/sghose7/moe-breakdown-runs/expt2/transport-conditions-qwen3/`

---

## expt2.5 — Extended Transport & Routing Study

**Goal:** Extends expt2 along three axes: (1) wider token sweep to 65k to find the compute-bound turnover; (2) NVLS×P2P ablation (4-cell factorial) to isolate which NCCL knob costs what; (3) bandwidth dose-response via NCCL_MAX_NCHANNELS 1→8; (4) routing-imbalance × transport interaction with 6 named routing modes.

**Hardware:** PACE ICE H200 SXM5, single-node, 4 GPUs/node

**Shape:** Qwen3-30B-A3B (hidden=2048, inter=768, E=128, topk=8, bf16)

**Study A — transport-extended (32 jobs, 448 conditions):**
- 8 transport conditions: nvlink_default, nvls_off, p2p_off, no_nvls_no_p2p, no_nvls_no_p2p_{8,4,2,1}ch
- Uniform routing; tokens 1–65536 (14 values)

**Study B — routing-sweep (8 jobs, 672 conditions):**
- 2 transport extremes: nvlink_default, no_nvls_no_p2p
- 6 routing modes: uniform, zipfian, random, skewed-2x, skewed-4x, worst-case
- Tokens 1–65536 (14 values)

**Total: 40 jobs, 1120 benchmark conditions**

**Results:** `expt2.5/all_runs.zip` — jobs 5442861–5442900 (all COMPLETED, 2026-06-28, nodes atl1-1-03-017/018)

**Code:** `expt2.5/src/fused_moe_kernel_study/`

**Configs:** `expt2.5/configs/`

**Cluster storage:** `~/scratch/moe-breakdown-runs/expt2.5/`

---

## expt-3-deeptest — DeepEP Low-Latency vs High-Throughput Backend Study [IN PROGRESS]

**Goal:** Compare `deepep_low_latency` (NVSHMEM one-sided RDMA) vs `deepep_high_throughput` (NCCL collective dispatch) on the same Qwen3-30B-A3B shape and H200 hardware, and characterize how routing imbalance, transport degradation, and model shape interact with each backend. Also includes a first multi-node IB bandwidth study.

**Hardware:** PACE ICE H200 SXM5, single-node (4 GPUs/node) for studies A–F; 2–4 nodes attempted for studies G/H

**Shape (primary):** Qwen3-30B-A3B (hidden=2048, inter=768, E=128, topk=8, bf16)

**Additional shapes (Study F):** DeepSeek-V3 (hidden=7168, inter=2048, E=256, topk=8), Llama-4 Scout (hidden=5120, inter=8192, E=16, topk=1)

**Studies (41 jobs, 1747 conditions):**

| Label | Backend | Routing | Transport | Jobs | Conditions | Status |
|-------|---------|---------|-----------|------|------------|--------|
| A | deepep_low_latency | uniform | nvlink_default | 3 | 33 | PENDING |
| B | deepep_high_throughput | uniform | nvlink_default | 4 | 56 | PENDING |
| C | deepep_low_latency | 4 modes | nvlink_default | 3 | 132 | PENDING |
| D | deepep_high_throughput | 4 modes | nvlink_default | 4 | 224 | PENDING |
| E | deepep_high_throughput | 4 modes | nvlink + no_nvls | 8 | 448 | PENDING |
| F | deepep_high_throughput | uniform (3 shapes) | nvlink_default | 3 | 126 | PENDING |
| G | deepep_high_throughput | uniform | nvlink_default | 4 | 56 | BLOCKED |
| H | deepep_high_throughput | 4 modes | nvlink+no_ib+no_ib4 | 12 | 672 | BLOCKED |

**Routing modes (Studies C/D/E/H):** uniform, zipfian, skewed-2x, skewed-4x

**Token range:** LL capped at 8192 (NVSHMEM buffer constraint); HT full sweep 1–65536

**Key findings:**
- DeepEP LL is empirically **slower** than HT at all token counts on single-node H200 — NVSHMEM P2P is slower than NCCL NVSwitch collectives; LL is designed for multi-node IB RDMA
- Studies G/H (multi-node) **BLOCKED** by DeepEP `LEGACY_NUM_MAX_NVL_PEERS=8` requiring exactly 8 GPUs/node; 4-GPU nodes cause topology mismatch / IPC assertion failures
- Bucket profiler time-budget decomposition (network vs compute) works for HT; LL network time = 0 (NVSHMEM not captured by NCCL events)
- GPU utilization columns added to Studies C–H via `GpuUtilSampler` background thread (SM%, memory-BW%, RAM, CPU)

**Results:** `expt-3-deeptest/all_runs.zip` — 7 completed run dirs, 89 conditions (Studies A–B initial; full results in cluster storage)

**Code:** `expt-3-deeptest/src/fused_moe_kernel_study/`

**Configs:** `expt-3-deeptest/configs/`

**Cluster storage:** `~/scratch/moe-breakdown-runs/expt-3-deeptest/`