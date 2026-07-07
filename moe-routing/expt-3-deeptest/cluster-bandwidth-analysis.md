# PACE ICE Cluster Bandwidth Analysis: MoE Dispatch at Multi-Node Scale

## 1. Cluster Hardware

PACE ICE H200 nodes: 8x H200 SXM5 per node (full HGX H200), 64-core Intel Xeon, 2 TB RAM.
IB fabric: Mellanox ConnectX-6, **HDR 100 Gb/s per port**, confirmed from sysfs. MLNX OFED 25.10.

| Link type | Effective unidirectional BW |
|-----------|---------------------------|
| NVLink 4.0 (H200 SXM5, 18 links) | **450 GB/s** |
| IB HDR per port | **~11 GB/s** (after encoding overhead) |

NVLink-to-IB ratio: **~41x**.

## 2. Dispatch Volume Math (Qwen3-30B-A3B, bfloat16)

Per-token dispatch payload: `hidden=2048 x 2B x topk=8 = 32 KB`.

**Cross-node fraction at ep=16 (2 nodes x 8 GPUs):** Each token selects topk=8 experts from 128
total, mapped uniformly across 16 ranks (8 experts/rank). Of the 8 selected expert-rank destinations,
on average 8 x (8 remote ranks / 16 total ranks) = **4 go cross-node**. Cross-node fraction = 4/8 = **50%**.

| Tokens per rank (T) | Bytes/rank dispatched | Cross-node bytes/rank (50%) | IB transit time @11 GB/s |
|---------------------|----------------------|----------------------------|--------------------------|
| 1 | 32 KB | 16 KB | **1.5 us** |
| 64 | 2 MB | 1 MB | **91 us** |
| 512 | 16 MB | 8 MB | **727 us** |
| 4,096 | 128 MB | 64 MB | **5.8 ms** |
| 8,192 | 256 MB | 128 MB | **11.6 ms** |

NVLink (ep=8, intra-node): same dispatch volume at 450 GB/s, **41x shorter** (e.g. T=8192: ~285 us).

## 3. Bandwidth Regime Analysis

**Inflection point** where IB transit time exceeds setup overhead (~50 us for QP bring-up, NVSHMEM
sync, kernel launch):

```
T_inflect = (setup_overhead x IB_BW) / (32 KB x cross_node_fraction)
          = (50 us x 11 GB/s) / (32 KB x 0.50)
          = 550 KB / 16 KB ~ 34 tokens per rank
```

Therefore: T=1 is **latency-bound** (setup dominates); T>=64 is **bandwidth-bound** (IB transit
dominates). Our sweep [1, 64, 512, 4096, 8192] tokens/rank enters the bandwidth-bound regime at
the *second* data point -- the entire range above T=1 directly measures the IB constraint.

## 4. Expected Experimental Signal

**ep=8 (NVLink only):** 450 GB/s ceiling is never hit at these token counts. Expect near-flat latency
across the sweep.

**ep=16 (IB cross-node):** Should match ep=8 only at T=1 (latency-bound). From T=64 onward, latency
diverges and scales linearly with T. The slope ratio ep=16/ep=8 should approach **41x** in the pure
bandwidth limit. This means the IB bottleneck is observable from our *second sweep point* -- a strong
experimental claim that multi-node MoE dispatch is fundamentally bandwidth-constrained even at modest
batch sizes (64 tokens/rank = 1024 total tokens at ep=16).

**`no_ib` transport condition (TCP fallback):** Disabling IB forces NCCL to fall back to TCP/IP over
Ethernet at ~1 GB/s effective throughput -- **11x worse** than HDR IB. At T>=64 this renders
cross-node dispatch effectively unusable, providing a dramatic lower-bound demonstration of the
bandwidth bottleneck.

**Hot-expert load imbalance:** If one expert receives 2-4x average tokens, the hosting node receives
proportionally more IB ingress, creating head-of-line blocking visible in 99th-percentile latency at
high T.

## 5. Structural Scaling Argument

General law: as EP grows beyond one node, `cross_node_fraction -> 1` and `dispatch_volume = T x
hidden x topk x 2B`. The IB bottleneck is **structural, not incidental** -- it applies to any MoE
where `hidden x topk > ~4 KB` and EP spans multiple nodes. DeepSeek-V3 (114 KB/token, EP=32+)
hits this from T=1; our Qwen3 experiment reproduces the same constraint at T>=64.
