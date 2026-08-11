"""s06: Post-hoc split-half stability check for the four v1 ladder captures
that never got a dedicated 2-shard GPU submission (OLMoE, Phi-3.5-MoE,
Gemma-4-26B) plus the GPT-OSS-120B 5000-pair replication.

Mixtral-8x7B and DBRX-132B already have a *capture-time* split-half check:
they were submitted as two independent --num-shards 2 GPU jobs, each
loading the model once and capturing only its assigned pairs
(`pairs[shard_idx::num_shards]` in scripts/run_bias_study.py -- an
interleaved / even-odd split of the pair list, NOT a contiguous split; see
that file for the exact slicing). This script computes the analogous check
post-hoc, directly from an already-captured `per_pair_phi.npy` array, by
partitioning its rows the same way (even/odd indices) and recomputing
concentration metrics independently on each half with the exact aggregation
formula used elsewhere in this pipeline (s04_bootstrap_cis.py:
`phi.mean(axis=0)`, then normalize by sum(|.|), then metrics.py's
entropy/Gini). This is a genuine reliability check (does the point estimate
depend on which half of the captured sample you look at?) but is
methodologically distinct from Mixtral/DBRX's two-independent-GPU-job
check, so results are reported separately and the distinction is called
out in the paper.

Outputs: outputs/s06_splithalf_extension.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

# (result dir name, model label)
TARGETS = [
    ("exp1-concentration-olmoe-1b-7b-v1", "olmoe-1b-7b"),
    ("exp1-concentration-phi3.5-moe-v1", "phi3.5-moe"),
    ("exp1-concentration-gemma4-26b-v1", "gemma4-26b"),
    ("exp1-concentration-gpt-oss-120b-5000", "gpt-oss-120b-5000"),
]


def normalized_entropy(p: np.ndarray) -> float:
    n = len(p)
    if n <= 1:
        return 0.0
    mass = np.abs(p)
    total = mass.sum()
    if total <= 0:
        return 0.0
    phat = mass / total
    nz = phat[phat > 0]
    return float(-np.sum(nz * np.log(nz)) / np.log(n))


def gini(p: np.ndarray) -> float:
    mass = np.sort(np.abs(p))
    n = len(mass)
    if n == 0 or mass.sum() == 0:
        return 0.0
    cum = np.cumsum(mass)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def point_metrics(per_pair_phi_subset: np.ndarray) -> dict:
    p = per_pair_phi_subset.mean(axis=0)
    p = p / np.abs(p).sum()
    return {"entropy": normalized_entropy(p), "gini": gini(p), "n_pairs": int(per_pair_phi_subset.shape[0])}


def main() -> None:
    rows = []
    for dirname, label in TARGETS:
        f = RESULTS / dirname / "per_pair_phi.npy"
        if not f.exists():
            rows.append({"model": label, "status": "MISSING", "source": str(f)})
            continue
        phi = np.load(f)
        full = point_metrics(phi)
        shard0 = point_metrics(phi[0::2])
        shard1 = point_metrics(phi[1::2])
        d_h = abs(shard0["entropy"] - shard1["entropy"])
        d_g = abs(shard0["gini"] - shard1["gini"])
        rows.append(
            {
                "model": label,
                "status": "ok",
                "source": str(f),
                "full": full,
                "shard_even": shard0,
                "shard_odd": shard1,
                "abs_delta_entropy": d_h,
                "abs_delta_gini": d_g,
            }
        )
        print(f"{label:<20} full H={full['entropy']:.4f} G={full['gini']:.4f}  "
              f"even H={shard0['entropy']:.4f} G={shard0['gini']:.4f}  "
              f"odd H={shard1['entropy']:.4f} G={shard1['gini']:.4f}  "
              f"|dH|={d_h:.4f} |dG|={d_g:.4f}")

    OUT.joinpath("s06_splithalf_extension.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
