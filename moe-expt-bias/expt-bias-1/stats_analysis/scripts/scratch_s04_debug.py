"""scratch_s04_debug.py: diagnose why s04 point estimates disagree with
result.json concentration_metrics.

Reproduces for two models (DBRX-v1, OLMoE-v1):

  (a) s04's current estimator:   p_i = mean_j |phi_j,i|      (mean-of-abs)
  (b) result.json estimator:     p_i = |mean_j phi_j,i|     (abs-of-mean;
      exactly compute_concentration_metrics(result.phi) in reporting.py)
  (c) permutation sanity check over player columns
  (d) per-layer normalization variants

All metrics use the exact s04/metrics.py formulas (identical implementations).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"


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
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(n))


def gini(p: np.ndarray) -> float:
    mass = np.sort(np.abs(p))
    n = len(mass)
    if n == 0 or mass.sum() == 0:
        return 0.0
    cum = np.cumsum(mass)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def top_fraction(p: np.ndarray, top_n: int) -> float:
    mass = np.abs(p)
    total = mass.sum()
    if total <= 0:
        return 0.0
    top = np.sort(mass)[::-1][:top_n]
    return float(top.sum() / total)


def metrics(p: np.ndarray) -> dict:
    n = len(p)
    return {
        "entropy": normalized_entropy(p),
        "gini": gini(p),
        "t5": top_fraction(p, 5),
        "t10": top_fraction(p, max(1, int(round(0.10 * n)))),
    }


def per_layer_equal_weight(p_layer: np.ndarray) -> np.ndarray:
    """p_layer: (n_layers, n_experts) of |phi| per layer. Normalize each layer
    row to sum 1, average rows equally, renormalize to sum 1."""
    row_sums = p_layer.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = p_layer / row_sums
    agg = norm.mean(axis=0)
    s = agg.sum()
    return agg / s if s > 0 else agg


def fmt(m: dict) -> str:
    return "  ".join(f"{k}={v:.6f}" for k, v in m.items())


def diagnose(model: str) -> None:
    rd = RESULTS / f"exp1-concentration-{model}"
    pp = np.load(rd / "per_pair_phi.npy").astype(np.float64)
    phi = np.load(rd / "phi.npy").astype(np.float64)
    player_ids = json.loads((rd / "player_ids.json").read_text())
    cm = json.loads((rd / "result.json").read_text())["concentration_metrics"]
    cm_sel = {"entropy": cm["entropy"], "gini": cm["gini"], "t5": cm["top_5_fraction"], "t10": cm["top_10pct_fraction"]}

    print(f"\n=== {model}  per_pair {pp.shape} players={len(player_ids)} pairs={pp.shape[0]} ===")
    assert len(player_ids) == pp.shape[1], "player_ids length != n_players"
    assert np.allclose(phi, pp.mean(axis=0), atol=1e-12), "phi.npy != mean(per_pair_phi)"

    # (a) s04 current: mean-of-abs
    a = metrics(np.abs(pp).mean(axis=0))
    # (b) result.json path: abs-of-mean (exact reporting.py pipeline)
    b = metrics(phi)

    # (c) permutation: metrics are permutation-invariant; ordering only matters
    #     for player labels, so confirm identity + invariant under a random perm
    rng = np.random.default_rng(0)
    perm = rng.permutation(pp.shape[1])
    c = metrics(phi[perm])
    perm_ok = all(abs(c[k] - b[k]) < 1e-12 for k in b)

    # (d) per-layer variants
    n_layers, n_experts = None, None
    # infer (layers, experts) from player_ids label pattern "layer<l>-expert<e>"
    import re
    m_l = re.match(r"^layer(\d+)-expert(\d+)$", player_ids[0])
    if m_l:
        layer_ids = sorted({int(p.split("-expert")[0].replace("layer", "")) for p in player_ids})
        max_experts = pp.shape[1] // len(layer_ids)
        # column order is layer-major (layer0 exp0..expE-1, layer1 ...)
        P = pp.reshape((pp.shape[0], len(layer_ids), max_experts))
        d1 = metrics(per_layer_equal_weight(np.abs(P).mean(axis=0)))
        gap_row = np.abs(P).mean(axis=0)  # (L,E)
        d2 = metrics(gap_row)  # global (== (a)); sanity
        # per-layer normalize, then weighted by per-layer |phi| mass (should equal (b)? no: equals global abs-of-mean| . no: equals (a) since weights cancel)
        row_mass = gap_row.sum(axis=1, keepdims=True)
        row_mass[row_mass == 0] = 1.0
        d3 = metrics((gap_row / row_mass).mean(axis=0))
    else:
        d1 = d2 = d3 = None

    print(f"[a] s04 (mean-of-abs)      {fmt(a)}   <- s04 point estimate")
    print(f"[b] result.json path       {fmt(b)}   <- reporting.py exact")
    print(f"[b] permuted cols          {fmt(c)}   perm_ok={perm_ok}")
    print(f"[d1] per-layer-equal-wt    {fmt(d1)}")
    for name, m_ in (("a", a), ("b", b)):
        diffs = {k: abs(m_[k] - cm_sel[k]) for k in cm_sel}
        print(f"    diff vs result.json [{name}]: " + "  ".join(f"{k}={v:.6f}" for k, v in diffs.items()))
    print(f"result.json                {fmt(cm_sel)}")


if __name__ == "__main__":
    for m in ["dbrx-v1", "olmoe-1b-7b-v1"]:
        diagnose(m)