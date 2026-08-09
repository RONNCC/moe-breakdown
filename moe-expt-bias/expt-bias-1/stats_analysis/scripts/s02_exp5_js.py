"""s02: Exp5 demographic JS-divergence statistical analysis (offline).

Uses only stored per-cohort attribution vectors phi_group_*.npy (85 groups).

Computes:
  1. Full pairwise JS-divergence matrix between demographic cohorts
     (paper's formula: JS over normalized absolute |phi| vectors).
  2. Point estimate + bootstrap CI of the mean pairwise JS (blocked over
     cohorts, resample-with-replacement of groups).
  3. "Expert-index permutation null" (marginal-preserving): JS between a
     pair of observed vectors if expert identities were permuted out.
     Reported as mean +/- SD + 95th percentile, plus the fraction of
     observed pairs exceeding it.
  4. Cohort-vs-model-pool divergence (demographic specificity / C3).
  5. Per-cohort concentration spread (mean +/- SD over 85 groups).

NOTE on the stored label-permutation null (paper: D_JS,null ~ 0.05 +- 0.01):
that null needs per-prompt cohort assignments (not on disk), so it is echoed
in the output as reported but not recomputed here.

Outputs:
  outputs/s02_exp5_js.json
  figures/s02_js_distribution.png   (violin/pairwise vs null)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)

EXP5 = RESULTS / "exp5-demographic-specificity-olmoe-1b-7b-v1"
RNG = np.random.default_rng(42)
N_BOOT = 2000
N_PAIR = 2000
N_PERM = 100


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """JS over normalized absolute-|phi| distributions, per the paper."""
    a = np.abs(a) + 1e-12
    b = np.abs(b) + 1e-12
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.sum(a * np.log(a / m))
        kl_b = np.sum(b * np.log(b / m))
    return 0.5 * (kl_a + kl_b)


def load_cohorts() -> tuple[list[str], np.ndarray]:
    names = sorted(p.stem.replace("phi_group_", "") for p in EXP5.glob("phi_group_*.npy"))
    vecs = np.stack([np.load(EXP5 / f"phi_group_{n}.npy") for n in names])
    return names, vecs


def main() -> None:
    names, V = load_cohorts()
    n = len(names)
    print(f"cohorts loaded: {n}, phi dim {V.shape[1]}")

    # 1) pairwise JS matrix
    pm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pm[i, j] = pm[j, i] = js_divergence(V[i], V[j])
    ij = np.triu_indices(n, k=1)
    pair_js = pm[ij]
    mean_js = float(pair_js.mean())
    sd_js = float(pair_js.std(ddof=1))

    # 2) block bootstrap over cohorts (resample groups, recompute pairwise mean)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.choice(n, size=n, replace=True)
        mu = 0.0
        cnt = 0.0
        for x in range(n):
            for y in range(x + 1, n):
                mu += js_divergence(V[idx[x]], V[idx[y]])
                cnt += 1.0
        boot[b] = mu / cnt
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])

    # 3) expert-index permutation null, per pair we draw the pair's two
    #    observed vectors and permute the expert identity of the second's
    null_draws = np.empty(N_PAIR)
    for k in range(N_PAIR):
        p1, p2 = RNG.integers(0, n, size=2)
        v2 = V[p2][RNG.permutation(V.shape[1])]  # permute expert identity of v2
        null_draws[k] = js_divergence(V[p1], v2)
    null_mean, null_sd = float(null_draws.mean()), float(null_draws.std(ddof=1))
    null_95 = float(np.percentile(null_draws, 95))
    null_99 = float(np.percentile(null_draws, 99))
    frac_gt_95 = float((pair_js > null_95).mean()) * 100.0

    # 4) cohort vs model-pool (global |phi|)
    global_v = np.abs(V).mean(axis=0)[None, :]
    pool_js = np.array([js_divergence(V[i], global_v[0]) for i in range(n)])
    pool_mean, pool_sd = float(pool_js.mean()), float(pool_js.std(ddof=1))

    # 5) per-cohort concentration spread from stored per-group metrics
    d = json.loads((EXP5 / "result.json").read_text())
    pgm = d["per_group_concentration_metrics"]
    grp_ent = np.array([v["entropy"] for v in pgm.values()])
    grp_gin = np.array([v["gini"] for v in pgm.values()])

    out = {
        "n_cohorts": n,
        "phi_dim": int(V.shape[1]),
        "pairwise_js": {
            "n_pairs": int(len(pair_js)),
            "mean": round(mean_js, 4),
            "sd_across_pairs": round(sd_js, 4),
            "min": round(float(pair_js.min()), 4),
            "max": round(float(pair_js.max()), 4),
            "bootstrap95_ci_of_mean": [round(float(lo_ci), 4), round(float(hi_ci), 4)],
        },
        "expert_index_permutation_null": {
            "mean": round(null_mean, 4),
            "sd": round(null_sd, 4),
            "p95": round(null_95, 4),
            "p99": round(null_99, 4),
            "pct_observed_pairs_gt_null_p95": round(frac_gt_95, 2),
        },
        "reported_label_permutation_null": {"mean": 0.05, "sd": 0.01, "NOTE": "paper-as-reported; needs per-prompt assignment, not on disk"},
        "cohort_vs_pool_js": {
            "mean": round(pool_js.mean(), 4),
            "sd": round(pool_js.std(ddof=1), 4),
        },
        "per_cohort_concentration": {
            "entropy_mean": round(float(grp_ent.mean()), 4),
            "entropy_sd": round(float(grp_ent.std(ddof=1)), 4),
            "gini_mean": round(float(grp_gin.mean()), 4),
        },
        "top_divergent_pairs": sorted(
            [(names[i], names[j], round(float(pm[i, j]), 3)) for i, j in zip(*np.unravel_index(np.argsort(pair_js)[-10:], pm.shape))],
            key=lambda t: -t[2],
        ),
    }
    OUT.joinpath("s02_exp5_js.json").write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pair_js, bins=40, alpha=0.65, label=f"observed pairs (n={len(pair_js)})")
    ax.hist(null_draws, bins=40, alpha=0.65, label="expert-permutation null")
    ax.axvline(mean_js, color="C0", ls="--", label=f"mean {mean_js:.3f}")
    ax.axvline(null_95, color="C1", ls=":", label=f"null p95 {null_95:.3f}")
    ax.axvline(0.05, color="k", ls="--", label="reported label-null mean 0.05")
    ax.set_xlabel("Jensen-Shannon divergence")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Exp5: demographic JS divergence vs. permutation nulls")
    fig.tight_layout()
    fig.savefig(FIG / "s02_js_distribution.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIG / 's02_js_distribution.png'}")


if __name__ == "__main__":
    main()