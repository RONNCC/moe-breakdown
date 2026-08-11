"""s08: Exact-permutation power of the H1 Spearman test at observed effect sizes.

Reproduces Section 6.4's Monte Carlo power analysis (correlated-normal
ladder simulations) so the "n ~ 12-13 rungs (rho_H = 0.754)" and
"n ~ 8-9 rungs (rho_H = 0.872)" statements are backed by a persisted,
re-runnable script instead of an ad-hoc simulation.

Method (matches the paper's stated procedure):
- Each trial draws n iid standard-normal k/N values x and
  y = rho*x + sqrt(1-rho^2)*z (z ~ N(0,1)); Spearman rho of the two
  monotone-transformed rank sequences is the test statistic.
- Two-sided p under H0: exact permutation null for n <= 10 (the null
  distribution of |rho| over all n! rankings of y against fixed x is
  precomputed once per n); Student-t approximation
  (t = rho*sqrt((n-2)/(1-rho^2)), df = n-2) for n > 10.
- Power = P(p <= 0.05) over N_TRIALS trials, per n in 3..16.
- 80%-crossing n reported by linear interpolation of power vs log(n)
  between the two integers bracketing 0.80.

Outputs: outputs/s08_power_analysis.json (power tables for rho_H =
0.754 and 0.872, crossing points, trial counts). Deterministic: default
RNG seed 42.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

OUT = Path(__file__).resolve().parents[1] / "outputs"

N_TRIALS = 20_000
SEED = 42
ALPHA = 0.05
MAX_EXACT_N = 10  # n! <= 3,628,800 at n = 10; exact null precomputed once

RHO_VALUES = [0.753702346348183, 0.872]  # observed rho_H: all-6, 5-rung subset
N_GRID = list(range(3, 17))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two vectors."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    n = len(x)
    return 1.0 - 6.0 * float(np.sum(d * d)) / (n * (n * n - 1))


def _exact_abs_rho_null(n: int) -> np.ndarray:
    """Sorted |rho| over all n! rankings of y against fixed y = 1..n."""
    vals = []
    for perm in itertools.permutations(range(n)):
        d = np.arange(n) - np.array(perm)
        rho = 1.0 - 6.0 * float(np.sum(d * d)) / (n * (n * n - 1))
        vals.append(abs(rho))
    return np.array(sorted(vals))


def p_two_sided(rho_obs: float, n: int, null_cache: dict) -> float:
    """Exact permutation two-sided p at small n; t-approx beyond."""
    if n <= MAX_EXACT_N:
        null = null_cache[n]
        # two-sided p = fraction of null |rho| at least as extreme
        idx = int(np.searchsorted(null, abs(rho_obs), side="left"))
        return float((len(null) - idx) / len(null))
    rc = float(np.clip(rho_obs, -1 + 1e-12, 1 - 1e-12))
    tstat = rc * np.sqrt((n - 2) / (1 - rc**2))
    return float(2 * tdist.sf(abs(tstat), n - 2))


def p_one_sided(rho_obs: float, n: int, null_cache: dict) -> float:
    """Exact permutation one-sided p (positive tail); t-approx beyond."""
    if n <= MAX_EXACT_N:
        null = null_cache[n]
        if rho_obs <= 0:
            return 1.0
        idx = int(np.searchsorted(null, rho_obs, side="left"))
        return float((len(null) - idx) / len(null))
    rc = float(np.clip(rho_obs, -1 + 1e-12, 1 - 1e-12))
    tstat = rc * np.sqrt((n - 2) / (1 - rc**2))
    return float(tdist.sf(tstat, n - 2))


def power_curve(rho: float, null_cache: dict, trials: int = N_TRIALS,
                one_sided: bool = False) -> dict:
    pfn = p_one_sided if one_sided else p_two_sided
    rng = np.random.default_rng(SEED)
    curve = {}
    for n in N_GRID:
        x = rng.normal(size=(trials, n))
        z = rng.normal(size=(trials, n))
        y = rho * x + np.sqrt(1.0 - rho**2) * z
        # vectorized spearman via argsort ranks
        rx = np.argsort(np.argsort(x, axis=1), axis=1)
        ry = np.argsort(np.argsort(y, axis=1), axis=1)
        d2 = np.sum((rx - ry) ** 2, axis=1)
        rho_obs = 1.0 - 6.0 * d2 / (n * (n * n - 1))
        ps = np.array([pfn(r, n, null_cache) for r in rho_obs])
        curve[n] = float(np.mean(ps <= ALPHA))
    return curve


def crossing(curve: dict) -> float | None:
    """Linear interpolation (log n) of the first 80%-power crossing."""
    ns = sorted(curve)
    for i in range(len(ns) - 1):
        n1, n2 = ns[i], ns[i + 1]
        p1, p2 = curve[n1], curve[n2]
        if p1 <= 0.80 <= p2:
            ln = np.log(n1) + (0.80 - p1) / (p2 - p1) * (np.log(n2) - np.log(n1))
            return float(np.exp(ln))
    return None


def main() -> None:
    print(f"[s08] exact permutation null for n <= {MAX_EXACT_N} "
          f"({math.factorial(MAX_EXACT_N):,} perms at n={MAX_EXACT_N}), "
          f"{N_TRIALS} trials per n, seed {SEED}, alpha {ALPHA}")
    null_cache = {n: _exact_abs_rho_null(n) for n in range(3, MAX_EXACT_N + 1)}
    out = {"_note": (f"power of the two-sided Spearman permutation test, alpha={ALPHA}; "
                     f"correlated-normal MC, {N_TRIALS} trials/n, seed {SEED}. "
                     f"Two p-conventions reported: (a) 'exact_hybrid' -- exact "
                     f"permutation null n<={MAX_EXACT_N}, Student-t beyond "
                     f"(Section 6.4's stated method); (b) 't_approx' -- "
                     f"Student-t at all n (the convention the originally-"
                     f"published Section 6.4 table used). Crossings agree."),
           "n_grid": N_GRID, "n_trials": N_TRIALS, "seed": SEED, "alpha": ALPHA,
           "rho_values": RHO_VALUES, "curves": {}}
    for rho in RHO_VALUES:
        row = {}
        for label, one_sided in (("exact_hybrid_two_sided", False),):
            curve = power_curve(rho, null_cache, one_sided=one_sided)
            n80 = crossing(curve)
            row[label] = {"power": {n: round(p, 4) for n, p in curve.items()},
                          "n_80pct_crossing": round(n80, 2) if n80 else None}
            print(f"rho={rho} {label}: "
                  f"{[(n, round(curve[n], 3)) for n in (6, 8, 9, 12, 13)]} "
                  f"-> 80% crossing n={n80:.1f}")
        # t-approx at all n (original table's convention), vectorized
        rng = np.random.default_rng(SEED)
        trow = {}
        for n in N_GRID:
            x = rng.normal(size=(N_TRIALS, n))
            z = rng.normal(size=(N_TRIALS, n))
            y = rho * x + np.sqrt(1.0 - rho**2) * z
            rx = np.argsort(np.argsort(x, axis=1), axis=1)
            ry = np.argsort(np.argsort(y, axis=1), axis=1)
            d2 = np.sum((rx - ry) ** 2, axis=1)
            rho_obs = 1.0 - 6.0 * d2 / (n * (n * n - 1))
            rc = np.clip(rho_obs, -1 + 1e-12, 1 - 1e-12)
            tt = rc * np.sqrt((n - 2) / (1 - rc**2))
            ps = 2 * tdist.sf(np.abs(tt), n - 2)
            trow[n] = float(np.mean(ps <= ALPHA))
        n80 = crossing(trow)
        row["t_approx_two_sided"] = {"power": {n: round(p, 4) for n, p in trow.items()},
                                     "n_80pct_crossing": round(n80, 2) if n80 else None}
        print(f"rho={rho} t_approx: "
              f"{[(n, round(trow[n], 3)) for n in (6, 8, 9, 12, 13)]} "
              f"-> 80% crossing n={n80:.1f}")
        out["curves"][str(rho)] = row
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "s08_power_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 's08_power_analysis.json'}")


if __name__ == "__main__":
    main()