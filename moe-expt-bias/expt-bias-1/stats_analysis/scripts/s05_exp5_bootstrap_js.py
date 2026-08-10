"""s05: Exp5 demographic JS divergence -- per-pair block-bootstrap CIs.

Uses the per-pair Shapley payload (per_pair_phi.npy + pair_meta.json,
downloaded from the cluster) instead of the aggregated phi_group_*.npy
vectors used by s02, so that bootstrap uncertainty can be computed by
resampling pairs WITHIN each cohort (block structure preserved).

Pair schema: pair_meta.json is a list of 5000 {"index", "benchmark",
"group"} entries, one per row of per_pair_phi.npy (shape (5000, 1024)).
Cohort (demographic group) = the pair's "group" field with " " -> "_"
normalization; the 85 unique groups match the 85 phi_group_*.npy stems
exactly. Benchmarks (stereoset/bbq) are NOT cohort identities (matches
s02, which treats the 85 group vectors as the cohort set).

JS metric (mirrors s02_exp5_js.py exactly):
  js_divergence(a, b) over |x| + 1e-12, normalized to sum 1;
  cohort vector V_c = signed mean of per-pair phi over the cohort's pairs
  (the |.| is applied inside js_divergence, as in s02).

Statistics computed:
  1. pairwise_js_mean : mean over the 85x85/2 cohort-pair JS values
     (s02's headline; paper currently reports s02 CI [0.206, 0.2311]).
  2. cohort_vs_pool_js_mean : mean over 85 cohorts of JS(V_c, pool),
     pool = mean over cohorts of |V_c| (s02 section 4).
  3. expert_perm_null_mean : s02's "expert-index permutation null"
     (JS between an observed cohort vector and another cohort vector with
     a fresh random permutation of expert/player positions).

Bootstrap (n_boot=5000, seed=42): for each draw, resample pairs within
each cohort (with replacement, to the cohort's original size), recompute
the 85 cohort vectors, and recompute stats 1-2 plus the mean of 2000
expert-permutation null draws. 95% percentile CIs over the 5000 draws.

Per-cohort table: for each of the 85 cohorts,
  - js_vs_pool        : JS(V_c, pool at point estimate), CI from resampling
                        pairs within cohort c only (pool fixed at point).
  - mean_pairwise_js  : mean JS of cohort c against the other 84 cohorts,
                        CI from the main block bootstrap.
  - exclusion counts vs the expert-permutation null mean and vs the
    paper-reported label-permutation null (0.05).

Outputs:
  outputs/s05_exp5_bootstrap_js.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

EXP5 = RESULTS / "exp5-demographic-specificity-olmoe-1b-7b-v1"
N_BOOT = 5000
SEED = 42
N_NULL = 2000
EPS = 1e-12


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """JS over normalized absolute-|phi| distributions, per the paper."""
    a = np.abs(a) + EPS
    b = np.abs(b) + EPS
    a /= a.sum()
    b /= b.sum()
    m = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_a = np.sum(a * np.log(a / m))
        kl_b = np.sum(b * np.log(b / m))
    return 0.5 * (kl_a + kl_b)


def _norm_rows(P: np.ndarray) -> np.ndarray:
    P = np.abs(P) + EPS
    P /= P.sum(axis=1, keepdims=True)
    return P


def _js_from_entropies(Es: np.ndarray, S: np.ndarray) -> np.ndarray:
    """JS = 0.5*(E_a + E_b - S) + log(2) with S = sum_k (p_a+p_b) log(p_a+p_b)."""
    return 0.5 * (Es[0] + Es[1] - S) + np.log(2.0)


def pairwise_js(V: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean pairwise JS over all cohort pairs + the upper-triangle JS vector."""
    n = V.shape[0]
    P = _norm_rows(V)
    E = (P * np.log(P)).sum(axis=1)
    I, J = np.triu_indices(n, k=1)
    S = ((P[I] + P[J]) * np.log(P[I] + P[J])).sum(axis=1)
    js = _js_from_entropies(np.stack([E[I], E[J]]), S)
    return float(js.mean()), js


def cohort_vs_pool_js(V: np.ndarray) -> tuple[float, np.ndarray]:
    pool = np.abs(V).mean(axis=0)[None, :]
    vals = np.array([js_divergence(V[i], pool[0]) for i in range(V.shape[0])])
    return float(vals.mean()), vals


def expert_perm_null_draws(V: np.ndarray, rng: np.random.Generator, m: int) -> np.ndarray:
    """s02's null: JS between an observed vector and a second vector whose
    player/expert positions are freshly permuted."""
    n, d = V.shape
    P = _norm_rows(V)
    E = (P * np.log(P)).sum(axis=1)
    ab = rng.integers(0, n, size=(m, 2))
    pa = P[ab[:, 0]]
    pb = P[ab[:, 1]]
    perm = np.argsort(rng.random((m, d)), axis=1)
    qb = np.take_along_axis(pb, perm, axis=1)
    S = ((pa + qb) * np.log(pa + qb)).sum(axis=1)
    return _js_from_entropies(np.stack([E[ab[:, 0]], E[ab[:, 1]]]), S)


def load_payload():
    phi = np.load(EXP5 / "per_pair_phi.npy")
    meta = json.loads((EXP5 / "pair_meta.json").read_text())
    assert len(meta) == phi.shape[0], f"meta {len(meta)} != rows {phi.shape[0]}"
    groups = [e["group"].replace(" ", "_") for e in meta]
    cohorts = sorted(set(groups))
    return np.asarray(phi, dtype=np.float64), cohorts, groups


def cohort_vectors(phi: np.ndarray, groups: list[str], cohorts: list[str]):
    """Signed mean of per-pair phi within each cohort + per-cohort pair counts."""
    gi = {c: i for i, c in enumerate(cohorts)}
    codes = np.array([gi[g] for g in groups])
    V = np.zeros((len(cohorts), phi.shape[1]))
    np.add.at(V, codes, phi)
    cnt = np.bincount(codes, minlength=len(cohorts))
    V /= cnt[:, None]
    return V, cnt, codes


def row_means_from_js(js: np.ndarray, I: np.ndarray, J: np.ndarray, n: int) -> np.ndarray:
    rm = np.zeros(n)
    np.add.at(rm, I, js)
    np.add.at(rm, J, js)
    return rm / (n - 1)


def main() -> None:
    phi, cohorts, groups = load_payload()
    n_pairs, n_players = phi.shape
    n_cohorts = len(cohorts)
    V_point, cnt, codes = cohort_vectors(phi, groups, cohorts)
    print(f"pairs={n_pairs} players={n_players} cohorts={n_cohorts} "
          f"pairs/cohort min={int(cnt.min())} max={int(cnt.max())}")

    # ---- point estimates ---------------------------------------------------
    pw_mean, pw_js = pairwise_js(V_point)
    pool_mean, pool_js = cohort_vs_pool_js(V_point)
    rng = np.random.default_rng(SEED)
    null_draws = expert_perm_null_draws(V_point, rng, N_NULL)
    null_mean_point = float(null_draws.mean())
    null_sd_point = float(null_draws.std(ddof=1))
    null_p95 = float(np.percentile(null_draws, 95))
    null_p99 = float(np.percentile(null_draws, 99))
    pct_pairs_gt_null_p95 = float((pw_js > null_p95).mean()) * 100.0

    # ---- main block bootstrap (resample pairs within cohort) ---------------
    rng = np.random.default_rng(SEED)
    boot_pw = np.empty(N_BOOT)
    boot_pool = np.empty(N_BOOT)
    boot_null = np.empty(N_BOOT)
    boot_row_mean = np.zeros((N_BOOT, n_cohorts))
    I, J = np.triu_indices(n_cohorts, k=1)
    for b in range(N_BOOT):
        idx = np.empty(n_pairs, dtype=np.int64)
        pos = 0
        for c in range(n_cohorts):
            gi = np.flatnonzero(codes == c)
            k = cnt[c]
            idx[pos : pos + k] = gi[rng.integers(0, k, size=k)]
            pos += k
        V = np.zeros_like(V_point)
        np.add.at(V, codes[idx], phi[idx])
        V /= cnt[:, None]
        boot_pw[b], js = pairwise_js(V)
        boot_pool[b], _ = cohort_vs_pool_js(V)
        boot_null[b] = expert_perm_null_draws(V, rng, N_NULL).mean()
        boot_row_mean[b] = row_means_from_js(js, I, J, n_cohorts)
        if (b + 1) % 1000 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT}")

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def ci_bc(theta: float, x: np.ndarray) -> list[float]:
        """Bias-corrected percentile CI: 2*theta - q(97.5), 2*theta - q(2.5).

        JS is jointly convex in its arguments, so resampling noise in the
        cohort-mean vectors inflates JS in expectation and the raw percentile
        interval is shifted up relative to the point estimate; the simple
        bias correction recentres it on theta while keeping the width.
        """
        q_lo, q_hi = np.percentile(x, [2.5, 97.5])
        return [float(2.0 * theta - q_hi), float(2.0 * theta - q_lo)]

    # ---- per-cohort CIs (within-cohort pair resampling, pool fixed) --------
    pool_point = np.abs(V_point).mean(axis=0)[None, :]
    pc_js = np.empty((n_cohorts, N_BOOT))
    rng = np.random.default_rng(SEED)
    for c in range(n_cohorts):
        gi = np.flatnonzero(codes == c)
        k = cnt[c]
        for b in range(N_BOOT):
            v = phi[gi[rng.integers(0, k, size=k)]].mean(axis=0)
            pc_js[c, b] = js_divergence(v, pool_point[0])
        if (c + 1) % 20 == 0:
            print(f"  per-cohort {c + 1}/{n_cohorts}")

    # ---- consistency check vs stored phi_group_*.npy -----------------------
    stored_diffs = []
    n_matched = 0
    for name in cohorts:
        f = EXP5 / f"phi_group_{name}.npy"
        if f.exists():
            w = np.load(f)
            stored_diffs.append(js_divergence(w, V_point[cohorts.index(name)]))
            n_matched += 1
    stored_diffs = np.array(stored_diffs)

    # ---- assemble output ----------------------------------------------------
    row_pw = row_means_from_js(pw_js, *np.triu_indices(n_cohorts, k=1), n_cohorts)
    per_cohort = {}
    n_below_exp_raw = n_above_exp_raw = n_below_label_raw = n_above_label_raw = 0
    n_below_exp_bc = n_above_exp_bc = n_below_label_bc = n_above_label_bc = 0
    for c, name in enumerate(cohorts):
        jvp = float(pool_js[c])
        jvp_raw = ci(pc_js[c])
        jvp_bc = ci_bc(jvp, pc_js[c])
        mpw = float(row_pw[c])
        mpw_raw = ci(boot_row_mean[:, c])
        mpw_bc = ci_bc(mpw, boot_row_mean[:, c])
        excl_exp = jvp_bc[1] < null_mean_point or jvp_bc[0] > null_mean_point
        excl_label = jvp_bc[1] < 0.05 or jvp_bc[0] > 0.05
        n_below_exp_raw += jvp_raw[1] < null_mean_point
        n_above_exp_raw += jvp_raw[0] > null_mean_point
        n_below_label_raw += jvp_raw[1] < 0.05
        n_above_label_raw += jvp_raw[0] > 0.05
        n_below_exp_bc += jvp_bc[1] < null_mean_point
        n_above_exp_bc += jvp_bc[0] > null_mean_point
        n_below_label_bc += jvp_bc[1] < 0.05
        n_above_label_bc += jvp_bc[0] > 0.05
        per_cohort[name] = {
            "js_vs_pool": round(jvp, 4),
            "js_vs_pool_ci_raw": [round(v, 4) for v in jvp_raw],
            "js_vs_pool_ci_bc": [round(v, 4) for v in jvp_bc],
            "mean_pairwise_js": round(mpw, 4),
            "mean_pairwise_js_ci_raw": [round(v, 4) for v in mpw_raw],
            "mean_pairwise_js_ci_bc": [round(v, 4) for v in mpw_bc],
            "excludes_expert_null_mean_bc": bool(excl_exp),
            "excludes_label_null_mean_0p05_bc": bool(excl_label),
        }

    boot_diag = {}
    for k, (theta, x) in {
        "pairwise_js_mean": (pw_mean, boot_pw),
        "cohort_vs_pool_js_mean": (pool_mean, boot_pool),
        "expert_perm_null_mean": (null_mean_point, boot_null),
    }.items():
        boot_diag[k] = {
            "bias": round(float(x.mean() - theta), 5),
            "bootstrap_se": round(float(x.std(ddof=1)), 5),
            "boot_mean": round(float(x.mean()), 5),
        }

    model = {
        "status": "ok",
        "n_pairs": int(n_pairs),
        "n_cohorts": int(n_cohorts),
        "n_players": int(n_players),
        "n_boot": N_BOOT,
        "seed": SEED,
        "block_method": "resample-pairs-within-cohort",
        "source": str(EXP5 / "per_pair_phi.npy"),
        "point_estimates": {
            "pairwise_js_mean": round(pw_mean, 4),
            "cohort_vs_pool_js_mean": round(pool_mean, 4),
            "expert_perm_null_mean": round(null_mean_point, 4),
        },
        "cis_bias_corrected": {
            "pairwise_js_mean": [round(v, 4) for v in ci_bc(pw_mean, boot_pw)],
            "cohort_vs_pool_js_mean": [round(v, 4) for v in ci_bc(pool_mean, boot_pool)],
            "expert_perm_null_mean": [round(v, 4) for v in ci_bc(null_mean_point, boot_null)],
        },
        "cis_raw_percentile": {
            "pairwise_js_mean": [round(v, 4) for v in ci(boot_pw)],
            "cohort_vs_pool_js_mean": [round(v, 4) for v in ci(boot_pool)],
            "expert_perm_null_mean": [round(v, 4) for v in ci(boot_null)],
        },
        "bootstrap_diagnostics": boot_diag,
        "expert_perm_null_details": {
            "mean": round(null_mean_point, 4),
            "sd": round(null_sd_point, 4),
            "p95": round(null_p95, 4),
            "p99": round(null_p99, 4),
            "pct_observed_pairs_gt_null_p95": round(pct_pairs_gt_null_p95, 2),
        },
        "per_cohort_summary": {
            "n_below_expert_null_mean_raw": int(n_below_exp_raw),
            "n_above_expert_null_mean_raw": int(n_above_exp_raw),
            "n_with_ci_excluding_expert_null_mean_raw": int(n_below_exp_raw + n_above_exp_raw),
            "n_below_expert_null_mean_bc": int(n_below_exp_bc),
            "n_above_expert_null_mean_bc": int(n_above_exp_bc),
            "n_with_ci_excluding_expert_null_mean_bc": int(n_below_exp_bc + n_above_exp_bc),
            "n_below_label_null_0p05_raw": int(n_below_label_raw),
            "n_above_label_null_0p05_raw": int(n_above_label_raw),
            "n_with_ci_excluding_label_null_0p05_raw": int(n_below_label_raw + n_above_label_raw),
            "n_below_label_null_0p05_bc": int(n_below_label_bc),
            "n_above_label_null_0p05_bc": int(n_above_label_bc),
            "n_with_ci_excluding_label_null_0p05_bc": int(n_below_label_bc + n_above_label_bc),
        },
        "per_cohort": per_cohort,
        "consistency_check": {
            "n_phi_group_files_matched": int(n_matched),
            "js_stored_vs_perpair_cohort_mean_median": round(float(np.median(stored_diffs)), 4),
            "js_stored_vs_perpair_cohort_mean_max": round(float(stored_diffs.max()), 4),
            "note": "stored phi_group_*.npy (Jul-7 run) and per-pair-derived cohort means (Aug-8 run) are "
                    "the same quantity but NOT identical (median JS ~0.01); s02 numbers were computed from "
                    "the stored vectors",
        },
    }
    out = {
        "_note": (
            "exp5 cohort-level JS: cohort vector = signed mean of per-pair phi over the cohort's pairs "
            "(pair_meta 'group' field, 85 groups matching phi_group_* stems); JS = Jensen-Shannon over "
            "normalized |x|+1e-12 (s02 formula). B = n_boot=5000 block-bootstrap draws (resample pairs "
            "WITH replacement within each cohort, seed=42). JS is jointly convex, so resampling noise in "
            "the cohort-mean vectors inflates JS in expectation: the raw percentile interval shifts ABOVE "
            "the point estimate (see bootstrap_diagnostics.bias). cis_bias_corrected recentres the raw "
            "percentile interval on the point estimate (2*theta - q97.5, 2*theta - q2.5) and is the "
            "headline for the paper; cis_raw_percentile is the uncorrected interval. expert_perm_null = "
            "s02's expert-index permutation null (fresh random permutation of player positions in the "
            "second vector of each null pair; null CI = bootstrap of the mean of 2000 null draws). "
            "label-permutation null (paper-reported mean 0.05) is NOT recomputable without per-prompt "
            "cohort assignment; per-cohort exclusion counts vs 0.05 use the paper-reported value. "
            "Per-cohort js_vs_pool CI: pairs of cohort c resampled, pool fixed at point estimate."
        ),
        "models": {"olmoe": model},
        "summary": {
            "n_with_data": 1,
            "n_missing": 0,
            "missing_models": [],
            "repro_stored_vectors": {
                "note": "s02 computed pairwise mean 0.2213 from stored phi_group_*.npy (Jul-7 run) with "
                        "cohort-bootstrap CI [0.206, 0.2311] (s02_exp5_js.json); s05 numbers come from the "
                        "Aug-8 per-pair payload and differ slightly",
            },
        },
    }
    OUT.joinpath("s05_exp5_bootstrap_js.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 's05_exp5_bootstrap_js.json'}")


if __name__ == "__main__":
    main()
