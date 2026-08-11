"""s07: Bootstrap CIs for the localizability ratio + model-level dense/MoE tests.

Offline-only analysis (no model inference, no GPU). Uses the same block-bootstrap
machinery as s04_bootstrap_cis.py (resample pairs with replacement, stratified by
prompt group where pair_meta.json exists, n_boot=5000, seed=42) to give the
paper's central localizability ratio

    LR = H_dense_bar / H_MoE          (mean dense entropy / MoE rung entropy)

an uncertainty estimate per MoE rung, plus model-level hypothesis tests that
need no per-pair gap data:

  1. LR per MoE rung with a 95% percentile CI.  The dense-mean entropy is itself
     a random quantity, so every bootstrap draw recomputes the mean of the four
     dense-baseline H draws and divides by the MoE rung's H draw.  The shared
     dense mean correlates the six LRs; this is reported explicitly rather than
     hidden, and the paper's inference therefore leans on the per-rung CIs and
     the grouped tests below, not on six "independent" ratios.
  2. Matched-family LR CIs: OLMoE-vs-OLMo-7B and Phi-3.5-MoE-vs-Phi-3.5-Mini
     (independent resamples of the dense and MoE captures, ratioed per draw).
  3. Exact permutation test at the model level (n=4 dense vs n=6 MoE H point
     estimates, all C(10,4)=210 labelings enumerated): two-sided p for the
     dense-vs-MoE mean difference, the one-sided p, and the exact probability
     of complete separation (all four dense H < all six MoE H).
  4. Sign test on the six LR point estimates < 1: P(6/6 < 1 | LR ~ 1) = 0.5^6
     one-sided, treating rungs as independent units (a conservative upper bound
     on the shared-mean bias).

Outputs: outputs/s07_lr_bootstrap.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# Reuse s04's exact per-model draw sequence so per-rung H CIs reproduce
# s04_bootstrap_cis.json (consistency check emitted in the output).
from s04_bootstrap_cis import (  # noqa: E402
    N_BOOT,
    SEED,
    RESULTS,
    concentration_metrics,
    find_pair_phi,
    load_pair_meta,
)

OUT = Path(__file__).resolve().parents[1] / "outputs"

MOE_RUNS = [
    "exp1-concentration-olmoe-1b-7b-v1",
    "exp1-concentration-phi3.5-moe-v1",
    "exp1-concentration-mixtral-8x7b-v1",
    "exp1-concentration-dbrx-v1",
    "exp1-concentration-gpt-oss-120b-v1",
    "exp1-concentration-gemma4-26b-v1",
]
DENSE_RUNS = [
    "exp2-dense-baseline-olmo-7b-v1",
    "exp2-dense-baseline-phi3.5-mini-v1",
    "exp2-dense-crosscheck-llama-2-7b-v1",
    "exp2-dense-crosscheck-llama-3.1-8b-v1",
]
MATCHED = [
    # (dense_run, moe_run, label)
    ("exp2-dense-baseline-olmo-7b-v1", "exp1-concentration-olmoe-1b-7b-v1", "OLMoE vs OLMo-7B"),
    ("exp2-dense-baseline-phi3.5-mini-v1", "exp1-concentration-phi3.5-moe-v1", "Phi-3.5-MoE vs Phi-3.5-Mini"),
]


def _draw_idx(rng: np.random.Generator, n_pairs: int, pair_groups) -> np.ndarray:
    """Resample pair indices: stratified within group strata, else iid."""
    if pair_groups is None:
        return rng.integers(0, n_pairs, size=n_pairs)
    idx = np.empty(n_pairs, dtype=np.int64)
    pos = 0
    for g in range(pair_groups.max() + 1):
        gi = np.flatnonzero(pair_groups == g)
        k = len(gi)
        idx[pos : pos + k] = gi[rng.integers(0, k, size=k)]
        pos += k
    return idx


def _H(phi: np.ndarray, idx: np.ndarray) -> float:
    """Normalized entropy of the signed-mean |phi| over the resampled rows,
    identical to s04/s04_bootstrap_cis.bootstrap_ci."""
    p = phi[idx].mean(axis=0)
    return concentration_metrics(p)["entropy"]


def _load(run: str) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    exp_dir = RESULTS / run
    phi_file = find_pair_phi(exp_dir)
    if phi_file is None:
        return None, None, f"missing per_pair_phi for {run}"
    phi = np.asarray(np.load(phi_file), dtype=np.float64)
    if phi.ndim != 2:
        return None, None, f"{phi_file.name} not 2D ({phi.shape})"
    return phi, load_pair_meta(exp_dir), None


def main() -> None:
    out: dict = {"_note": "LR = mean(dense H) / MoE H; 95% percentile CIs, n_boot=5000, seed=42, block resample as s04. Dense H draws are shared across the six rungs -> LRs are correlated; matched-family rows resample each capture independently."}

    # ---- load all captures -------------------------------------------------
    dense = {}
    for run in DENSE_RUNS:
        phi, groups, err = _load(run)
        if err:
            print(f"[MISSING] dense {run}: {err}")
            dense[run] = None
            continue
        rng = np.random.default_rng(SEED)
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            draws[b] = _H(phi, _draw_idx(rng, len(phi), groups))
        dense[run] = {"phi": phi, "groups": groups, "draws": draws,
                      "point": concentration_metrics(phi.mean(axis=0))["entropy"]}
        print(f"[ok] dense {run}: H={dense[run]['point']:.4f}")

    dense_mean_point = float(np.mean([d["point"] for d in dense.values() if d]))
    dense_draws_all = np.stack([d["draws"] for d in dense.values() if d is not None])  # (4, N_BOOT)
    out["dense"] = {
        "point_hrs": {run: round(d["point"], 4) for run, d in dense.items() if d},
        "mean_point": round(dense_mean_point, 4),
        "shared_draw_matrix_shape": list(dense_draws_all.shape),
    }

    # ---- 1. per-rung LR point + CI -----------------------------------------
    lr_rows = {}
    for run in MOE_RUNS:
        phi, groups, err = _load(run)
        if err:
            print(f"[MISSING] moe {run}: {err}")
            continue
        rng = np.random.default_rng(SEED)
        h_moe, lr = np.empty(N_BOOT), np.empty(N_BOOT)
        for b in range(N_BOOT):
            h_moe[b] = _H(phi, _draw_idx(rng, len(phi), groups))
            lr[b] = dense_draws_all[:, b].mean() / h_moe[b]
        h_point = concentration_metrics(phi.mean(axis=0))["entropy"]
        lr_point = dense_mean_point / h_point
        ci = [float(np.percentile(lr, 2.5)), float(np.percentile(lr, 97.5))]
        lr_rows[run] = {
            "H_point": round(h_point, 6),
            "H_ci": [round(float(np.percentile(h_moe, 2.5)), 6), round(float(np.percentile(h_moe, 97.5)), 6)],
            "LR_point": round(lr_point, 4),
            "LR_ci": [round(ci[0], 4), round(ci[1], 4)],
            "n_pairs": int(len(phi)),
        }
        print(f"[ok] {run}: H={h_point:.4f} LR={lr_point:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}]")
    out["lr_per_rung"] = lr_rows

    # ---- 2. matched-family LR CIs ------------------------------------------
    matched = {}
    for dense_run, moe_run, label in MATCHED:
        d = dense[dense_run]
        m_phi, m_groups, err = _load(moe_run)
        if d is None or err:
            matched[label] = {"status": "MISSING"}
            continue
        d_rng = np.random.default_rng(SEED)
        m_rng = np.random.default_rng(SEED)
        ratio = np.empty(N_BOOT)
        for b in range(N_BOOT):
            h_d = _H(d["phi"], _draw_idx(d_rng, len(d["phi"]), d["groups"]))
            h_m = _H(m_phi, _draw_idx(m_rng, len(m_phi), m_groups))
            ratio[b] = h_d / h_m
        point = d["point"] / concentration_metrics(m_phi.mean(axis=0))["entropy"]
        ci = [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]
        matched[label] = {"LR_point": round(point, 4), "LR_ci": [round(ci[0], 4), round(ci[1], 4)]}
        print(f"[ok] matched {label}: LR={point:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}]")
    out["matched_family"] = matched

    # ---- 3. exact permutation test on model-level H -------------------------
    Hs = [(run, dense[run]["point"]) for run in DENSE_RUNS if dense[run]]
    for run in MOE_RUNS:
        phi, _, err = _load(run)
        if not err:
            Hs.append((run, concentration_metrics(phi.mean(axis=0))["entropy"]))
    n_dense = sum(1 for r, _ in Hs if r.startswith("exp2-"))
    h_dense = np.array([h for r, h in Hs if r.startswith("exp2-")])
    h_moe = np.array([h for r, h in Hs if not r.startswith("exp2-")])
    obs = h_dense.mean() - h_moe.mean()
    # Enumerate every labeling of the 10 model-level Hs into 4 dense / 6 MoE.
    idx = np.arange(len(Hs))
    diffs = []
    n_sep = 0
    for comb in itertools.combinations(idx, n_dense):
        comb = set(comb)
        d = np.array([Hs[i][1] for i in range(len(Hs)) if i in comb])
        m = np.array([Hs[i][1] for i in range(len(Hs)) if i not in comb])
        diffs.append(d.mean() - m.mean())
        if (d < m.min()).all():
            n_sep += 1
    diffs = np.array(diffs)
    perm = {
        "n_labelings": int(len(diffs)),
        "obs_mean_dense_minus_moe": round(float(obs), 4),
        "two_sided_p": float(np.mean(np.abs(diffs) >= abs(obs))),
        "one_sided_p_dense_gt_moe": float(np.mean(diffs >= obs)),
        "complete_separation_labelings": int(n_sep),
        "complete_separation_p_exact": float(n_sep / len(diffs)),
        "max_dense_H": round(float(h_dense.max()), 4),
        "min_moe_H": round(float(h_moe.min()), 4),
        "n_dense": int(n_dense),
        "n_moe": int(len(h_moe)),
    }
    print(json.dumps(perm, indent=2))
    out["model_level_permutation"] = perm

    # ---- 4. sign test on six LR < 1 ----------------------------------------
    lr_points = [r["LR_point"] for r in lr_rows.values()]
    n_lt = sum(1 for x in lr_points if x < 1.0)
    out["sign_test"] = {
        "n_rungs": len(lr_points),
        "n_lr_lt_1": n_lt,
        "one_sided_p_all_lt_1": float(0.5 ** len(lr_points)),
        "note": "treats rungs as iid units (upper bound; LRs share the dense mean).",
    }

    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "s07_lr_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 's07_lr_bootstrap.json'}")

    # ---- 5. bias-gap-magnitude parity (model-level, from result.json) ------
    # Run-level mean_bias_gap is the only magnitude quantity persisted;
    # per-pair gap values are NOT recoverable from per_pair_phi.npy (per-pair
    # phi sums are ~0 because sum_e mean_w = 1 per MoE layer). So the parity
    # claim is tested at the model level, exactly like the H-permutation above.
    gap_parity(run_means=_result_json_gaps(),
               out=out,
               print_=print)

def _result_json_gaps() -> dict[str, float]:
    """mean_bias_gap per run from result.json bias_scores."""
    runs = {
        "dense": [
            "exp2-dense-baseline-olmo-7b-v1",
            "exp2-dense-baseline-phi3.5-mini-v1",
            "exp2-dense-crosscheck-llama-2-7b-v1",
            "exp2-dense-crosscheck-llama-3.1-8b-v1",
        ],
        "moe": [
            "exp1-concentration-olmoe-1b-7b-v1",
            "exp1-concentration-phi3.5-moe-v1",
            "exp1-concentration-mixtral-8x7b-v1",
            "exp1-concentration-dbrx-v1",
            "exp1-concentration-gpt-oss-120b-v1",
            "exp1-concentration-gemma4-26b-v1",
        ],
    }
    out = {}
    for group, dirs in runs.items():
        out[group] = []
        for run in dirs:
            p = RESULTS / run / "result.json"
            if not p.exists():
                out.setdefault("missing", []).append(run)
                continue
            data = json.loads(p.read_text())
            try:
                gap = float(data["bias_scores"]["mean_bias_gap"])
            except (KeyError, TypeError):
                out.setdefault("missing", []).append(run)
                continue
            out[group].append({"run": run, "mean_bias_gap": round(gap, 6)})
    return out


def gap_parity(run_means: dict, out: dict, print_=print) -> None:
    """Exact permutation + Welch test on model-level mean_bias_gap."""
    from scipy.stats import t as tdist

    def vals(group: str):
        return [r["mean_bias_gap"] for r in run_means.get(group, [])]

    gd = vals("dense")
    gm_all = vals("moe")
    gm5 = gm_all[:-1]  # exclude Gemma (null-bias) for the primary comparison
    n_d, n_m = len(gd), len(gm5)

    def pooled_d(a, b):
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
        return (np.mean(a) - np.mean(b)) / sp

    def welch_p(a, b):
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(va / len(a) + vb / len(b))
        tv = (np.mean(a) - np.mean(b)) / se
        df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
        return float(2 * tdist.sf(abs(tv), df)), float(tv), float(df)

    def exact_perm(a, b):
        obs = np.mean(a) - np.mean(b)
        allv = np.array(a + b)
        count, total = 0, 0
        for comb in itertools.combinations(range(len(a) + len(b)), len(a)):
            comb = set(comb)
            d = allv[[i for i in range(len(allv)) if i in comb]].mean()
            m = allv[[i for i in range(len(allv)) if i not in comb]].mean()
            total += 1
            if abs(d - m) >= abs(obs) - 1e-12:
                count += 1
        return count, total, float(obs)

    res = {
        "n_dense": n_d,
        "n_moe_nonnull": n_m,
        "dense_mean": round(float(np.mean(gd)), 4),
        "moe_nonnull_mean": round(float(np.mean(gm5)), 4),
        "cohen_d_excl_gemma": round(float(pooled_d(gd, gm5)), 4),
    }
    c, tot, obs = exact_perm(gd, gm5)
    res["exact_perm_4v5"] = {"labelings": int(tot), "obs_diff": round(obs, 4),
                             "two_sided_p": round(c / tot, 4)}
    p, tv, df = welch_p(gd, gm5)
    res["welch_4v5"] = {"t": round(tv, 3), "df": round(df, 2), "two_sided_p": round(p, 4)}

    # signed inclusion of Gemma as a robustness variant
    if len(gm_all) == len(gm5) + 1:
        c6, tot6, obs6 = exact_perm(gd, gm_all)
        res["exact_perm_4v6_signed_incl_gemma"] = {
            "labelings": int(tot6), "obs_diff": round(obs6, 4),
            "two_sided_p": round(c6 / tot6, 4),
            "moe_all_signed_mean": round(float(np.mean(gm_all)), 4),
            "gemma_signed_mean": round(float(gm_all[-1]), 4),
        }
    out["gap_magnitude_parity"] = res
    print_(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()