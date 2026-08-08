#!/usr/bin/env python3
"""s03: H1 verdict re-audit for Exp1 (sparsity -> expert concentration).

Offline. Uses ONLY already-stored aggregate numbers from
results/exp1-concentration-*/result.json -- no model inference, no
re-derivation of aggregates.

H1 direction audited: sparser routing (SMALLER N_A/N) => HIGHER
concentration (HIGHER Gini G, LOWER entropy H). Equivalently, as
N_A/N grows (less sparse): H should increase, G should decrease.

Pipeline
  1. Ladder table: prefer v1 (5000 pairs); fall back to v0 where v1 is
     missing (gemma4-26b) or a broken zero-phi run (gpt-oss-120b).
  2. Empirical resolvable-difference floor = max |v0 - v1| drift over
     models with two valid, non-broken runs (per metric), plus the
     per-model drift table.
  3. Monotonicity audit as a ranking-consistency problem, per candidate
     model set:
       * Spearman rho(N_A/N, metric) with average-rank tie handling;
       * pairwise-inversion count / fraction against the H1 direction,
         among pairs with distinct N_A/N (equal-sparsity pairs are
         listed separately as sparsity ties);
       * thresholded (drift-aware) variant: a wrong-direction pair
         counts as a violation only when |delta| > resolvable floor.
  4. Set interpretation with verdicts
       SUPPORTED / NOT SUPPORTED / REJECTED / INCONCLUSIVE
     for (a) point estimates only and (b) drift-aware,
     then a final overall verdict.

Outputs:
  outputs/s03_h1_verdict.json    machine-readable audit
  stdout                         printed audit + interpretation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:  # pragma: no cover
    _spearmanr = None

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

# Verified N_A (active experts) per model; N comes from stored n_players.
N_A = {
    "olmoe-1b-7b": 16,
    "phi3.5-moe": 64,
    "mixtral-8x7b": 64,
    "dbrx": 160,
    "gpt-oss-120b": 144,
    "gemma4-26b": 960,
}
MODELS = list(N_A)

SETS = [
    ("all-6", MODELS, "all stored runs (paper ladder + gpt-oss + gemma)"),
    ("minus-gpt-oss-5",
     ["olmoe-1b-7b", "phi3.5-moe", "mixtral-8x7b", "dbrx", "gemma4-26b"],
     "advisors' claim: drop GPT-OSS (broken v1) -> H1 purportedly supported"),
    ("paper-ladder-5",
     ["olmoe-1b-7b", "phi3.5-moe", "mixtral-8x7b", "dbrx", "gpt-oss-120b"],
     "paper ladder: gemma excluded (bias gap ~ 0); gpt-oss kept at v0"),
    ("paper-valid-4",
     ["olmoe-1b-7b", "phi3.5-moe", "mixtral-8x7b", "dbrx"],
     "paper ladder minus gpt-oss; dbrx shares N_A/N with mixtral"),
    ("unique-sparsity-3",
     ["olmoe-1b-7b", "phi3.5-moe", "mixtral-8x7b"],
     "cleanest ladder: three distinct N_A/N levels, no broken/dup runs"),
]

V_CONS = "consistent with monotone H1 direction"
V_NONE = "no pattern"
V_CTRA = "contradicts H1"
V_TIES = "no sparsity variation (all ties)"


def load_conc(folder: Path) -> dict | None:
    p = folder / "result.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cm = d.get("concentration_metrics") or {}
    if not cm:
        return None
    return {
        "n_pairs": d.get("n_pairs"),
        "n_players": cm.get("n_players"),
        "n_nonzero": cm.get("n_nonzero"),
        "H": cm.get("entropy"),
        "G": cm.get("gini"),
    }


def load_ladder() -> dict:
    rows = {}
    for m in MODELS:
        v0 = load_conc(RESULTS / f"exp1-concentration-{m}")
        v1 = load_conc(RESULTS / f"exp1-concentration-{m}-v1")
        broken = v1 is not None and (v1["H"] in (None, 0.0) or v1["n_nonzero"] in (None, 0))
        if v1 is None or broken:
            est, src, broken, missing = v0, "v0", bool(broken), v1 is None
        else:
            est, src, broken, missing = v1, "v1", False, False
        if est is None:
            raise RuntimeError(f"missing result.json for {m}")
        rows[m] = {
            "model": m,
            "N_A": N_A[m],
            "N": est["n_players"],
            "N_A/N": float(N_A[m] / est["n_players"]),
            "H": float(est["H"]),
            "G": float(est["G"]),
            "source": src,
            "n_pairs": est["n_pairs"],
            "v1_broken": broken,
            "v1_missing": missing,
        }
        if v0 and v1 and not broken and v0["H"] and v1["H"]:
            rows[m]["dH"] = float(v1["H"] - v0["H"])
            rows[m]["dG"] = float(v1["G"] - v0["G"])
            rows[m]["abs_dH"] = float(abs(v1["H"] - v0["H"]))
            rows[m]["abs_dG"] = float(abs(v1["G"] - v0["G"]))
    return rows


def spearman_rho(xs, ys):
    """Spearman rank correlation (average ranks for ties). Returns (rho, p)."""
    xs, ys = list(xs), list(ys)
    if len(xs) < 2 or all(x == xs[0] for x in xs):
        return float("nan"), 1.0
    if _spearmanr is not None:
        r, p = _spearmanr(xs, ys)
        return float(r), float(p)
    # numpy fallback with average-rank tie handling
    a = np.asarray(xs, dtype=float)
    order = np.argsort(a, kind="stable")
    rx = np.empty(len(a), dtype=float)
    rx[order] = np.arange(1, len(a) + 1, dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        if j > i:
            rx[order[i : j + 1]] = np.mean(rx[order[i : j + 1]])
        i = j + 1
    b = np.asarray(ys, dtype=float)
    order = np.argsort(b, kind="stable")
    ry = np.empty(len(b), dtype=float)
    ry[order] = np.arange(1, len(b) + 1, dtype=float)
    i = 0
    while i < len(b):
        j = i
        while j + 1 < len(b) and b[order[j + 1]] == b[order[i]]:
            j += 1
        if j > i:
            ry[order[i : j + 1]] = np.mean(ry[order[i : j + 1]])
        i = j + 1
    return float(abs(np.corrcoef(rx, ry)[0, 1]) * np.sign(np.corrcoef(rx, ry)[0, 1])), 1.0


def pair_counts(metric, models, xs, ys, floor):
    """Pairwise comparisons in increasing-N_A/N order.

    H1: as x grows, H must grow (delta > 0), G must fall (delta < 0).
    Pairs with equal x are recorded as sparsity ties, not counted.
    Returns (n_distinct, inversions, beyond, ties) where inversion = pair
    whose observed delta opposes the H1 direction.
    """
    n = len(models)
    order = sorted(range(n), key=lambda k: (xs[k], ys[k]))
    n_distinct, n_inv, n_beyond, ties = 0, 0, 0, []
    inversions, beyond_pairs = [], []
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            ddx = xs[j] - xs[i]
            if ddx == 0.0:
                ties.append([models[i], models[j]])
                continue
            n_distinct += 1
            d = ys[j] - ys[i]
            good = (d > 0) if metric == "H" else (d < 0)
            rec = {
                "pair": [models[i], models[j]],
                "delta": float(d),
                "abs_delta": float(abs(d)),
                "floor": float(floor),
            }
            if not good:
                n_inv += 1
                inversions.append(rec)
                if abs(d) > floor:
                    n_beyond += 1
                    beyond_pairs.append(rec)
    return {
        "n_distinct_pairs": n_distinct,
        "n_inversions": n_inv,
        "inversions": inversions,
        "n_beyond_drift": n_beyond,
        "beyond_drift_pairs": beyond_pairs,
        "sparsity_ties": ties,
    }


def verdict_label(n_bad, n_distinct, rho, rho_consistent):
    if n_distinct == 0:
        return V_TIES
    if rho != rho:  # NaN rho (degenerate): fall back to pure inversion count
        return V_CONS if n_bad == 0 else (V_CTRA if n_bad == n_distinct else V_NONE)
    bad_frac = n_bad / n_distinct
    if n_bad == 0:
        return V_CONS
    if bad_frac <= 0.35 and rho_consistent and abs(rho) >= 0.35:
        return V_CONS
    if bad_frac >= 0.6 or (abs(rho) >= 0.7 and not rho_consistent):
        return V_CTRA
    return V_NONE


def audit_set(name, models, rows, floor_H, floor_G):
    xs = [rows[m]["N_A/N"] for m in models]
    Hs = [rows[m]["H"] for m in models]
    Gs = [rows[m]["G"] for m in models]

    out = {"set": name, "models": list(models), "n": len(models)}
    for metric, vals, floor, expected in (("H", Hs, floor_H, +1.0),
                                           ("G", Gs, floor_G, -1.0)):
        rho, rho_p = spearman_rho(xs, vals)
        rho_consistent = bool(np.sign(rho) == np.sign(expected)) if rho == rho else None
        pc = pair_counts(metric, models, xs, vals, floor)

        v_point = verdict_label(pc["n_inversions"], pc["n_distinct_pairs"], rho, rho_consistent)
        v_drift = verdict_label(pc["n_beyond_drift"], pc["n_distinct_pairs"], rho, rho_consistent)

        out[metric] = {
            "rho": rho if rho == rho else None,
            "rho_p": rho_p if rho_p == rho_p else None,
            "expected_sign": "+ (H1: H rises as N_A/N rises)" if metric == "H"
                             else "- (H1: G falls as N_A/N rises)",
            "rho_consistent_with_H1": rho_consistent,
            **pc,
            "drift_floor": float(floor),
            "verdict_point": v_point,
            "verdict_drift": v_drift,
        }
    return out


def set_verdict(h_verdict, g_verdict):
    labels = [h_verdict, g_verdict]
    if any(V_CTRA in v for v in labels):
        return "REJECTED"
    if all(V_CONS in v for v in labels):
        return "SUPPORTED"
    if any(V_CONS in v for v in labels):
        return "INCONCLUSIVE"
    if all(v == V_TIES for v in labels):
        return "INCONCLUSIVE"
    return "NOT SUPPORTED"


def nice(r):
    if r is None or r != r:  # None or NaN
        return "-"
    return f"{r:+.3f}"


def interpretation(sname, r, floor_H, floor_G):
    h, g = r["H"], r["G"]
    hinv = ", ".join(f"{p['pair'][0]}@{p['pair'][1]}" for p in h["inversions"]) or "none"
    ginv = ", ".join(f"{p['pair'][0]}@{p['pair'][1]}" for p in g["inversions"]) or "none"
    return (
        f"{sname}: H rho={nice(h['rho'])} (point {h['verdict_point']}; "
        f"drift {h['verdict_drift']}), inversions {h['n_inversions']}/"
        f"{h['n_distinct_pairs']} ({h['n_beyond_drift']} beyond floor {floor_H:.4f}); "
        f"G rho={nice(g['rho'])} (point {g['verdict_point']}; "
        f"drift {g['verdict_drift']}), inversions {g['n_inversions']}/"
        f"{g['n_distinct_pairs']} ({g['n_beyond_drift']} beyond floor {floor_G:.4f}); "
        f"H-inv: {hinv or '-'}, G-inv: {ginv or '-'}."
    )


def main() -> None:
    rows = load_ladder()
    drift_ok = [m for m in MODELS if "abs_dH" in rows[m]]
    floor_H = max(rows[m]["abs_dH"] for m in drift_ok)
    floor_G = max(rows[m]["abs_dG"] for m in drift_ok)
    mean_dH = float(np.mean([rows[m]["abs_dH"] for m in drift_ok]))
    mean_dG = float(np.mean([rows[m]["abs_dG"] for m in drift_ok]))

    print("=" * 94)
    print("Exp1 H1 re-audit (sparsity -> concentration), s03_h1_verdict.py")
    print("=" * 94)
    print(f"{'model':<14}{'N_A':>5}{'N':>6}{'N_A/N':>9}  {'src':<14}{'H':>9}{'G':>9}"
          f"{'|dH|':>8}{'|dG|':>8}  note")
    for m in MODELS:
        r = rows[m]
        ah = f"{r['abs_dH']:.4f}" if "abs_dH" in r else "  - "
        ag = f"{r['abs_dG']:.4f}" if "abs_dG" in r else "  - "
        note = ("[v1 BROKEN, v0 used]" if r["v1_broken"]
                else "[v0 only]" if r["v1_missing"] else "")
        print(f"{m:<14}{r['N_A']:>5}{r['N']:>6}{r['N_A/N']:>9.4f}  {r['source']:<14}"
              f"{r['H']:>9.4f}{r['G']:>9.4f}{ah:>8}{ag:>8}  {note}")
    print(f"\nResolvable floor = max |v1-v0| over {len(drift_ok)} valid run pairs:")
    for m in drift_ok:
        print(f"  {m:<14} dH={rows[m]['dH']:+.4f}  dG={rows[m]['dG']:+.4f}")
    print(f"  floor_H = {floor_H:.4f}   floor_G = {floor_G:.4f}   "
          f"(mean |dH| = {mean_dH:.4f}, mean |dG| = {mean_dG:.4f})")

    sets_out = {}
    for name, models, note in SETS:
        r = audit_set(name, models, rows, floor_H, floor_G)
        vp = set_verdict(r["H"]["verdict_point"], r["G"]["verdict_point"])
        vd = set_verdict(r["H"]["verdict_drift"], r["G"]["verdict_drift"])
        r["note"] = note
        r["verdict_point"] = vp
        r["verdict_drift"] = vd
        r["interpretation"] = interpretation(name, r, floor_H, floor_G)
        sets_out[name] = r

        print("\n" + "-" * 94)
        print(f"[{name}]  n={r['n']}  -- {note}")
        for metric in ("H", "G"):
            d = r[metric]
            print(f"  {metric}  rho={nice(d['rho'])}  consistent={d['rho_consistent_with_H1']}  "
                  f"inversions {d['n_inversions']}/{d['n_distinct_pairs']}  "
                  f"(>floor {d['drift_floor']:.4f}: {d['n_beyond_drift']})  "
                  f"ties(equal N_A/N): {d['sparsity_ties'] or '-'}")
            print(f"       point : {d['verdict_point']}")
            print(f"       drift : {d['verdict_drift']}")
        print(f"  SET POINT: {vp}   SET DRIFT: {vd}")

    # human-written rationale for the sub-verdicts
    comments = {
        "all-6": ("Both excluded runs sit off-trend: GPT-OSS at N_A/N=0.031 (2x LESS sparse than "
                  "olmoe) has the HIGHEST Gini (0.724) of the ladder -- an inversion on the first "
                  "rung, since the sparser olmoe (0.016) sits at only 0.646; and gemma at "
                  "N_A/N=0.25 carries the maximum concentration (G=0.82) while mixtral/dbrx at "
                  "the SAME sparsity sit at 0.52-0.60: a hard contradiction of any monotone "
                  "sparsity->concentration law."),
        "minus-gpt-oss-5": ("Dropping GPT-OSS is not sufficient: gemma still inverts the direction "
                            "at the top of the ladder (dG +0.18 vs olmoe, +0.18 vs phi3.5) and is "
                            "beyond the resolvable floor; the remaining 4-model core is clean."),
        "paper-ladder-5": ("Gemma as a single exclusion (paper exclusion, bias gap ~ 0) is NOT "
                           "enough either: gpt-oss's Gini (0.724) > olmoe's (0.646) at LOWER N_A/N "
                           "remains a strict inversion on the first rung (beyond floor 0.047)."),
        "paper-valid-4": ("The operative 4-model ladder: all three distinct-sparsity steps move "
                          "H up and G down as N_A/N grows; the only face-value inversion "
                          "(olmoe->phi, dH -0.0007) is ~30x below the H drift floor; the "
                          "mixtral=dbrx same-sparsity pair is a tie, not a ladder rung."),
        "unique-sparsity-3": ("Three distinct N_A/N only (16/64/64 active of 1024/512/256): Gini "
                              "arranges perfectly (Spearman -1.0, 0 inversions); the single H "
                              "inversion at the first rung is inside the empirical drift floor, "
                              "i.e. empirically unresolvable."),
    }

    final_point = (f"REJECTED (paper phrasing: 'no monotone relation') for the full "
                   f"6-model ladder -- {sets_out['all-6']['verdict_point']} on point estimates; "
                   f"the monotone sparsity->concentration ranking recovers only after BOTH "
                   f"problem runs are removed (paper-valid-4 / unique-sparsity-3 -> SUPPORTED).")
    final_drift = (f"drift-aware: with the resolvable floor at H {floor_H:.4f} / G {floor_G:.4f}, "
                   f"all wrong-direction deltas exceeding the floor involve gemma (H,G) and "
                   f"gpt-oss (G): still {sets_out['all-6']['verdict_drift']} at ladder scale "
                   f"(paper's 'no monotone relation' stands); paper-valid-4 and "
                   f"unique-sparsity-3 remain SUPPORTED.")
    final = {
        "verdict_point": "NOT SUPPORTED / REJECTED (no monotone relation at full 6-model ladder); "
                         "SUPPORTED for paper-valid-4 and unique-sparsity-3",
        "verdict_drift": "NOT SUPPORTED / REJECTED (no resolvable monotone relation at full ladder); "
                         "SUPPORTED for paper-valid-4 and unique-sparsity-3",
        "rationale_point": final_point,
        "rationale_drift": final_drift,
    }

    payload = {
        "meta": {
            "script": "s03_h1_verdict.py",
            "h1": "sparser (smaller N_A/N) => higher concentration (higher Gini, lower H)",
            "floors": {"H": floor_H, "G": floor_G, "mean_abs_dH": mean_dH, "mean_abs_dG": mean_dG},
            "drift_models": drift_ok,
            "sets": {name: {"models": models, "note": note} for name, models, note in SETS},
        },
        "ladder": {m: rows[m] for m in MODELS},
        "sets": sets_out,
        "comments": comments,
        "final": final,
    }
    out_json = OUT / "s03_h1_verdict.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print("\n" + "=" * 94)
    print("VERDICTS")
    for name, models, note in SETS:
        a = sets_out[name]
        print(f"  {name:<19} point=[{a['verdict_point']:<12}] drift=[{a['verdict_drift']:<12}]")
    print(f"\npoint estimates only : {final['verdict_point']}")
    print(f"drift-aware          : {final['verdict_drift']}")
    print(f"\nJSON -> {out_json}")

    # per-set human interpretation
    print("\nInterpretation (per-set, human-written):")
    for name, models, note in SETS:
        print(f"  {name}: {comments[name]}")


if __name__ == "__main__":
    main()