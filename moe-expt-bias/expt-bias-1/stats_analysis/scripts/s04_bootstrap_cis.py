"""s04: Block-bootstrap 95% CIs for concentration metrics, per model.

Per model with stored per-pair Shapley contributions
(results/exp1-concentration-*/per_pair_phi*.npy, shape (n_pairs, n_players)),
compute point estimates + block-bootstrap 95% percentile CIs for:

  - entropy H   : normalized Shannon entropy of the |mean per-player phi|
                  distribution, H / log(n)  (matches metrics.py /
                  result.json concentration_metrics)
  - gini G      : Gini coefficient over |phi| (metrics.py formula)
  - t5          : share of |phi| mass held by the top-5 players
  - t10         : share of |phi| mass held by the top-10% of players

Method (mirrors s02_exp5_js.py conventions):
  1. Load per_pair_phi as float64; per-pair weight w_j = sum_i |phi_j,i|.
  2. p_i = |mean_j phi_j,i|, normalized to sum 1. The |.| is taken by the
     metric functions AFTER averaging, exactly as reporting.py does
     (concentration_metrics = compute_concentration_metrics(result.phi)
     on the signed mean over pairs), so point estimates coincide with
     result.json's stored values.
  3. Bootstrap: resample pairs WITH replacement (n_boot=5000, seed=42),
     recompute the signed mean and all four metrics per draw; 95%
     percentile CI. If pair_meta.json (with per-pair benchmark/group
     fields) is present, resample WITHIN each group stratum (block
     structure preserved) and note it in the output; otherwise plain iid
     resample over pairs.
  4. Cross-check point estimates against result.json's
     concentration_metrics; mismatches > 1e-3 are reported loudly.
  5. Emit MISSING entries (status="MISSING") for any exp1-concentration-*
     dir without per_pair_phi data -- never crash.

Outputs:
  outputs/s04_bootstrap_cis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

N_BOOT = 5000
SEED = 42
CONC_KEYS = ("entropy", "gini", "top_5_fraction", "top_10pct_fraction")


def normalized_entropy(p: np.ndarray) -> float:
    """Shannon entropy of the normalized |phi| distribution / log(n)."""
    n = len(p)
    if n <= 1:
        return 0.0
    total = np.abs(p).sum()
    if total <= 0:
        return 0.0
    phat = np.abs(p) / total
    nz = phat[phat > 0]
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(n))


def gini(p: np.ndarray) -> float:
    """Gini coefficient over |phi| (metrics.py formula, scale-invariant)."""
    mass = np.sort(np.abs(p))
    n = len(mass)
    if n == 0 or mass.sum() == 0:
        return 0.0
    cum = np.cumsum(mass)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def top_fraction(p: np.ndarray, top_n: int) -> float:
    total = np.abs(p).sum()
    if total <= 0:
        return 0.0
    top = np.sort(np.abs(p))[::-1][:top_n]
    return float(top.sum() / total)


def concentration_metrics(p: np.ndarray) -> dict[str, float]:
    n = len(p)
    top10pct_n = max(1, int(round(0.10 * n)))
    return {
        "entropy": normalized_entropy(p),
        "gini": gini(p),
        "t5": top_fraction(p, 5),
        "t10": top_fraction(p, top10pct_n),
    }


def load_pair_meta(exp_dir: Path) -> np.ndarray | None:
    """Per-pair group/benchmark ids for block (stratified) resampling."""
    meta_file = exp_dir / "pair_meta.json"
    if not meta_file.exists():
        return None
    d = json.loads(meta_file.read_text())
    # Accept list of dicts or dict-of-pairs; group field may be
    # "benchmark", "group", or "benchmark_group".
    if isinstance(d, list):
        pairs = d
    elif isinstance(d, dict):
        pairs = list(d.values()) if d and isinstance(next(iter(d.values())), dict) else list(d)
    groups = []
    for e in pairs:
        if isinstance(e, dict):
            groups.append(str(e.get("group", e.get("benchmark", "unknown"))))
        else:
            groups.append(str(e))
    if len(groups) != len(pairs):
        return None
    _, codes = np.unique(groups, return_inverse=True)
    return codes


def find_pair_phi(exp_dir: Path) -> Path | None:
    """Locate per-pair phi file: per_pair_phi.npy / per_pair_phi-v1.npy, and
    for non-v1 dirs also fall back to the sibling '-v1' dir's file."""
    for pat in ("per_pair_phi.npy", "per_pair_phi-v1.npy", "per_pair_phi_v1.npy"):
        f = exp_dir / pat
        if f.exists():
            return f
    globbed = sorted(exp_dir.glob("per_pair_phi*.npy"))
    if globbed:
        return globbed[0]
    if not exp_dir.name.endswith("-v1"):
        v1 = Path(str(exp_dir) + "-v1")
        if v1.exists():
            for pat in ("per_pair_phi.npy", "per_pair_phi-v1.npy"):
                f = v1 / pat
                if f.exists():
                    return f
    return None


def bootstrap_ci(phi: np.ndarray, pair_groups: np.ndarray | None) -> tuple[dict, dict]:
    """Return (point_metrics, ci_dict) for a (n_pairs, n_players) array."""
    n_pairs, n_players = phi.shape
    w = np.abs(phi).sum(axis=1)  # per-pair weight vector (documented; the
    # estimator is the signed mean of phi -- the |.| is applied by the
    # metrics functions -- so w is not needed)

    p_point = phi.mean(axis=0)  # signed mean, exactly reporting.py's result.phi
    p_point /= np.abs(p_point).sum()
    point = concentration_metrics(p_point)

    rng = np.random.default_rng(SEED)
    boot = {k: np.empty(N_BOOT) for k in point}
    for b in range(N_BOOT):
        if pair_groups is None:
            idx = rng.integers(0, n_pairs, size=n_pairs)
        else:
            # block/stratified resample: within each group stratum, resample
            # the group's pairs with replacement to the group's original size
            idx = np.empty(n_pairs, dtype=np.int64)
            pos = 0
            for g in range(pair_groups.max() + 1):
                gi = np.flatnonzero(pair_groups == g)
                k = len(gi)
                idx[pos : pos + k] = gi[rng.integers(0, k, size=k)]
                pos += k
        p = phi[idx].mean(axis=0)  # signed mean over resampled pairs
        p /= np.abs(p).sum()
        m = concentration_metrics(p)
        for k, v in m.items():
            boot[k][b] = v

    ci = {k: [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))] for k in point}
    return point, ci


def check_vs_result_json(exp_dir: Path, point: dict) -> dict:
    """Compare point estimates against stored result.json concentration_metrics."""
    res_file = exp_dir / "result.json"
    if not res_file.exists():
        return {"result_json": "absent"}
    cm = json.loads(res_file.read_text()).get("concentration_metrics", {})
    if not cm:
        return {"result_json": "no concentration_metrics"}
    tol = 1e-3
    mapping = {"entropy": "entropy", "gini": "gini", "t5": "top_5_fraction", "t10": "top_10pct_fraction"}
    mism = {}
    for k, rk in mapping.items():
        if rk not in cm or cm[rk] is None:
            continue
        diff = abs(point[k] - float(cm[rk]))
        if diff > tol:
            mism[k] = {"mine": round(point[k], 6), "result_json": round(float(cm[rk]), 6), "abs_diff": round(diff, 6)}
    if mism:
        print(f"!!! MISMATCH vs result.json in {exp_dir.name}: {json.dumps(mism)}")
    return {"result_json": "ok", "mismatches": mism} if not mism else {"result_json": "MISMATCH", "mismatches": mism}


def process_model(exp_dir: Path) -> dict:
    model = _model_key(exp_dir)
    phi_file = find_pair_phi(exp_dir)
    if phi_file is None:
        return {
            "status": "MISSING",
            "note": "no per_pair_phi*.npy on disk (check cluster results; data not synced locally)",
            "entropy": None, "gini": None, "t5": None, "t10": None, "n_pairs": None,
            "source": None,
        }

    phi = np.load(phi_file)
    if phi.ndim != 2:
        print(f"!!! {model}: {phi_file.name} has shape {phi.shape} (expected (n_pairs, n_players)); treating as MISSING")
        return {
            "status": "MISSING",
            "note": f"{phi_file.name} is not 2D (shape {phi.shape}) -- aggregated, not per-pair",
            "entropy": None, "gini": None, "t5": None, "t10": None, "n_pairs": None,
            "source": str(phi_file),
        }
    phi = np.asarray(phi, dtype=np.float64)
    n_pairs, n_players = phi.shape

    pair_groups = load_pair_meta(exp_dir)
    block_kind = "stratified-by-group" if pair_groups is not None else "iid-over-pairs"

    point, ci = bootstrap_ci(phi, pair_groups)
    check = check_vs_result_json(exp_dir, point)
    print(f"[ok] {model}: {phi_file.name} pairs={n_pairs} players={n_players} block={block_kind} "
          f"H={point['entropy']:.4f} G={point['gini']:.4f} t5={point['t5']:.4f} t10={point['t10']:.4f}")
    return {
        "status": "ok",
        "n_pairs": int(n_pairs),
        "n_players": int(n_players),
        "n_boot": N_BOOT,
        "seed": SEED,
        "block_method": block_kind,
        "source": str(phi_file),
        "entropy": ci["entropy"],
        "gini": ci["gini"],
        "t5": ci["t5"],
        "t10": ci["t10"],
        "point_estimates": {k: round(v, 6) for k, v in point.items()},
        "consistency_check": check,
    }


def _dense_model_name(exp_dir: Path) -> str:
    """Strip exp2- prefix + dense-baseline/crosscheck qualifiers for a clean model key."""
    name = exp_dir.name.removeprefix("exp2-")
    for h in ("-dense-baseline", "-dense-crosscheck"):
        name = name.replace(h, "")
    return name


def _model_key(exp_dir: Path) -> str:
    name = exp_dir.name
    if name.startswith("exp1-concentration-"):
        return name.removeprefix("exp1-concentration-")
    if name.startswith("exp8-lloo-"):
        # Namespaced distinctly from exp1-concentration keys: exp8 is the
        # same-mechanism (dense-style LOO) comparison, a different estimator
        # over the same model, and must never collide with (overwrite) the
        # ladder's routing_contrast entry for that model.
        return "lloo-" + name.removeprefix("exp8-lloo-")
    if name.startswith("exp2-"):
        return _dense_model_name(exp_dir)
    return name


def main() -> None:
    exp_dirs = (
        sorted(RESULTS.glob("exp1-concentration-*"))
        + sorted(RESULTS.glob("exp2-dense-*"))
        + sorted(RESULTS.glob("exp8-lloo-*"))
    )

    out = {"_note": "entropy is normalized H/log(n_players); CIs are 95% percentile over n_boot block-bootstrap draws (resample pairs with replacement, seed=42); MISSING = no per_pair_phi data on disk"}
    results = {_model_key(d): process_model(d) for d in exp_dirs}
    out["models"] = results
    out["summary"] = {
        "n_with_data": sum(1 for r in results.values() if r["status"] == "ok"),
        "n_missing": sum(1 for r in results.values() if r["status"] == "MISSING"),
        "missing_models": [k for k, r in results.items() if r["status"] == "MISSING"],
    }
    OUT.joinpath("s04_bootstrap_cis.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 's04_bootstrap_cis.json'}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
