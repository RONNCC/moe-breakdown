"""s01: Exp1 point-estimate stability audit (offline, no model inference).

Uses only stored result.json files to quantify how much the headline H/Gini
point estimates move with (a) prompt-pair count (v0 400 vs v1 5000) and
(b) between-split halves (shard0 vs shard1 when 2 shards were stored).

Outputs:
  outputs/s01_exp1_stability.json   machine-readable audit
  stdout table for quick review

Also FLAGS broken/zero-phi runs (entropy==0 or n_nonzero==0), which are
silent-failure captures (known failure mode: GPT-OSS hook capture).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

MODELS = [
    "olmoe-1b-7b",
    "phi3.5-moe",
    "mixtral-8x7b",
    "dbrx",
    "gpt-oss-120b",
    "gemma4-26b",
]


def load_conc(folder: Path) -> dict | None:
    p = folder / "result.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    cm = d.get("concentration_metrics") or {}
    return {
        "n_pairs": d.get("n_pairs"),
        "n_players": cm.get("n_players"),
        "n_nonzero": cm.get("n_nonzero"),
        "entropy": cm.get("entropy"),
        "gini": cm.get("gini"),
        "top5": cm.get("top_5_fraction"),
        "mean_bias_gap": (d.get("bias_scores") or {}).get("mean_bias_gap"),
    }


def main() -> None:
    rows: list[dict] = []
    for m in MODELS:
        v0 = load_conc(RESULTS / f"exp1-concentration-{m}")
        v1 = load_conc(RESULTS / f"exp1-concentration-{m}-v1")
        shards = []
        base = RESULTS / f"exp1-concentration-{m}-v1"
        for tag in ("result_shard0of2.json", "result_shard1of2.json"):
            sp = base / tag
            if sp.exists():
                d = json.loads(sp.read_text())
                cm = d.get("concentration_metrics") or {}
                shards.append(
                    {
                        "tag": tag.replace("result_", "").replace(".json", ""),
                        "n_pairs": d.get("n_pairs"),
                        "entropy": cm.get("entropy"),
                        "gini": cm.get("gini"),
                    }
                )
        v2 = load_conc(RESULTS / f"exp1-concentration-{m}-5000")
        row = {
            "model": m,
            "v0": v0,
            "v1": v1,
            "v2": v2,
            "shards": shards,
            "broken": (v1 is not None and (v1["entropy"] == 0 or v1["n_nonzero"] == 0)),
        }
        if v0 and v1 and not row["broken"] and v0["entropy"] and v1["entropy"]:
            row["dh"] = v1["entropy"] - v0["entropy"]
            row["dg"] = v1["gini"] - v0["gini"] if v0["gini"] and v1["gini"] else None
        if v1 and v2 and v1["entropy"] and v2["entropy"]:
            row["dh_v1v2"] = v2["entropy"] - v1["entropy"]
            row["dg_v1v2"] = (v2["gini"] - v1["gini"]) if v1["gini"] and v2["gini"] else None
        rows.append(row)

    OUT.joinpath("s01_exp1_stability.json").write_text(json.dumps(rows, indent=2))

    print(f"{'model':<14} {'set':<8} {'pairs':>6} {'H':>8} {'G':>7} {'nonzero':>8}  note")
    for r in rows:
        for label, d, broken in (("v0", r["v0"], False), ("v1", r["v1"], r["broken"])):
            if d is None:
                print(f"{r['model']:<14} {label:<8} {'--':>6}")
                continue
            note = "  <-- BROKEN (all-zero phi)" if broken else ""
            print(
                f"{r['model']:<14} {label:<8} {d['n_pairs']:>6} "
                f"{d['entropy']:>8.4f} {d['gini']:>8.4f} {d['n_nonzero']:>8}  {note}"
            )
            if not broken and label == "v1":
                for s in r["shards"]:
                    print(
                        f"{'':<14} {s['tag']:<8} {s['n_pairs']:>6} "
                        f"{s['entropy']:>8.4f} {s['gini']:>8.4f}"
                    )
                if r.get("dh") is not None:
                    print(f"    -> dH(v1-v0) = {r['dh']:+.4f}   dG = {r['dg']:+.4f}")
    print("\nBroken runs detected:", [r["model"] for r in rows if r["broken"]] or "none")


if __name__ == "__main__":
    main()