#!/usr/bin/env python3
"""Audit Appendix A table rows in the paper against authoritative JSONs.

Per the caption:
  - MoE H, Gini, t5, t10%  -> v1 result.json concentration_metrics
  - CIs -> s04_bootstrap_cis.json v1 model blocks
  - |dH|, |dG| -> s01_exp1_stability.json absolute drifts
  - Dense rows -> v0 result.json concentration_metrics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent.parent / "moe-expt-bias-2" / "moe_bias_report_acm_v2.tex"
OUTS = ROOT / "stats_analysis" / "outputs"
RESULTS = ROOT / "results"

s01 = {r["model"]: r for r in json.loads((OUTS / "s01_exp1_stability.json").read_text())}
s04 = json.loads((OUTS / "s04_bootstrap_cis.json").read_text())

paper = PAPER.read_text()
LB = "\\begin{table*}"
LC = "\\caption{Full measurement table."
LL = "\\label{tab:appendix-full}"
i0 = paper.index(LL)
i_tbl = paper.rindex(LB, 0, i0)
i_cap = paper.rindex(LC, 0, i0)
block = paper[i_tbl:i_cap]

rows = []
for line in block.split("\n"):
    line = line.strip()
    if " & " in line and line.endswith("\\\\"):
        rows.append(line)
print(f"Parsed {len(rows)} data rows from appendix table")

KEYMAP = {
    "OLMoE-1B-7B": "olmoe-1b-7b",
    "Phi-3.5-MoE": "phi3.5-moe",
    "Mixtral-8x7B": "mixtral-8x7b",
    "DBRX-132B": "dbrx",
    "GPT-OSS-120B": "gpt-oss-120b",
    "Gemma-4-26B": "gemma4-26b",
    "OLMo-7B (dense)": "exp2-dense-baseline-olmo-7b",
    "Phi-3.5-Mini (dense)": "exp2-dense-baseline-phi3.5-mini",
    "Llama-2-7B (dense)": "exp2-dense-crosscheck-llama-2-7b",
    "Llama-3.1-8B (dense)": "exp2-dense-crosscheck-llama-3.1-8b",
}

RES_DIR = {
    "olmoe-1b-7b": "exp1-concentration-olmoe-1b-7b-v1",
    "phi3.5-moe": "exp1-concentration-phi3.5-moe-v1",
    "mixtral-8x7b": "exp1-concentration-mixtral-8x7b-v1",
    "dbrx": "exp1-concentration-dbrx-v1",
    "gpt-oss-120b": "exp1-concentration-gpt-oss-120b-v1",
    "gemma4-26b": "exp1-concentration-gemma4-26b-v1",
    "exp2-dense-baseline-olmo-7b": "exp2-dense-baseline-olmo-7b",
    "exp2-dense-baseline-phi3.5-mini": "exp2-dense-baseline-phi3.5-mini",
    "exp2-dense-crosscheck-llama-2-7b": "exp2-dense-crosscheck-llama-2-7b",
    "exp2-dense-crosscheck-llama-3.1-8b": "exp2-dense-crosscheck-llama-3.1-8b",
}

DENSE_KEYS = [
    "exp2-dense-baseline-olmo-7b",
    "exp2-dense-baseline-phi3.5-mini",
    "exp2-dense-crosscheck-llama-2-7b",
    "exp2-dense-crosscheck-llama-3.1-8b",
]


def num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_concentration(dirname: str) -> dict:
    f = RESULTS / dirname / "result.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("concentration_metrics", {})


mismatches = []
for r in rows:
    cells = [c.strip() for c in r.rstrip("\\\\").split(" & ")]
    model = cells[0].replace("$^{\\ddagger}$", "").replace("$^{*}$", "").strip()
    key = KEYMAP.get(model)
    if not key:
        continue
    H, G, t5, t10, adH, adG = (num(c) for c in (cells[6], cells[7], cells[10], cells[11], cells[12], cells[13]))
    ciH, ciG = cells[8].strip(), cells[9].strip()
    res_dir = RES_DIR.get(key)
    if not res_dir:
        continue

    cm = load_concentration(res_dir)
    if not cm:
        mismatches.append((model, "result.json missing concentration_metrics"))
        continue

    # Compare H, G, t5, t10% against result.json (full precision)
    checks = {
        "H": abs(H - cm.get("entropy", float('nan'))) < 1e-12,
        "G": abs(G - cm.get("gini", float('nan'))) < 1e-12,
        "t5": abs(t5 - cm.get("top_5_fraction", float('nan'))) < 1e-12,
        "t10": abs(t10 - cm.get("top_10pct_fraction", float('nan'))) < 1e-12,
    }
    bad = [m for m, ok in checks.items() if not ok]
    if bad:
        mismatches.append((model, f"metrics:{','.join(bad)} (paper={dict(H=H,G=G,t5=t5,t10=t10)} vs json={cm})"))

    # CIs: MoE rows only; dense rows paper=---, s04=MISSING
    if key in s04["models"] and key not in DENSE_KEYS:
        s04k = (key + "-v1") if (key + "-v1") in s04["models"] else key
        ci = s04["models"].get(s04k, {})
        entropy = ci.get("entropy")
        gini = ci.get("gini")
        if entropy is not None and gini is not None:
            ciH_exp = "[" + ", ".join(str(x) for x in entropy) + "]"
            ciG_exp = "[" + ", ".join(str(x) for x in gini) + "]"
            if ciH != ciH_exp:
                mismatches.append((model, f"ciH paper={ciH} json={ciH_exp}"))
            if ciG != ciG_exp:
                mismatches.append((model, f"ciG paper={ciG} json={ciG_exp}"))

    # Drifts: compare against s01 v0->v1 absolute drifts
    if key in s01:
        if abs(adH - abs(s01[key]["dh"])) > 1e-12:
            mismatches.append((model, f"|dH| paper={adH} s01={abs(s01[key]['dh'])}"))
        if abs(adG - abs(s01[key]["dg"])) > 1e-12:
            mismatches.append((model, f"|dG| paper={adG} s01={abs(s01[key]['dg'])}"))

if mismatches:
    print(f"\n[FAIL] {len(mismatches)} issues:")
    for m in mismatches:
        print(" -", *m)
    sys.exit(1)
print("\n[PASS] ALL APPENDIX ROWS MATCH AUTHORITATIVE SOURCES EXACTLY")
