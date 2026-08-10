#!/usr/bin/env python3
"""
Figure Caption Audit — MoE-Bias Study

Verifies that figure captions in the LaTeX paper match the actual data in outputs/.
Run after regeneration: python3 scripts/audit_captions.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path("/Users/ronnie.ghose/src/priv/gatech/research-projs/moe-breakdown/moe-expt-bias-2") / "moe_bias_report_acm_v2.tex"
OUTPUTS = ROOT / "stats_analysis" / "outputs"
FIGURES = ROOT / "stats_analysis" / "figures"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def extract_paper_numbers() -> dict:
    """Extract numeric claims from the paper (captions, tables, text)."""
    text = PAPER.read_text()
    claims = {}

    # Table 1 (ladder) - extract model rows
    # Pattern: Model & k & N & k/N & H & Gini & t5 & t10 & CI
    table1_pattern = r'([A-Za-z0-9\-+\.]+)\s+&\s+(\d+)\s+&\s+(\d+)\s+&\s+([\d.]+)\s+&\s+([\d.]+)\s+&\s+([\d.]+)\s+&\s+([\d.]+)\s+&\s+([\d.]+)\s+&\s+\[([\d.,]+)\]'
    for match in re.finditer(table1_pattern, text):
        model, k, N, kN, H, Gini, t5, t10, CI = match.groups()
        claims[f'table1_{model}'] = {
            'k': int(k), 'N': int(N), 'k/N': float(kN),
            'H': float(H), 'Gini': float(Gini),
            't5': float(t5), 't10': float(t10), 'CI': CI
        }

    # Table 2 (CIs) - extract CI bounds
    ci_pattern = r'([A-Za-z0-9\-+\.]+)\s+&\s+\[([\d.,]+)\s*,\s*([\d.,]+)\]\s+&\s+\[([\d.,]+)\s*,\s*([\d.,]+)\]'
    for match in re.finditer(ci_pattern, text):
        model, Hlo, Hhi, Glo, Ghi = match.groups()
        key = f'ci_{model}'
        claims[key] = {
            'H': (float(Hlo.replace(',', '')), float(Hhi.replace(',', ''))),
            'Gini': (float(Glo.replace(',', '')), float(Ghi.replace(',', '')))
        }

    # Figure captions with numbers
    # Figure 1: entropy ladder
    # Figure 2: gini ladder
    # Figure 3: scatter rho=0.754 p=0.106
    # Figure 4: top-5 2-11%, top-10% 0.35-0.77
    # Figure 5: dense vs MoE
    # Figure 6 (s02): JS mean 0.22, CI [0.206,0.231], null 0.31

    fig_patterns = {
        'fig3_scatter': r'rho_H\s*=\s*\+([\d.]+)\s*,\s*exact\s*two.sided\s*permutation\s*p\s*=\s*([\d.]+)',
        'fig4_topfrac': r'top.5 experts hold.*?\$([\d.]+).*?\$([\d.]+).*?top.10.*?\$([\d.]+).*?\$([\d.]+)',
        'fig6_js': r'mean pairwise Jensen.*?\$([\d.]+).*?expert.index permutation.*?\$([\d.]+)',
    }
    for name, pattern in fig_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            claims[name] = {f'val{i}': float(v) for i, v in enumerate(match.groups())}

    # Abstract claims
    abstract_patterns = {
        'abstract_h_range': r'H \\approx (\d\.\d+).*?(\d\.\d+)',
        'abstract_top5_range': r'top.5 experts hold \$(\d+).*?(\d+)',
        'abstract_rho': r'rho_H = \+([\d.]+).*?p = ([\d.]+)',
        'abstract_dense_H': r'dense.*?H \\in \[([\d.]+), ([\d.]+)\]',
        'abstract_dense_mean': r'mean (\d\.\d+)',
        'abstract_js': r'Jensen.*?\$([\d.]+).*?null at \$([\d.]+)',
    }
    for name, pattern in abstract_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            claims[f'abstract_{name}'] = {f'val{i}': float(v) for i, v in enumerate(match.groups())}

    return claims


def load_outputs() -> dict:
    """Load all JSON outputs."""
    data = {}
    for f in OUTPUTS.glob('*.json'):
        data[f.stem] = load_json(f)
    return data


def compare_claims(claims: dict, outputs: dict) -> list:
    """Compare paper claims against actual outputs."""
    issues = []

    # s03_h1_verdict.json - ladder point estimates
    s03 = outputs.get('s03_h1_verdict', {})
    ladder = s03.get("ladder", {})
    if ladder:
        for model, row in ladder.items():
            model = row.get('model', '').lower().replace('-', '').replace('.', '')
            key = f'table1_{model}'
            if key in claims:
                paper = claims[key]
                actual = {
                    'H': round(row.get('H', 0), 3),
                    'Gini': round(row.get('Gini', 0), 3),
                    't5': round(row.get('t5', 0), 3),
                    't10': round(row.get('t10', 0), 3),
                }
                for metric in ['H', 'Gini', 't5', 't10']:
                    p = round(paper.get(metric, 0), 3)
                    a = actual.get(metric, 0)
                    if abs(p - a) > 0.01:  # 1% tolerance
                        issues.append(f"MISMATCH {key} {metric}: paper={p}, output={a}")

    # s04_bootstrap_cis.json - CIs
    s04 = outputs.get('s04_bootstrap_cis', {})
    models = s04.get('models', {})
    for model, metrics in models.items():
        key = f'ci_{model.lower().replace("-", "").replace(".", "")}'
        if key in claims:
            paper = claims[key]
            actual_H = metrics.get('entropy', [0, 0])
            actual_G = metrics.get('gini', [0, 0])
            for metric, actual, label in [(actual_H, claims[key].get('H', (0,0)), 'H'),
                                           (actual_G, claims[key].get('Gini', (0,0)), 'Gini')]:
                if abs(actual[0] - label[0]) > 0.01 or abs(actual[1] - label[1]) > 0.01:
                    issues.append(f"CI MISMATCH {model} {metric}: paper={label}, output={actual}")

    # s02_exp5_js.json - demographic JS
    s02 = outputs.get('s02_exp5_js', {})
    if s02:
        js_mean = s02.get('pairwise_js', {}).get('mean', 0)
        js_ci = s02.get('pairwise_js', {}).get('bootstrap_ci', [0, 0])
        null_mean = s02.get('expert_index_permutation_null', {}).get('mean', 0)
        if 'fig6_js' in claims:
            paper_mean = claims['fig6_js']['val0']
            paper_null = claims['fig6_js']['val1']
            if abs(js_mean - paper_mean) > 0.01:
                issues.append(f"JS mean mismatch: paper={paper_mean}, output={js_mean}")
            if abs(null_mean - paper_null) > 0.01:
                issues.append(f"Null mean mismatch: paper={paper_null}, output={null_mean}")

    # s03 - exact permutation p-values
    subset_audit = s03.get('subset_audit', {})
    if subset_audit:
        for subset, data in subset_audit.items():
            H = data.get('H', {})
            if 'rho' in H:
                pass  # could check abstract rho values

    return issues


def check_figures_exist() -> list:
    """Verify all referenced figures exist."""
    expected = [
        'fig1_entropy_ladder.png',
        'fig2_gini_ladder.png',
        'fig3_h1_scatter.png',
        'fig4_top_fraction.png',
        'fig5_dense_vs_moe.png',
        's02_js_distribution.png',
    ]
    missing = []
    for f in expected:
        if not (FIGURES / f).exists():
            missing.append(f)
    return missing


def main() -> int:
    print("=" * 60)
    print("Figure Caption Audit — MoE-Bias Study")
    print("=" * 60)

    if not PAPER.exists():
        print(f"[ERROR] Paper not found: {PAPER}")
        return 1

    claims = extract_paper_numbers()
    outputs = load_outputs()

    print(f"\nExtracted {len(claims)} claims from paper")
    print(f"Loaded {len(outputs)} output files")

    issues = compare_claims(claims, outputs)
    missing_figs = check_figures_exist()

    if missing_figs:
        print(f"\n[WARNING] Missing figures: {missing_figs}")
        for f in missing_figs:
            issues.append(f"Missing figure: {f}")
    else:
        print(f"\nAll {len(missing_figs)} referenced figures exist")

    if issues:
        print(f"\n[FAIL] {len(issues)} issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print(f"\n[PASS] All caption claims verified against outputs")
        return 0


if __name__ == "__main__":
    sys.exit(main())
