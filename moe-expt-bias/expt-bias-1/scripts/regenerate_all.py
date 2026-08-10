#!/usr/bin/env python3
"""
Master Regeneration Script — MoE-Bias Study (expt-bias-1)

Runs the complete statistical pipeline from raw results to paper figures/tables.
Single entry point: python3 scripts/regenerate_all.py [--dry-run] [--skip-gpu]

Pipeline stages (in dependency order):
  1. s01_exp1_stability.py      → v0/v1 drift floors, split-half shard agreement
  2. s02_exp5_js.py             → Demographic JS distances + permutation nulls
  3. s03_h1_verdict.py          → Ladder point estimates, exact permutation p-values
  4. s04_bootstrap_cis.py       → 95% block-bootstrap CIs for all MoE rungs
  5. paper_figures_seaborn.py   → All paper figures (1-6) + Appendix tables

Outputs:
  stats_analysis/outputs/s01_exp1_stability.json
  stats_analysis/outputs/s02_exp5_js.json
  stats_analysis/outputs/s03_h1_verdict.json
  stats_analysis/outputs/s04_bootstrap_cis.json
  stats_analysis/figures/fig1_entropy_ladder.png
  stats_analysis/figures/fig2_gini_ladder.png
  stats_analysis/figures/fig3_h1_scatter.png
  stats_analysis/figures/fig4_top_fraction.png
  stats_analysis/figures/fig5_dense_vs_moe.png
  stats_analysis/figures/s02_js_distribution.png

Usage:
  python3 scripts/regenerate_all.py           # full regeneration
  python3 scripts/regenerate_all.py --dry-run # show what would run
  python3 scripts/regenerate_all.py --stage 3 # run only stage 3 and later
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "stats_analysis"
SCRIPTS = STATS / "scripts"
OUTPUTS = STATS / "outputs"
FIGURES = STATS / "figures"

STAGES = [
    {
        "name": "s01_stability",
        "script": "s01_exp1_stability.py",
        "desc": "v0/v1 drift floors + split-half shard agreement",
        "outputs": [OUTPUTS / "s01_exp1_stability.json"],
        "deps": [],
    },
    {
        "name": "s02_exp5_js",
        "script": "s02_exp5_js.py",
        "desc": "Demographic JS distances + expert-index permutation null",
        "outputs": [OUTPUTS / "s02_exp5_js.json"],
        "deps": ["s01_stability"],  # uses drift floors
    },
    {
        "name": "s03_h1_verdict",
        "script": "s03_h1_verdict.py",
        "desc": "Ladder point estimates + exact permutation p-values",
        "outputs": [OUTPUTS / "s03_h1_verdict.json"],
        "deps": ["s01_stability"],  # uses drift floors
    },
    {
        "name": "s04_bootstrap_cis",
        "script": "s04_bootstrap_cis.py",
        "desc": "95% block-bootstrap CIs for MoE ladder (5000 draws, seed 42)",
        "outputs": [OUTPUTS / "s04_bootstrap_cis.json"],
        "deps": ["s03_h1_verdict"],  # uses ladder point estimates
    },
    {
        "name": "paper_figures",
        "script": "paper_figures_seaborn.py",
        "desc": "All paper figures (1-6) + Appendix table data",
        "outputs": [
            FIGURES / "fig1_entropy_ladder.png",
            FIGURES / "fig2_gini_ladder.png",
            FIGURES / "fig3_h1_scatter.png",
            FIGURES / "fig4_top_fraction.png",
            FIGURES / "fig5_dense_vs_moe.png",
            FIGURES / "s02_js_distribution.png",
        ],
        "deps": ["s03_h1_verdict", "s04_bootstrap_cis", "s02_exp5_js"],
    },
]


def run_stage(stage: dict, python: str, dry_run: bool) -> int:
    """Run a single stage. Returns exit code."""
    script_path = SCRIPTS / stage["script"]
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    cmd = [python, str(script_path)]
    print(f"\n{'='*60}")
    print(f"Running stage: {stage['name']} — {stage['desc']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Expected outputs: {[str(o) for o in stage['outputs']]}")

    if dry_run:
        print("[DRY-RUN] Would execute above command")
        return 0

    # Ensure output dirs exist
    for out in stage["outputs"]:
        out.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def check_outputs(stage: dict) -> bool:
    """Check if all expected outputs exist."""
    return all(out.exists() for out in stage["outputs"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate all MoE-bias analysis artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--stage", type=int, default=0, help="Start from stage N (0-based, 0=all)")
    parser.add_argument("--skip-gpu", action="store_true", help="Alias for --dry-run (no GPU needed for stats)")
    parser.add_argument("--python", default="python3", help="Python executable")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = parser.parse_args()

    python = args.python
    dry_run = args.dry_run or args.skip_gpu
    force = args.force
    start_idx = max(0, min(args.stage, len(STAGES) - 1))

    print(f"MoE-Bias Study — Full Regeneration Pipeline")
    print(f"Root: {ROOT}")
    print(f"Python: {python}")
    print(f"Dry run: {dry_run}")
    print(f"Force: {force}")
    print(f"Start stage: {start_idx} ({STAGES[start_idx]['name'] if start_idx < len(STAGES) else 'N/A'})")

    # Verify scripts exist
    missing = [s for s in STAGES if not (SCRIPTS / s["script"]).exists()]
    if missing:
        print(f"[ERROR] Missing scripts: {[s['script'] for s in missing]}")
        return 1

    # Run stages
    for i, stage in enumerate(STAGES):
        if i < start_idx:
            print(f"\nSkipping stage {i}: {stage['name']} (before start index)")
            continue

        if not force and check_outputs(stage):
            print(f"\nStage {i}: {stage['name']} — outputs exist, skipping (use --force to re-run)")
            continue

        rc = run_stage(stage, python, dry_run)
        if rc != 0:
            print(f"\n[ERROR] Stage {stage['name']} failed with exit code {rc}")
            return rc

        if not dry_run and not check_outputs(stage):
            print(f"\n[WARNING] Stage {stage['name']} completed but some outputs missing")

    if not dry_run:
        print(f"\n{'='*60}")
        print(f"ALL STAGES COMPLETED SUCCESSFULLY")
        print(f"Outputs in: {OUTPUTS}")
        print(f"Figures in: {FIGURES}")
        print(f"\nTo compile paper:")
        print(f"  cd moe-expt-bias-2 && pdflatex -interaction=nonstopmode -halt-on-error moe_bias_report_acm_v2.tex")

    return 0


if __name__ == "__main__":
    sys.exit(main())
