#!/usr/bin/env python3
"""Experiment 4 driver: independent ablation cross-check (robustness, Sec 3.4).

Loads an existing Experiment 1/2 result (player_ids.json + phi.npy, ranked by
|phi| descending) and re-derives "biased players" independently by
cumulatively zero-ablating the top-k ranked players and measuring the
resulting bias-gap disparity drop on a prompt subsample — a purely causal
check on the correlational Shapley ranking (no Shapley computation here).

Usage:
  python3 scripts/run_experiment4_ablation.py \\
      --config configs/study.olmoe.concentration.yaml \\
      --result-dir ~/scratch/moe-breakdown-bias-runs/expt-bias-1/exp1-concentration-olmoe-1b-7b
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_bias_shapley.config import load_bias_study_config  # noqa: E402
from moe_bias_shapley.benchmarks import load_benchmarks  # noqa: E402
from moe_bias_shapley.modeling import load_model_and_tokenizer  # noqa: E402
from moe_bias_shapley.shapley import compute_ablation_curve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_experiment4_ablation")


def _load_ranked_player_ids(result_dir: Path) -> list[str]:
    player_ids = json.loads((result_dir / "player_ids.json").read_text())
    phi = np.load(result_dir / "phi.npy")
    order = np.argsort(-np.abs(phi))
    return [player_ids[i] for i in order]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Experiment 4 — independent ablation cross-check")
    p.add_argument("--config", required=True, help="The study config used to produce --result-dir")
    p.add_argument("--result-dir", required=True, help="Experiment 1/2 output dir (player_ids.json + phi.npy)")
    p.add_argument("--out-dir", default=None, help="Override output dir (default: <result-dir>/experiment4)")
    p.add_argument("--max-pairs", type=int, default=30, help="Prompt subsample size for measuring bias-gap")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_bias_study_config(args.config)
    result_dir = Path(args.result_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else result_dir / "experiment4"

    log.info("=== Experiment 4 (ablation cross-check): %s ===", cfg.study_name)
    log.info("Model: %s | family: %s | max_pairs: %d | result_dir: %s",
              cfg.model_id, cfg.model_family, args.max_pairs, result_dir)

    if args.dry_run:
        log.info("[dry-run] Would load ranked players from %s, load model, load a %d-pair "
                  "subsample, compute the ablation curve, and save to %s", result_dir, args.max_pairs, out_dir)
        return 0

    ranked_player_ids = _load_ranked_player_ids(result_dir)
    log.info("Loaded %d ranked players from %s", len(ranked_player_ids), result_dir)

    pairs = load_benchmarks(cfg.benchmarks, max_items=args.max_pairs)
    if not pairs:
        raise RuntimeError("No prompt pairs loaded — check benchmark config")

    model, tokenizer = load_model_and_tokenizer(cfg)
    device = str(next(model.parameters()).device)

    curve = compute_ablation_curve(model, tokenizer, pairs, ranked_player_ids, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": cfg.study_name,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "n_pairs": len(pairs),
        "n_players": len(ranked_player_ids),
        "ablation_curve": curve,
    }
    (out_dir / "experiment4_ablation_curve.json").write_text(json.dumps(payload, indent=2))
    log.info("Done. Experiment 4 results in %s", out_dir / "experiment4_ablation_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
