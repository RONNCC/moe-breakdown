#!/usr/bin/env python3
"""Main driver for a single bias-Shapley study run.

Usage:
  python3 scripts/run_bias_study.py --config configs/study.olmoe.concentration.yaml
  python3 scripts/run_bias_study.py --config configs/study.olmoe.concentration.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_bias_shapley.config import BiasStudyConfig, load_bias_study_config  # noqa: E402
from moe_bias_shapley.benchmarks import load_benchmarks  # noqa: E402
from moe_bias_shapley.shapley import (  # noqa: E402
    compute_routing_contrast,
    compute_dense_layer_contrast,
)
from moe_bias_shapley.reporting import save_results  # noqa: E402
from moe_bias_shapley.modeling import DENSE_FAMILIES, load_model_and_tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_bias_study")


def run_study(cfg: BiasStudyConfig, out_dir: Path, dry_run: bool = False) -> None:
    log.info("=== Study: %s ===", cfg.study_name)
    log.info("Model: %s | family: %s | shapley_method: %s", cfg.model_id, cfg.model_family, cfg.shapley_method)
    log.info("Benchmarks: %s | max_prompts: %s", cfg.benchmarks, cfg.max_prompts)

    if dry_run:
        log.info("[dry-run] Would load model, load benchmarks, compute attribution, and save to %s", out_dir)
        return

    pairs = load_benchmarks(cfg.benchmarks, max_items=cfg.max_prompts)
    if not pairs:
        raise RuntimeError("No prompt pairs loaded — check benchmark config")

    model, tokenizer = load_model_and_tokenizer(cfg)
    device = next(model.parameters()).device

    demographic_key = "target" if "demographic" in cfg.study_name else None

    if cfg.model_family in DENSE_FAMILIES:
        result = compute_dense_layer_contrast(model, tokenizer, pairs, device=str(device))
    else:
        result = compute_routing_contrast(model, tokenizer, pairs, device=str(device), demographic_key=demographic_key)

    metadata = {
        "study_name": cfg.study_name,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "benchmarks": cfg.benchmarks,
        "shapley_method": cfg.shapley_method,
        "seed": cfg.seed,
    }
    save_results(out_dir, result, metadata)
    log.info("Done. Results in %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single bias-Shapley study")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None, help="Override output_root/study_name")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_bias_study_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(cfg.output_root).expanduser() / cfg.study_name)
    run_study(cfg, out_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
