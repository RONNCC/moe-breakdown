#!/usr/bin/env python3
"""Experiment 3 driver: Shapley interaction/synergy check (C2-lite, Sec 3.3).

Reuses an Experiment 1 (MoE concentration) config for model_id/benchmarks,
but runs `shapley.compute_shapley_interactions_for_pair` on a small prompt
subsample over a small subset of MoE layers (exact ablation is 2^K forward
passes per layer per pair, so this must stay small — see study-catalog.txt).

Usage:
  python3 scripts/run_experiment3_interactions.py --config configs/study.olmoe.concentration.yaml
  python3 scripts/run_experiment3_interactions.py --config configs/study.olmoe.concentration.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_bias_shapley.config import load_bias_study_config  # noqa: E402
from moe_bias_shapley.benchmarks import load_benchmarks  # noqa: E402
from moe_bias_shapley.hooks import discover_moe_layers  # noqa: E402
from moe_bias_shapley.modeling import DENSE_FAMILIES, load_model_and_tokenizer  # noqa: E402
from moe_bias_shapley.shapley import (  # noqa: E402
    aggregate_interaction_results,
    compute_shapley_interactions_for_pair,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_experiment3_interactions")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Experiment 3 — Shapley interaction/synergy check")
    p.add_argument("--config", required=True, help="An Experiment 1 (MoE) study config")
    p.add_argument("--out-dir", default=None, help="Override output_root/study_name/experiment3")
    p.add_argument("--max-pairs", type=int, default=20, help="Prompt subsample size (2^K forwards/pair/layer)")
    p.add_argument("--max-layers", type=int, default=2, help="Number of MoE layers to test (evenly spaced)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_bias_study_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(cfg.output_root).expanduser() / cfg.study_name / "experiment3")

    log.info("=== Experiment 3 (interactions/synergy): %s ===", cfg.study_name)
    log.info("Model: %s | family: %s | max_pairs: %d | max_layers: %d",
              cfg.model_id, cfg.model_family, args.max_pairs, args.max_layers)

    if cfg.model_family in DENSE_FAMILIES:
        raise ValueError(f"Experiment 3 requires an MoE model config, got dense family {cfg.model_family!r}")

    if args.dry_run:
        log.info("[dry-run] Would load model, load a %d-pair subsample, run interaction analysis "
                  "on %d MoE layers, and save to %s", args.max_pairs, args.max_layers, out_dir)
        return 0

    pairs = load_benchmarks(cfg.benchmarks, max_items=args.max_pairs)
    if not pairs:
        raise RuntimeError("No prompt pairs loaded — check benchmark config")

    model, tokenizer = load_model_and_tokenizer(cfg)
    device = str(next(model.parameters()).device)

    moe_layers = discover_moe_layers(model)
    if not moe_layers:
        raise RuntimeError("No MoE layers discovered — check hooks.py architecture support")

    if cfg.ablation_layer_indices:
        selected = [h for h in moe_layers if h.layer_index in cfg.ablation_layer_indices][: args.max_layers]
    else:
        n = len(moe_layers)
        k = min(args.max_layers, n)
        idxs = sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
        selected = [moe_layers[i] for i in idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_layer_summary = {}
    for handle in selected:
        log.info("Layer %d (%s): running interaction analysis over %d pairs",
                  handle.layer_index, handle.module_name, len(pairs))
        results = []
        for i, pair in enumerate(pairs):
            results.append(compute_shapley_interactions_for_pair(model, tokenizer, pair, handle, device=device))
            if (i + 1) % 5 == 0:
                log.info("  layer %d: %d/%d pairs done", handle.layer_index, i + 1, len(pairs))
        summary = aggregate_interaction_results(results)
        per_layer_summary[f"layer{handle.layer_index}"] = {
            "module_name": handle.module_name,
            **summary,
        }

    payload = {
        "study_name": cfg.study_name,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "n_pairs": len(pairs),
        "layers": per_layer_summary,
    }
    (out_dir / "experiment3_interactions.json").write_text(json.dumps(payload, indent=2))
    log.info("Done. Experiment 3 results in %s", out_dir / "experiment3_interactions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
