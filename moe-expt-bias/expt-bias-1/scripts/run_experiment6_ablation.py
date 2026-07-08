#!/usr/bin/env python3
"""Experiment 6 driver: ladder-wide ablation with proxy-top-k, frequency-top-k, random-k.

Extends Exp4 by:
  1. Adding a high-routing-frequency control condition (experts ranked by mean
     activation frequency, not Shapley phi) to test whether proxy-ranked ablation
     achieves better debiasing efficiency than a naive traffic-based baseline.
  2. Tracking perplexity and selectivity (Δbias / Δperplexity) at each step,
     allowing the "selectivity collapse" claim to be quantified directly.
  3. Computing routing_freq on-the-fly if routing_freq.npy is absent from the
     result directory (the original Exp1 runs did not save it).

Usage:
  python3 scripts/run_experiment6_ablation.py \\
      --config configs/study.olmoe.concentration.yaml \\
      --result-dir ~/scratch/.../exp1-concentration-olmoe-1b-7b \\
      [--max-pairs 60] [--routing-freq-pairs 200]
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
from moe_bias_shapley.shapley import compute_ablation_curve, compute_routing_contrast  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_experiment6_ablation")


def _load_ranked_players(result_dir: Path, model, tokenizer, pairs_for_freq, device: str) -> tuple[list[str], dict[str, list[str]]]:
    """Load phi-ranked player list and build control orderings.

    Returns (ranked_phi_ids, controls) where controls has keys 'random' and
    optionally 'high_routing' (skipped for dense models without routing_freq).
    """
    player_ids: list[str] = json.loads((result_dir / "player_ids.json").read_text())
    phi = np.load(result_dir / "phi.npy")
    phi_order = np.argsort(-np.abs(phi))
    ranked_phi_ids = [player_ids[i] for i in phi_order]

    controls: dict[str, list[str]] = {}

    # Random control.
    rng = np.random.default_rng(42)
    random_order = rng.permutation(len(player_ids))
    controls["random"] = [player_ids[i] for i in random_order]

    # High-routing-frequency control: use routing_freq.npy if present, else compute.
    routing_freq_path = result_dir / "routing_freq.npy"
    if routing_freq_path.exists():
        log.info("Loading routing_freq from %s", routing_freq_path)
        routing_freq = np.load(routing_freq_path)
    elif pairs_for_freq:
        log.info("routing_freq.npy not found — computing from %d pairs (forward-pass only)", len(pairs_for_freq))
        try:
            freq_result = compute_routing_contrast(model, tokenizer, pairs_for_freq, device=device)
            routing_freq = freq_result.routing_freq
            if routing_freq is not None:
                np.save(routing_freq_path, routing_freq)
                log.info("Saved routing_freq to %s", routing_freq_path)
            else:
                log.warning("compute_routing_contrast returned routing_freq=None — skipping high_routing control")
                routing_freq = None
        except ValueError as exc:
            log.warning("Could not compute routing_freq (%s) — skipping high_routing control", exc)
            routing_freq = None
    else:
        routing_freq = None

    if routing_freq is not None and len(routing_freq) == len(player_ids):
        freq_order = np.argsort(-routing_freq)
        controls["high_routing"] = [player_ids[i] for i in freq_order]
        log.info("high_routing control: top expert by freq is %s (freq=%.4f)",
                  player_ids[freq_order[0]], routing_freq[freq_order[0]])
    elif routing_freq is not None:
        log.warning("routing_freq length %d != player_ids length %d — skipping high_routing", len(routing_freq), len(player_ids))

    return ranked_phi_ids, controls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Experiment 6 — ladder-wide ablation (phi/freq/random)")
    p.add_argument("--config", required=True, help="Study config used to produce --result-dir")
    p.add_argument("--result-dir", required=True, help="Exp1/2 output dir (player_ids.json + phi.npy)")
    p.add_argument("--out-dir", default=None, help="Override output dir (default: <result-dir>/experiment6)")
    p.add_argument("--max-pairs", type=int, default=60, help="Prompt pairs for ablation curve (default: 60)")
    p.add_argument("--routing-freq-pairs", type=int, default=200,
                   help="Pairs for routing-freq computation if routing_freq.npy absent (default: 200)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_bias_study_config(args.config)
    result_dir = Path(args.result_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else result_dir / "experiment6"

    log.info("=== Experiment 6 (ladder-wide ablation + controls): %s ===", cfg.study_name)
    log.info("Model: %s | family: %s | max_pairs: %d | result_dir: %s",
              cfg.model_id, cfg.model_family, args.max_pairs, result_dir)

    if args.dry_run:
        log.info("[dry-run] Would load ranked players from %s, compute routing_freq if absent, "
                  "load model, load %d-pair subsample, compute 3-condition ablation curve, save to %s",
                  result_dir, args.max_pairs, out_dir)
        return 0

    # Load model once; reuse for routing_freq computation and ablation.
    model, tokenizer = load_model_and_tokenizer(cfg)
    device = str(next(model.parameters()).device)

    # Pairs for routing_freq computation (can be more than ablation pairs for better freq estimate).
    freq_pairs = load_benchmarks(cfg.benchmarks, max_items=args.routing_freq_pairs)
    ablation_pairs = load_benchmarks(cfg.benchmarks, max_items=args.max_pairs)
    if not ablation_pairs:
        raise RuntimeError("No prompt pairs loaded — check benchmark config")

    ranked_phi_ids, controls = _load_ranked_players(result_dir, model, tokenizer, freq_pairs, device)
    log.info("Players: %d phi-ranked | controls: %s", len(ranked_phi_ids), list(controls))

    curves = compute_ablation_curve(model, tokenizer, ablation_pairs, ranked_phi_ids,
                                    device=device, controls=controls)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": cfg.study_name,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "n_pairs": len(ablation_pairs),
        "n_players": len(ranked_phi_ids),
        "routing_freq_pairs": len(freq_pairs),
        "ablation_curves": curves,
    }
    out_path = out_dir / "experiment6_ablation_curve.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Done. Experiment 6 results in %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
