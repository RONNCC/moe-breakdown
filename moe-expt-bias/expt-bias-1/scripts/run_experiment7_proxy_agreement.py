#!/usr/bin/env python3
"""Experiment 7: proxy-vs-causal agreement test (Sec 3.4 / reviewer response).

On a small prompt subsample and a tractable MoE model (Mixtral, K=2/8 per token,
2^8=256 max coalitions), compares:
  - routing_contrast Shapley ranking (proxy, fast: 2 fwd passes / pair)
  - exact active-coalition Shapley ranking (causal, slow: 2^K fwd passes / pair / layer)

Computes Spearman rank correlation (ρ) between |phi_rc| and |phi_exact| restricted
to the per-pair active expert set. Reports per-layer mean ρ ± std across the subsample.

If ρ is high (≥ 0.7), routing_contrast is a trustworthy proxy for the causal ranking
and the concentration claims from Exp1 hold without requiring exact computation on
all 6 models. If ρ is low, the proxy is unreliable and claims must be narrowed.

Best suited for Mixtral (K=2/8 → 2^8=256 coalitions) where exact Shapley is cheap.
Avoid on OLMoE (top-1/64: up to 2^10=1024 coalitions with max_k=10 truncation) or
GPT-OSS (top-4/128: very slow even with truncation).

Usage:
  python3 scripts/run_experiment7_proxy_agreement.py --config configs/study.mixtral-8x7b.concentration.yaml
  python3 scripts/run_experiment7_proxy_agreement.py --config configs/study.mixtral-8x7b.concentration.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_bias_shapley.benchmarks import load_benchmarks  # noqa: E402
from moe_bias_shapley.config import load_bias_study_config  # noqa: E402
from moe_bias_shapley.hooks import (  # noqa: E402
    MoeLayerHandle,
    attach_router_hooks,
    detach_hooks,
    discover_moe_layers,
)
from moe_bias_shapley.modeling import load_model_and_tokenizer  # noqa: E402
from moe_bias_shapley.shapley import (  # noqa: E402
    _bias_gap_from_logits,
    _sequence_logprob,
    compute_exact_shapley_for_pair,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_experiment7_proxy_agreement")


def _per_pair_routing_contrast(
    model: Any,
    tokenizer: Any,
    pair: Any,
    moe_layers: List[MoeLayerHandle],
    device: str,
) -> Tuple[float, Dict[int, np.ndarray]]:
    """Single-pair routing_contrast attribution: returns (bias_gap, {layer_idx: phi_array}).

    phi_array[e] = (mean_w_stereo[e] - mean_w_anti[e]) * bias_gap for expert e.
    """
    state = attach_router_hooks(model)

    logp_s = _sequence_logprob(model, tokenizer, pair.stereo, device)
    captured_s = {li: {k: v.clone() for k, v in d.items()} for li, d in state.captured.items()}

    logp_a = _sequence_logprob(model, tokenizer, pair.anti_stereo, device)
    captured_a = {li: {k: v.clone() for k, v in d.items()} for li, d in state.captured.items()}

    detach_hooks(state)
    bias_gap = _bias_gap_from_logits(logp_s, logp_a)

    per_layer: Dict[int, np.ndarray] = {}
    for handle in moe_layers:
        li = handle.layer_index
        if li not in captured_s or li not in captured_a:
            continue
        n_e = handle.num_experts
        if n_e <= 0:
            continue

        def _mean_w(captured: Dict) -> np.ndarray:
            idx = captured[li]["topk_idx"].reshape(-1, captured[li]["topk_idx"].shape[-1])
            w = captured[li]["topk_weight"].reshape(-1, captured[li]["topk_weight"].shape[-1])
            idx_np = idx.cpu().numpy()
            w_np = w.float().cpu().numpy()
            mean = np.zeros(n_e, dtype=np.float64)
            cnt = np.zeros(n_e, dtype=np.float64)
            for tok in range(idx_np.shape[0]):
                for k in range(idx_np.shape[1]):
                    e = int(idx_np[tok, k])
                    mean[e] += w_np[tok, k]
                    cnt[e] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, mean / np.maximum(cnt, 1), 0.0)
            return mean

        mean_s = _mean_w(captured_s)
        mean_a = _mean_w(captured_a)
        per_layer[li] = (mean_s - mean_a) * bias_gap

    return bias_gap, per_layer


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between x and y (same length)."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    return float(1.0 - 6.0 * d2 / (n * (n * n - 1)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Experiment 7 — proxy-vs-causal agreement test")
    p.add_argument("--config", required=True, help="An Experiment 1 MoE study config (Mixtral recommended)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-pairs", type=int, default=20,
                   help="Prompt subsample size (2^K forward passes per pair per layer)")
    p.add_argument("--max-layers", type=int, default=2,
                   help="MoE layers to test: first + last (default 2)")
    p.add_argument("--max-k", type=int, default=8,
                   help="Max active expert set size for exact Shapley (default 8)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_bias_study_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg.output_root).expanduser() / cfg.study_name / "experiment7"
    )

    log.info("=== Experiment 7 (proxy-vs-causal agreement): %s ===", cfg.study_name)
    log.info("Model: %s | max_pairs: %d | max_layers: %d | max_k: %d",
              cfg.model_id, args.max_pairs, args.max_layers, args.max_k)

    if args.dry_run:
        log.info("[dry-run] Would load model, %d pairs, run routing_contrast + exact Shapley "
                  "on %d MoE layers, compute Spearman ρ, save to %s",
                  args.max_pairs, args.max_layers, out_dir)
        return 0

    pairs = load_benchmarks(cfg.benchmarks, max_items=args.max_pairs)
    if not pairs:
        raise RuntimeError("No prompt pairs loaded")

    model, tokenizer = load_model_and_tokenizer(cfg)
    device = str(next(model.parameters()).device)

    moe_layers = discover_moe_layers(model)
    if not moe_layers:
        raise RuntimeError("No MoE layers discovered")

    n = len(moe_layers)
    k = min(args.max_layers, n)
    idxs = sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
    selected = [moe_layers[i] for i in idxs]
    log.info("Selected layers: %s", [h.layer_index for h in selected])

    per_layer_results: Dict[str, Any] = {}
    for handle in selected:
        li = handle.layer_index
        log.info("Layer %d (%s): running %d pairs", li, handle.module_name, len(pairs))

        rho_values: List[float] = []
        scatter: List[Dict] = []

        for pi, pair in enumerate(pairs):
            _, rc_by_layer = _per_pair_routing_contrast(model, tokenizer, pair, [handle], device)
            if li not in rc_by_layer:
                continue
            phi_rc = rc_by_layer[li]

            # Exact Shapley over active expert set (may be truncated to max_k).
            import moe_bias_shapley.shapley as _shp
            orig_max_k = _shp._discover_active_experts.__defaults__
            phi_exact_full = compute_exact_shapley_for_pair(model, tokenizer, pair, handle, device=device)

            active = np.where(phi_exact_full != 0)[0].tolist()
            if len(active) < 2:
                log.debug("Pair %d layer %d: fewer than 2 active experts — skipping", pi, li)
                continue

            # Restrict both phi vectors to the active expert set for fair comparison.
            rc_active = np.abs(phi_rc[active])
            ex_active = np.abs(phi_exact_full[active])

            rho = _spearman_rho(rc_active, ex_active)
            rho_values.append(rho)
            scatter.append({
                "pair_idx": pi,
                "n_active": len(active),
                "spearman_rho": rho,
                "rc_ranking": [int(a) for a in np.argsort(-rc_active)],
                "exact_ranking": [int(a) for a in np.argsort(-ex_active)],
            })
            if (pi + 1) % 5 == 0:
                log.info("  layer %d: %d/%d pairs done, running mean rho=%.3f",
                          li, pi + 1, len(pairs),
                          float(np.mean(rho_values)) if rho_values else float("nan"))

        per_layer_results[f"layer{li}"] = {
            "module_name": handle.module_name,
            "n_pairs_evaluated": len(rho_values),
            "mean_spearman_rho": float(np.mean(rho_values)) if rho_values else float("nan"),
            "std_spearman_rho": float(np.std(rho_values)) if rho_values else float("nan"),
            "scatter": scatter,
        }
        log.info("Layer %d: mean Spearman ρ = %.3f ± %.3f (n=%d pairs)",
                  li,
                  per_layer_results[f"layer{li}"]["mean_spearman_rho"],
                  per_layer_results[f"layer{li}"]["std_spearman_rho"],
                  len(rho_values))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_name": cfg.study_name,
        "model_id": cfg.model_id,
        "model_family": cfg.model_family,
        "n_pairs": len(pairs),
        "layers": per_layer_results,
    }
    (out_dir / "experiment7_proxy_agreement.json").write_text(json.dumps(payload, indent=2))
    log.info("Done. Experiment 7 results in %s", out_dir / "experiment7_proxy_agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
