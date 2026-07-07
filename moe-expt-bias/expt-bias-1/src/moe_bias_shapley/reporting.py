"""Result serialization: save/load AttributionResult + metrics as JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .metrics import ConcentrationMetrics, compute_concentration_metrics
from .shapley import AttributionResult

log = logging.getLogger(__name__)


def save_results(
    out_dir: str | Path,
    result: AttributionResult,
    metadata: Dict[str, Any],
    shard_tag: Optional[str] = None,
) -> Path:
    """Save an AttributionResult + its concentration metrics + run metadata.

    Single-shard (default): writes result.json / phi.npy.
    Multi-shard: writes result_<shard_tag>.json / phi_<shard_tag>.npy so that
    all shards for the same study land in the same directory and can be merged
    post-hoc by globbing result_shard*.json.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_concentration_metrics(result.phi)

    payload: Dict[str, Any] = {
        "metadata": metadata,
        "method": result.method,
        "n_pairs": result.n_pairs,
        "n_players": len(result.player_ids),
        "bias_scores": result.bias_scores,
        "concentration_metrics": metrics.as_dict(),
        "top_players": _top_players(result, n=20),
    }

    if result.per_group_phi:
        # per_group_phi entries carry the same (num_layers, max_experts) shape as
        # the pre-flatten main phi accumulator — flatten before computing
        # metrics so they're comparable to the main (already-flat) phi vector.
        payload["per_group_concentration_metrics"] = {
            group: compute_concentration_metrics(phi.flatten()).as_dict()
            for group, phi in result.per_group_phi.items()
        }

    result_stem = f"result_{shard_tag}" if shard_tag else "result"
    phi_stem = f"phi_{shard_tag}" if shard_tag else "phi"

    (out_dir / f"{result_stem}.json").write_text(json.dumps(payload, indent=2))
    np.save(out_dir / f"{phi_stem}.npy", result.phi)
    (out_dir / "player_ids.json").write_text(json.dumps(result.player_ids))
    if result.routing_freq is not None:
        routing_stem = f"routing_freq_{shard_tag}" if shard_tag else "routing_freq"
        np.save(out_dir / f"{routing_stem}.npy", result.routing_freq)

    if result.per_group_phi:
        for group, phi in result.per_group_phi.items():
            safe_name = group.replace("/", "_").replace(" ", "_")
            suffix = f"_{shard_tag}" if shard_tag else ""
            np.save(out_dir / f"phi_group_{safe_name}{suffix}.npy", phi.flatten())

    log.info("Saved bias-Shapley results to %s", out_dir)
    return out_dir / f"{result_stem}.json"


def _top_players(result: AttributionResult, n: int = 20) -> list[dict]:
    order = np.argsort(-np.abs(result.phi))[:n]
    return [
        {"player_id": result.player_ids[i], "phi": float(result.phi[i])}
        for i in order
    ]


def load_results(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir).expanduser()
    result_path = out_dir / "result.json"
    payload = json.loads(result_path.read_text())
    phi_path = out_dir / "phi.npy"
    if phi_path.exists():
        payload["phi"] = np.load(phi_path)
    return payload
