#!/usr/bin/env python3
"""Script to merge sharded bias-Shapley result files.

Combines result_shard*of*.json, phi_shard*of*.npy, and routing_freq_shard*of*.npy
into final, un-sharded result.json, phi.npy, and routing_freq.npy, recomputing
properly weighted mean bias scores, pooled standard deviations, and final
concentration metrics.
"""

import argparse
import json
import logging
from pathlib import Path
import re
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("merge_shards")


def merge_study_shards(study_dir: Path) -> None:
    study_dir = Path(study_dir).expanduser()
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory does not exist: {study_dir}")

    log.info("Merging shards in study directory: %s", study_dir)

    # Find all result shards (supports both 0-based and 1-based, e.g. shard0of2 or shard1of2)
    shard_paths = sorted(list(study_dir.glob("result_shard*of*.json")))
    if not shard_paths:
        log.warning("No sharded result JSON files found under %s", study_dir)
        return

    log.info("Found %d shard(s) to merge:", len(shard_paths))
    for p in shard_paths:
        log.info("  - %s", p.name)

    # We need to load local package for metrics computation
    import sys
    src_dir = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_dir))
    
    from moe_bias_shapley.metrics import compute_concentration_metrics

    # Variables to pool
    total_pairs = 0
    shards_data = []

    # First pass: Load all shards and metadata
    for p in shard_paths:
        shard_tag_match = re.search(r"result_(shard\d+of\d+)\.json", p.name)
        if not shard_tag_match:
            log.warning("Skipping file with unrecognized pattern: %s", p.name)
            continue
        shard_tag = shard_tag_match.group(1)
        
        payload = json.loads(p.read_text())
        n_pairs = payload["n_pairs"]
        total_pairs += n_pairs
        
        # Load associated arrays
        phi_path = study_dir / f"phi_{shard_tag}.npy"
        freq_path = study_dir / f"routing_freq_{shard_tag}.npy"
        
        if not phi_path.exists():
            raise FileNotFoundError(f"Associated phi array not found: {phi_path}")
            
        phi = np.load(phi_path)
        routing_freq = np.load(freq_path) if freq_path.exists() else None
        
        shards_data.append({
            "n_pairs": n_pairs,
            "payload": payload,
            "phi": phi,
            "routing_freq": routing_freq
        })

    if not shards_data:
        log.error("No valid shard data processed.")
        return

    # 1. Merge player IDs
    player_ids_path = study_dir / "player_ids.json"
    if player_ids_path.exists():
        player_ids = json.loads(player_ids_path.read_text())
    else:
        # Fallback if player_ids.json not found
        player_ids = [f"player_{i}" for i in range(len(shards_data[0]["phi"]))]

    # 2. Compute weighted averages for phi and routing_freq
    phi_merged = np.zeros_like(shards_data[0]["phi"], dtype=np.float64)
    routing_freq_merged = np.zeros_like(shards_data[0]["routing_freq"], dtype=np.float64) if shards_data[0]["routing_freq"] is not None else None

    for s in shards_data:
        weight = s["n_pairs"] / total_pairs
        phi_merged += s["phi"] * weight
        if routing_freq_merged is not None and s["routing_freq"] is not None:
            routing_freq_merged += s["routing_freq"] * weight

    # 3. Combine bias scores (pooled mean and variance)
    # Mean of means
    mean_bias_gap_merged = 0.0
    for s in shards_data:
        weight = s["n_pairs"] / total_pairs
        mean_bias_gap_merged += s["payload"]["bias_scores"]["mean_bias_gap"] * weight

    # Pooled variance:
    # Var_pooled = (sum(n_j * (var_j + mean_j^2)) / n_total) - mean_pooled^2
    sum_weighted_sq = 0.0
    for s in shards_data:
        var_j = s["payload"]["bias_scores"]["std_bias_gap"] ** 2
        mean_j = s["payload"]["bias_scores"]["mean_bias_gap"]
        sum_weighted_sq += s["n_pairs"] * (var_j + mean_j ** 2)
        
    var_merged = (sum_weighted_sq / total_pairs) - (mean_bias_gap_merged ** 2)
    std_bias_gap_merged = np.sqrt(max(0.0, var_merged))

    # 4. Recompute final concentration metrics
    metrics = compute_concentration_metrics(phi_merged)

    # 5. Determine top players
    order = np.argsort(-np.abs(phi_merged))[:20]
    top_players = [
        {"player_id": player_ids[i], "phi": float(phi_merged[i])}
        for i in order
    ]

    # 6. Build final payload
    merged_payload = {
        "metadata": shards_data[0]["payload"]["metadata"],
        "method": shards_data[0]["payload"]["method"],
        "n_pairs": total_pairs,
        "n_players": len(player_ids),
        "bias_scores": {
            "mean_bias_gap": float(mean_bias_gap_merged),
            "std_bias_gap": float(std_bias_gap_merged)
        },
        "concentration_metrics": metrics.as_dict(),
        "top_players": top_players
    }
    
    # Preserve metadata updates
    merged_payload["metadata"].pop("shard_idx", None)
    merged_payload["metadata"].pop("num_shards", None)

    # Save outputs
    (study_dir / "result.json").write_text(json.dumps(merged_payload, indent=2))
    np.save(study_dir / "phi.npy", phi_merged)
    if routing_freq_merged is not None:
        np.save(study_dir / "routing_freq.npy", routing_freq_merged)

    log.info("=== Merged Results ===")
    log.info("Total Pairs: %d", total_pairs)
    log.info("Mean Bias Gap: %.4f (std: %.4f)", mean_bias_gap_merged, std_bias_gap_merged)
    log.info("Concentration Metrics:")
    log.info("  Entropy (H): %.4f", metrics.entropy)
    log.info("  Gini: %.4f", metrics.gini)
    log.info("  Top 5 Fraction: %.4f", metrics.top_5_fraction)
    log.info("  Top 10%% Fraction: %.4f", metrics.top_10pct_fraction)
    log.info("Successfully saved merged results to %s", study_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge sharded study results")
    p.add_argument("--study-dir", required=True, help="Path to study output folder")
    args = p.parse_args()
    merge_study_shards(Path(args.study_dir))


if __name__ == "__main__":
    main()
