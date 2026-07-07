"""Bias-Shapley computation.

Two attribution methods are implemented, matching the study design (Sec 2.4):

1. `compute_routing_contrast` — fast, approximate, router-guided attribution.
   For each MoE layer and each active expert, the contribution is the
   router-weight-scaled difference in the bias payoff between the
   stereotype and anti-stereotype prompt in a pair. This is the practical
   RGIS-style approximation referenced in the study design (Sec 2.4) and is
   what should be used for the full model ladder (Experiment 1) given
   compute constraints.

2. `compute_exact_shapley_for_layer` — exact Shapley over the active
   top-K expert set for a single token/layer (2^K coalitions), via expert
   output ablation. This is the "genuine MoE-only" exact computation
   described in Sec 2.4, used for validation on a small prompt subsample
   (see Experiment 4 cross-check) since it needs 2^K forward passes.

For dense models, `compute_dense_layer_contrast` computes a leave-one-out
(LOO) approximation of Shapley over transformer-layer FFN modules (there is
no small "active expert set" to exploit, so exact Shapley over all layers is
intractable; LOO is the standard cheap approximation and is explicitly
flagged as such in metadata).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .benchmarks import PromptPair
from .hooks import (
    DenseLayerHandle,
    MoeLayerHandle,
    RouterCaptureState,
    attach_router_hooks,
    detach_hooks,
    discover_dense_ffn_layers,
)

log = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Aggregated expert (or layer) bias-Shapley attribution for a study run."""
    phi: np.ndarray                 # [num_players] aggregated signed attribution
    player_ids: List[str]           # human-readable ids, e.g. "layer3-expert12"
    method: str                     # "routing_contrast" | "exact" | "dense_loo"
    n_pairs: int
    bias_scores: Dict[str, float]   # aggregate bias metrics for sanity-checking
    per_group_phi: Optional[Dict[str, np.ndarray]] = None  # for demographic split (RQ3)


def _bias_gap_from_logits(logp_stereo: float, logp_anti: float) -> float:
    """Scalar bias payoff contribution for one prompt pair: positive means the
    model favors the stereotype completion over the anti-stereotype one.
    """
    return float(logp_stereo - logp_anti)


def _sequence_logprob(model: Any, tokenizer: Any, text: str, device: str) -> float:
    """Average per-token log-probability of `text` under the model (teacher-forced)."""
    import torch
    import torch.nn.functional as F

    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        # out.loss is mean NLL over tokens; convert to mean logprob.
        return float(-out.loss.item())


# =========================================================================== #
# Method 1: routing-contrast (fast, RGIS-style approximation)                  #
# =========================================================================== #

def compute_routing_contrast(
    model: Any,
    tokenizer: Any,
    pairs: List[PromptPair],
    device: str = "cuda",
    demographic_key: Optional[str] = None,
) -> AttributionResult:
    """Fast expert-level bias attribution using router weights as the
    coalition-membership signal (no expert ablation needed — one forward
    pass per prompt).

    phi[layer, expert] += topk_weight(expert | stereo_prompt) * bias_gap
                         - topk_weight(expert | anti_prompt) * bias_gap_baseline

    In practice we use the simpler, still-principled form:
      contribution(expert) = mean over tokens where expert is active of
                              (routing_weight_stereo - routing_weight_anti) * bias_gap
    which is the router-weighted version of the counterfactual bias-gap
    payoff (Sec 2.3), attributing more of the gap to experts that route more
    strongly on the stereotype side relative to the anti-stereotype side.
    """
    state = attach_router_hooks(model)
    if not state.moe_layers:
        detach_hooks(state)
        raise ValueError("No MoE layers discovered — use compute_dense_layer_contrast instead")

    num_layers = len(state.moe_layers)
    max_experts = max(h.num_experts for h in state.moe_layers)
    phi = np.zeros((num_layers, max_experts), dtype=np.float64)
    per_group_phi: Dict[str, np.ndarray] = {}

    gaps: List[float] = []

    for i, pair in enumerate(pairs):
        logp_stereo = _sequence_logprob(model, tokenizer, pair.stereo, device)
        state_stereo = {li: {k: v.clone() for k, v in d.items()} for li, d in state.captured.items()}

        logp_anti = _sequence_logprob(model, tokenizer, pair.anti_stereo, device)
        state_anti = {li: {k: v.clone() for k, v in d.items()} for li, d in state.captured.items()}

        bias_gap = _bias_gap_from_logits(logp_stereo, logp_anti)
        gaps.append(bias_gap)

        for li in range(num_layers):
            if li not in state_stereo or li not in state_anti:
                continue
            topk_idx_s = state_stereo[li]["topk_idx"].reshape(-1, state_stereo[li]["topk_idx"].shape[-1])
            topk_w_s = state_stereo[li]["topk_weight"].reshape(-1, state_stereo[li]["topk_weight"].shape[-1])
            topk_idx_a = state_anti[li]["topk_idx"].reshape(-1, state_anti[li]["topk_idx"].shape[-1])
            topk_w_a = state_anti[li]["topk_weight"].reshape(-1, state_anti[li]["topk_weight"].shape[-1])

            # Mean routing weight per expert, averaged over tokens in the sequence.
            n_experts = state.moe_layers[li].num_experts
            if n_experts <= 0:
                log.warning("Layer %d has unresolved num_experts=%d — skipping", li, n_experts)
                continue
            mean_w_s = np.zeros(n_experts, dtype=np.float64)
            mean_w_a = np.zeros(n_experts, dtype=np.float64)

            idx_s = topk_idx_s.cpu().numpy()
            w_s = topk_w_s.float().cpu().numpy()
            idx_a = topk_idx_a.cpu().numpy()
            w_a = topk_w_a.float().cpu().numpy()

            counts_s = np.zeros(n_experts, dtype=np.float64)
            counts_a = np.zeros(n_experts, dtype=np.float64)
            for tok in range(idx_s.shape[0]):
                for k in range(idx_s.shape[1]):
                    e = int(idx_s[tok, k])
                    mean_w_s[e] += w_s[tok, k]
                    counts_s[e] += 1
            for tok in range(idx_a.shape[0]):
                for k in range(idx_a.shape[1]):
                    e = int(idx_a[tok, k])
                    mean_w_a[e] += w_a[tok, k]
                    counts_a[e] += 1

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_w_s = np.where(counts_s > 0, mean_w_s / np.maximum(counts_s, 1), 0.0)
                mean_w_a = np.where(counts_a > 0, mean_w_a / np.maximum(counts_a, 1), 0.0)

            contribution = (mean_w_s - mean_w_a) * bias_gap
            phi[li, :n_experts] += contribution

            if demographic_key is not None:
                if hasattr(pair, demographic_key):
                    group = getattr(pair, demographic_key) or "unknown"
                else:
                    group = pair.extra.get(demographic_key, "unknown")
                if group not in per_group_phi:
                    per_group_phi[group] = np.zeros_like(phi)
                per_group_phi[group][li, :n_experts] += contribution

        if (i + 1) % 25 == 0:
            log.info("routing_contrast: processed %d/%d pairs", i + 1, len(pairs))

    detach_hooks(state)

    if len(pairs) > 0:
        phi /= len(pairs)
        for g in per_group_phi:
            per_group_phi[g] /= len(pairs)

    player_ids = [
        f"layer{li}-expert{e}"
        for li in range(num_layers)
        for e in range(max_experts)
    ]

    return AttributionResult(
        phi=phi.flatten(),
        player_ids=player_ids,
        method="routing_contrast",
        n_pairs=len(pairs),
        bias_scores={
            "mean_bias_gap": float(np.mean(gaps)) if gaps else 0.0,
            "std_bias_gap": float(np.std(gaps)) if gaps else 0.0,
        },
        per_group_phi=per_group_phi or None,
    )


# =========================================================================== #
# Method 2: exact Shapley over active expert set (per token/layer, 2^K)        #
# =========================================================================== #

def _shapley_weights(k: int) -> Dict[int, float]:
    """Precompute Shapley coalition weights for a player set of size k, keyed
    by |S| (size of coalition excluding the player)."""
    import math
    return {s: (math.factorial(s) * math.factorial(k - s - 1)) / math.factorial(k) for s in range(k)}


def _discover_active_experts(
    model: Any,
    tokenizer: Any,
    pair: PromptPair,
    moe_layer: MoeLayerHandle,
    device: str,
    max_k: int = 10,
) -> List[int]:
    """Find the top-K active expert set for `moe_layer` on `pair.stereo`,
    truncated to `max_k` so 2^K coalition enumeration stays tractable."""
    state = attach_router_hooks(model)
    _ = _sequence_logprob(model, tokenizer, pair.stereo, device)
    if moe_layer.layer_index not in state.captured:
        detach_hooks(state)
        raise ValueError(f"Layer {moe_layer.layer_index} did not fire during forward pass")
    topk_idx = state.captured[moe_layer.layer_index]["topk_idx"]
    detach_hooks(state)

    active_experts = sorted(set(topk_idx.flatten().tolist()))
    if len(active_experts) > max_k:
        log.warning(
            "Active expert set size %d too large for exact Shapley — truncating to %d",
            len(active_experts), max_k,
        )
        active_experts = active_experts[:max_k]
    return active_experts


def _build_coalition_payoff_cache(
    model: Any,
    tokenizer: Any,
    pair: PromptPair,
    moe_layer: MoeLayerHandle,
    active_experts: List[int],
    device: str,
) -> Dict[frozenset, float]:
    """Ablate every subset of `active_experts` (2^k forward-pass pairs) and
    cache the resulting bias-gap payoff V(S) for each coalition S.

    Shared by `compute_exact_shapley_for_pair` (marginal Shapley values) and
    `compute_shapley_interactions_for_pair` (pairwise interaction/synergy
    values, Experiment 3) since both need the same full coalition lattice —
    computing it once avoids duplicating the 2^K forward passes.
    """
    import torch

    k = len(active_experts)
    original_forwards = {e: moe_layer.experts[e].forward for e in active_experts}

    def make_zero_forward():
        def zero_forward(x, *args, **kwargs):
            return torch.zeros_like(x)
        return zero_forward

    def set_coalition(active_mask: set) -> None:
        for e in active_experts:
            if e in active_mask:
                moe_layer.experts[e].forward = original_forwards[e]
            else:
                moe_layer.experts[e].forward = make_zero_forward()

    def restore() -> None:
        for e in active_experts:
            moe_layer.experts[e].forward = original_forwards[e]

    def payoff(active_mask: set) -> float:
        set_coalition(active_mask)
        logp_s = _sequence_logprob(model, tokenizer, pair.stereo, device)
        logp_a = _sequence_logprob(model, tokenizer, pair.anti_stereo, device)
        return _bias_gap_from_logits(logp_s, logp_a)

    cache: Dict[frozenset, float] = {}
    try:
        for r in range(k + 1):
            for combo in itertools.combinations(active_experts, r):
                s = frozenset(combo)
                cache[s] = payoff(set(s))
    finally:
        restore()

    return cache


def compute_exact_shapley_for_pair(
    model: Any,
    tokenizer: Any,
    pair: PromptPair,
    moe_layer: MoeLayerHandle,
    device: str = "cuda",
) -> np.ndarray:
    """Exact Shapley values for the top-K active experts at `moe_layer`, using
    expert-output ablation, evaluated on the bias-gap payoff for `pair`.

    Because only the top-K routed experts for the *last* token's forward
    pass are considered "active" (per Sec 2.4), the player set is small
    (K, typically 1-8) and 2^K coalitions is tractable.

    NOTE: this requires 2^K forward passes per pair per layer, so it should
    only be run on a prompt subsample (Experiment 4 cross-check), not the
    full benchmark set.
    """
    active_experts = _discover_active_experts(model, tokenizer, pair, moe_layer, device)
    k = len(active_experts)
    phi = np.zeros(moe_layer.num_experts, dtype=np.float64)
    if k == 0:
        return phi

    weights = _shapley_weights(k)
    cache = _build_coalition_payoff_cache(model, tokenizer, pair, moe_layer, active_experts, device)

    for e in active_experts:
        others = [x for x in active_experts if x != e]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                s_with_e = frozenset(combo + (e,))
                marginal = cache[s_with_e] - cache[s]
                total += weights[r] * marginal
        phi[e] = total

    return phi


# =========================================================================== #
# Experiment 3: Shapley interaction values (collectivity / synergy check)      #
# =========================================================================== #

@dataclass
class InteractionResult:
    """Marginal (phi) vs pairwise-synergy (interaction) decomposition of bias
    attribution for one MoE layer, aggregated over a prompt subsample."""
    active_experts: List[int]
    phi: Dict[int, float]                    # marginal Shapley value per expert
    interactions: Dict[tuple, float]          # {(e_i, e_j): interaction value}
    synergy_fraction: float                   # |synergy| / (|marginal| + |synergy|)
    n_pairs: int


def _shapley_interaction_weights(n: int) -> Dict[int, float]:
    """Coalition weights for the pairwise Shapley interaction index (Grabisch
    & Roubens), keyed by |S| where S ranges over subsets of N\\{i,j}
    (|N\\{i,j}| = n - 2)."""
    import math
    return {
        s: (math.factorial(s) * math.factorial(n - s - 2)) / math.factorial(n - 1)
        for s in range(n - 1)
    }


def compute_shapley_interactions_for_pair(
    model: Any,
    tokenizer: Any,
    pair: PromptPair,
    moe_layer: MoeLayerHandle,
    device: str = "cuda",
) -> InteractionResult:
    """Decompose the bias-gap payoff at `moe_layer` into per-expert marginal
    Shapley values and pairwise Shapley *interaction* values (Sec 3.3 / C2-lite
    collectivity check), using the standard interaction index:

        Phi_ij = sum_{S subseteq N\\{i,j}} w(|S|) *
                 (V(S+{i,j}) - V(S+{i}) - V(S+{j}) + V(S))

    A large |Phi_ij| relative to the individual |phi_i|, |phi_j| indicates
    bias is a property of the expert *pair/coalition* (synergy), not either
    expert alone — evidence for the "standing committees" hypothesis
    [arXiv 2601.03425] rather than single-expert specialization.
    """
    active_experts = _discover_active_experts(model, tokenizer, pair, moe_layer, device)
    n = len(active_experts)
    if n < 2:
        return InteractionResult(active_experts=active_experts, phi={}, interactions={}, synergy_fraction=0.0, n_pairs=0)

    cache = _build_coalition_payoff_cache(model, tokenizer, pair, moe_layer, active_experts, device)

    # Marginal Shapley values (reuse the same cache — no extra forward passes).
    marginal_weights = _shapley_weights(n)
    phi: Dict[int, float] = {}
    for e in active_experts:
        others = [x for x in active_experts if x != e]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                s_with_e = frozenset(combo + (e,))
                total += marginal_weights[r] * (cache[s_with_e] - cache[s])
        phi[e] = total

    # Pairwise interaction values.
    interaction_weights = _shapley_interaction_weights(n)
    interactions: Dict[tuple, float] = {}
    for i_idx, e_i in enumerate(active_experts):
        for e_j in active_experts[i_idx + 1:]:
            rest = [x for x in active_experts if x not in (e_i, e_j)]
            total = 0.0
            for r in range(len(rest) + 1):
                for combo in itertools.combinations(rest, r):
                    s = frozenset(combo)
                    s_ij = frozenset(combo + (e_i, e_j))
                    s_i = frozenset(combo + (e_i,))
                    s_j = frozenset(combo + (e_j,))
                    delta = cache[s_ij] - cache[s_i] - cache[s_j] + cache[s]
                    total += interaction_weights[r] * delta
            interactions[(e_i, e_j)] = total

    abs_marginal = sum(abs(v) for v in phi.values())
    abs_synergy = sum(abs(v) for v in interactions.values())
    synergy_fraction = abs_synergy / (abs_marginal + abs_synergy) if (abs_marginal + abs_synergy) > 0 else 0.0

    return InteractionResult(
        active_experts=active_experts,
        phi=phi,
        interactions=interactions,
        synergy_fraction=synergy_fraction,
        n_pairs=1,
    )


def aggregate_interaction_results(results: List[InteractionResult]) -> Dict[str, Any]:
    """Aggregate per-pair InteractionResults into a single summary dict:
    mean synergy fraction (the headline C2-lite number) plus the top
    expert-pairs by mean |interaction| across the subsample.
    """
    if not results:
        return {"n_pairs": 0, "mean_synergy_fraction": 0.0, "top_interactions": []}

    synergy_fracs = [r.synergy_fraction for r in results if r.n_pairs > 0]
    pair_totals: Dict[tuple, float] = {}
    pair_counts: Dict[tuple, int] = {}
    for r in results:
        for key, val in r.interactions.items():
            pair_totals[key] = pair_totals.get(key, 0.0) + abs(val)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    mean_interactions = {k: pair_totals[k] / pair_counts[k] for k in pair_totals}
    top_interactions = sorted(mean_interactions.items(), key=lambda kv: -kv[1])[:20]

    return {
        "n_pairs": len(results),
        "mean_synergy_fraction": float(np.mean(synergy_fracs)) if synergy_fracs else 0.0,
        "top_interactions": [
            {"experts": list(k), "mean_abs_interaction": v} for k, v in top_interactions
        ],
    }


# =========================================================================== #
# Dense fallback: leave-one-out layer ablation (approximate Shapley)           #
# =========================================================================== #

def compute_dense_layer_contrast(
    model: Any,
    tokenizer: Any,
    pairs: List[PromptPair],
    device: str = "cuda",
) -> AttributionResult:
    """Leave-one-out (LOO) approximation of per-layer bias contribution for
    dense models. Not exact Shapley (the full-layer coalition space is
    intractable), but a standard, cheap approximation: contribution(layer) =
    V(full model) - V(model with that layer's FFN zero-ablated), averaged
    over prompt pairs. This is explicitly flagged as an approximation in the
    returned AttributionResult.method field so downstream H/Gini comparisons
    (Experiment 2 / RQ2) are interpreted with that caveat (Sec 2.6).
    """
    import torch

    layers = discover_dense_ffn_layers(model)
    n_layers = len(layers)
    phi = np.zeros(n_layers, dtype=np.float64)
    gaps: List[float] = []

    for i, pair in enumerate(pairs):
        logp_s_full = _sequence_logprob(model, tokenizer, pair.stereo, device)
        logp_a_full = _sequence_logprob(model, tokenizer, pair.anti_stereo, device)
        full_gap = _bias_gap_from_logits(logp_s_full, logp_a_full)
        gaps.append(full_gap)

        for layer in layers:
            original_forward = layer.module.forward

            def zero_forward(x, *args, __orig=original_forward, **kwargs):
                return torch.zeros_like(x)

            layer.module.forward = zero_forward
            try:
                logp_s = _sequence_logprob(model, tokenizer, pair.stereo, device)
                logp_a = _sequence_logprob(model, tokenizer, pair.anti_stereo, device)
                ablated_gap = _bias_gap_from_logits(logp_s, logp_a)
            finally:
                layer.module.forward = original_forward

            phi[layer.layer_index] += (full_gap - ablated_gap)

        if (i + 1) % 10 == 0:
            log.info("dense_loo: processed %d/%d pairs", i + 1, len(pairs))

    if len(pairs) > 0:
        phi /= len(pairs)

    player_ids = [f"layer{layer.layer_index}-ffn" for layer in layers]

    return AttributionResult(
        phi=phi,
        player_ids=player_ids,
        method="dense_loo",
        n_pairs=len(pairs),
        bias_scores={
            "mean_bias_gap": float(np.mean(gaps)) if gaps else 0.0,
            "std_bias_gap": float(np.std(gaps)) if gaps else 0.0,
        },
    )
