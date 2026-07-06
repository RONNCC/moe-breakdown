"""Router / expert hook management for MoE models (and FFN-layer hooks for dense).

We rely on the fact that current HF implementations of MoE blocks (OLMoE,
Mixtral, Qwen3-MoE, ERNIE-4.5 MoE) all expose a `.gate` (router linear layer)
and a `.experts` (ModuleList of per-expert FFNs) on a "SparseMoeBlock"-style
submodule. Rather than hardcoding per-architecture module paths (fragile
across transformers versions), we auto-discover these blocks generically by
inspecting submodule attributes. This keeps one hooking implementation
working across the whole model ladder in the study design.

For dense models (no MoE blocks found), we fall back to per-transformer-layer
MLP modules as the "player" set for a leave-one-out layer-ablation
approximation of Shapley (see shapley.compute_dense_layer_contrast).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class MoeLayerHandle:
    """A discovered MoE block within the model."""
    layer_index: int
    module_name: str
    module: Any               # the SparseMoeBlock-like submodule
    gate: Any                 # nn.Linear router
    experts: Any               # nn.ModuleList of expert FFNs
    num_experts: int
    topk: int


@dataclass
class RouterCaptureState:
    """Holds captured router logits/top-k indices for the *last* forward pass.

    captured[layer_index] = {
        "router_logits": Tensor [num_tokens, num_experts],
        "topk_idx": Tensor [num_tokens, topk],
        "topk_weight": Tensor [num_tokens, topk],
    }
    """
    moe_layers: List[MoeLayerHandle] = field(default_factory=list)
    captured: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _hook_handles: List[Any] = field(default_factory=list)

    def clear(self) -> None:
        self.captured = {}


def _looks_like_moe_block(module: Any) -> bool:
    return hasattr(module, "gate") and hasattr(module, "experts")


def _resolve_num_experts(module: Any) -> int:
    """Determine the number of experts for a SparseMoeBlock-like module.

    Different transformers architectures/versions store this differently:
    - Mixtral/Qwen3-MoE (older style): `.gate` is a plain `nn.Linear` whose
      `out_features` == num_experts, and `.experts` is an `nn.ModuleList`.
    - OLMoE (current transformers): `.gate` is a custom `OlmoeTopKRouter` with
      its own `.num_experts` attribute (not `nn.Linear`, no `out_features`),
      and `.experts` is a fused `OlmoeExperts` module with `.num_experts` and
      fused `nn.Parameter` weight tensors (no `__len__`).
    Check the most specific/reliable sources first (gate/experts submodule
    attributes), then fall back to the block itself, then to Linear shape or
    ModuleList length.
    """
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        for owner in (gate, experts, module):
            val = getattr(owner, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    out_features = getattr(gate, "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features
    try:
        n = len(experts)
        if n > 0:
            return n
    except TypeError:
        pass
    return 0


def _resolve_topk(module: Any) -> int:
    """Determine top-k routing count, checking gate/experts submodules first
    (see `_resolve_num_experts` docstring for why the block itself often
    lacks this attribute in newer architectures like OLMoE)."""
    gate = getattr(module, "gate", None)
    experts = getattr(module, "experts", None)
    for attr in ("top_k", "topk", "num_experts_per_tok"):
        for owner in (gate, experts, module):
            val = getattr(owner, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return 1


def discover_moe_layers(model: Any) -> List[MoeLayerHandle]:
    """Walk the model and find all SparseMoeBlock-like submodules."""
    handles: List[MoeLayerHandle] = []
    layer_idx = 0
    for name, module in model.named_modules():
        if _looks_like_moe_block(module):
            num_experts = _resolve_num_experts(module)
            topk = _resolve_topk(module)
            handles.append(MoeLayerHandle(
                layer_index=layer_idx,
                module_name=name,
                module=module,
                gate=module.gate,
                experts=module.experts,
                num_experts=num_experts,
                topk=int(topk),
            ))
            layer_idx += 1
    log.info("Discovered %d MoE layers", len(handles))
    return handles


def attach_router_hooks(model: Any) -> RouterCaptureState:
    """Attach forward hooks on each MoE block's `.gate` to capture router logits
    and (if available) the resulting top-k expert indices/weights.
    """
    import torch

    state = RouterCaptureState(moe_layers=discover_moe_layers(model))

    def make_hook(layer_index: int, topk: int):
        def hook(_module, _inputs, output):
            router_logits = output if not isinstance(output, tuple) else output[0]
            with torch.no_grad():
                routing_weights = torch.softmax(router_logits.float(), dim=-1)
                topk_weight, topk_idx = torch.topk(routing_weights, k=topk, dim=-1)
            state.captured[layer_index] = {
                "router_logits": router_logits.detach(),
                "topk_idx": topk_idx.detach(),
                "topk_weight": topk_weight.detach(),
            }
        return hook

    for h in state.moe_layers:
        handle = h.gate.register_forward_hook(make_hook(h.layer_index, h.topk))
        state._hook_handles.append(handle)

    return state


def detach_hooks(state: RouterCaptureState) -> None:
    for handle in state._hook_handles:
        handle.remove()
    state._hook_handles = []


# --------------------------------------------------------------------------- #
# Dense-model FFN layer discovery (fallback when no MoE blocks are found)      #
# --------------------------------------------------------------------------- #

@dataclass
class DenseLayerHandle:
    layer_index: int
    module_name: str
    module: Any


def discover_dense_ffn_layers(model: Any) -> List[DenseLayerHandle]:
    """Find per-transformer-block MLP/FFN modules for dense models.

    Looks for common attribute names used by HF dense decoder layers
    (`.mlp` on OlmoDecoderLayer / LlamaDecoderLayer / etc.).
    """
    handles: List[DenseLayerHandle] = []
    layers = None
    for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            layers = obj
            break

    if layers is None:
        raise ValueError("Could not locate transformer layers for dense FFN discovery")

    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        handles.append(DenseLayerHandle(layer_index=idx, module_name=f"layers.{idx}.mlp", module=mlp))

    log.info("Discovered %d dense FFN layers", len(handles))
    return handles
