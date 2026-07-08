"""Router / expert hook management for MoE models (and FFN-layer hooks for dense).

We rely on the fact that current HF implementations of MoE blocks (OLMoE,
Mixtral, Phi-3.5-MoE) all expose a router submodule (`.gate` for Mixtral/OLMoE,
`.router` for Phimoe/Phi-3.5-MoE — see `_get_gate`) and a `.experts` (either an
`nn.ModuleList` of per-expert FFNs, or a fused single-module implementation
like OlmoeExperts/PhimoeExperts) on a "SparseMoeBlock"-style submodule. Rather
than hardcoding per-architecture module paths (fragile across transformers
versions), we auto-discover these blocks generically by inspecting submodule
attributes. This keeps one hooking implementation working across the whole
model ladder in the study design.

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
    return (hasattr(module, "gate") or hasattr(module, "router")) and hasattr(module, "experts")


def _get_gate(module: Any) -> Any:
    """Return the router/gate submodule, whichever attribute name this
    architecture uses. Most HF SparseMoeBlock impls use `.gate` (Mixtral,
    OLMoE), but Phimoe (Phi-3.5-MoE) names it `.router` instead."""
    gate = getattr(module, "gate", None)
    return gate if gate is not None else getattr(module, "router", None)


def _resolve_num_experts(module: Any) -> int:
    """Determine the number of experts for a SparseMoeBlock-like module.

    Different transformers architectures/versions store this differently:
    - Mixtral (older style): `.gate` is a plain `nn.Linear` whose
      `out_features` == num_experts, and `.experts` is an `nn.ModuleList`.
    - OLMoE (current transformers): `.gate` is a custom `OlmoeTopKRouter` with
      its own `.num_experts` attribute (not `nn.Linear`, no `out_features`),
      and `.experts` is a fused `OlmoeExperts` module with `.num_experts` and
      fused `nn.Parameter` weight tensors (no `__len__`).
    Check the most specific/reliable sources first (gate/experts submodule
    attributes), then fall back to the block itself, then to Linear shape or
    ModuleList length.
    """
    gate = _get_gate(module)
    experts = getattr(module, "experts", None)
    for attr in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
        for owner in (gate, experts, module):
            val = getattr(owner, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    # Plain nn.Linear router (e.g. Mixtral, some DBRX custom code)
    out_features = getattr(gate, "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features
    # Wrapped router (e.g. DbrxRouter where the Linear lives at .layer)
    inner = getattr(gate, "layer", None)
    out_features = getattr(inner, "out_features", None)
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
    gate = _get_gate(module)
    experts = getattr(module, "experts", None)
    for attr in ("top_k", "topk", "num_experts_per_tok", "moe_top_k", "num_experts_per_token"):
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
                gate=_get_gate(module),
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

    Two hooks are registered per layer:
    1. Primary: on h.gate (the router submodule). Works for Mixtral/OLMoE/Phi where
       the gate is a plain nn.Linear or a simple submodule.
    2. Fallback: on h.module (the outer MoE block, e.g. GptOssMLP). Fires only if the
       primary hook did not populate state.captured[layer_index]. Re-runs the gate
       manually using the block's input to capture routing.

    The fallback handles architectures (e.g. openai/gpt-oss-120b, transformers>=5.13)
    where @use_kernel_forward_from_hub or accelerate device_map hooks prevent the gate
    submodule's register_forward_hook from firing even when USE_HUB_KERNELS=NO.
    """
    import torch

    state = RouterCaptureState(moe_layers=discover_moe_layers(model))

    def make_gate_hook(layer_index: int, topk: int):
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

    def make_outer_fallback_hook(layer_index: int, gate_module: Any, topk: int):
        # Fires on the outer MoE block after its forward completes.
        # If the gate hook already captured routing for this layer, this is a no-op.
        # Otherwise re-runs the gate on the block's input to obtain routing info.
        def hook(_module, inputs, _output):
            if layer_index in state.captured:
                return
            try:
                with torch.no_grad():
                    h_flat = inputs[0].reshape(-1, inputs[0].shape[-1])
                    gate_out = gate_module(h_flat)
                    logits = gate_out if not isinstance(gate_out, tuple) else gate_out[0]
                    routing_weights = torch.softmax(logits.float(), dim=-1)
                    topk_weight, topk_idx = torch.topk(routing_weights, k=topk, dim=-1)
                state.captured[layer_index] = {
                    "router_logits": logits.detach(),
                    "topk_idx": topk_idx.detach(),
                    "topk_weight": topk_weight.detach(),
                }
            except Exception as e:
                log.warning("outer_fallback_hook layer %d: %s", layer_index, e)
        return hook

    for h in state.moe_layers:
        handle = h.gate.register_forward_hook(make_gate_hook(h.layer_index, h.topk))
        state._hook_handles.append(handle)
        fallback = h.module.register_forward_hook(
            make_outer_fallback_hook(h.layer_index, h.gate, h.topk)
        )
        state._hook_handles.append(fallback)

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
