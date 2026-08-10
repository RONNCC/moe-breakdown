"""Shared model/tokenizer loading for all study driver scripts.

Factored out of scripts/run_bias_study.py so Experiment 3 (interactions) and
Experiment 4 (ablation cross-check) driver scripts can reuse the exact same
loading path (dtype handling, quantization, trust_remote_code) instead of
duplicating it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import torch

from .config import BiasStudyConfig

log = logging.getLogger(__name__)

DENSE_FAMILIES = {"olmo", "dense", "llama"}


def torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _patch_dbrx_dynamic_cache() -> None:
    # Databricks' modeling_dbrx.py calls from_legacy_cache() and to_legacy_cache(),
    # both removed in transformers 5.x. Re-add them for trust_remote_code=True.
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'from_legacy_cache'):
            @classmethod  # type: ignore[misc]
            def from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, layer_past in enumerate(past_key_values):
                        cache.update(layer_past[0], layer_past[1], layer_idx)
                return cache
            DynamicCache.from_legacy_cache = from_legacy_cache
            log.info("Patched DynamicCache.from_legacy_cache for DBRX compatibility")
        if not hasattr(DynamicCache, 'to_legacy_cache'):
            def to_legacy_cache(self):
                # transformers 5.x: DynamicCache.layers is a list of DynamicLayer,
                # each storing keys/values in .keys/.values (not key_cache/value_cache).
                # transformers 4.x: stored in .key_cache/.value_cache lists.
                if hasattr(self, 'layers'):
                    return tuple(
                        (layer.keys, layer.values)
                        for layer in self.layers
                        if getattr(layer, 'is_initialized', False) and layer.keys.numel() > 0
                    )
                # 4.x fallback
                return tuple(
                    (self.key_cache[i], self.value_cache[i])
                    for i in range(len(self.key_cache))
                )
            DynamicCache.to_legacy_cache = to_legacy_cache
            log.info("Patched DynamicCache.to_legacy_cache for DBRX compatibility")
    except Exception as exc:
        log.warning("Could not patch DynamicCache: %s", exc)


def _patch_dbrx_config_cache(model_id: str) -> None:
    # alpindale/dbrx-instruct config.json stores moe_jitter_eps as int 0, but
    # newer huggingface_hub strict dataclass validation requires float | None.
    # Patch the cached file in-place before any transformers call reads it.
    try:
        import json
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            cfg_dict = json.load(f)
        ffn = cfg_dict.get("ffn_config", {})
        if isinstance(ffn.get("moe_jitter_eps"), int):
            ffn["moe_jitter_eps"] = float(ffn["moe_jitter_eps"])
            cfg_dict["ffn_config"] = ffn
            with open(config_path, "w") as f:
                json.dump(cfg_dict, f)
            log.info("Patched moe_jitter_eps int→float in %s", config_path)
    except Exception as exc:
        log.warning("Could not patch DBRX config cache: %s", exc)


def force_eager_gpt_oss(model: Any) -> int:
    """Rebind every GptOssMLP.forward back to the eager transformers class method.

    transformers>=5.13 auto-attaches the fused MXFP4 hub-kernel forward
    (transformers.integrations.mxfp4.mlp_forward) to each layer's GptOssMLP
    when loading gpt-oss checkpoints (e.g. openai/gpt-oss-120b) with
    torch_dtype=bfloat16. That fused path never runs the GptOssTopKRouter
    module (it calls nn.functional.linear on the router weights directly), so
    (a) forward-hooks on the router stay empty and (b) the model's native
    output_router_logits capture (OutputRecorder on GptOssTopKRouter) also
    stays empty. On PACE (torch 2.6) it additionally crashes with
    AttributeError: _CudaDeviceProperties has no 'shared_memory_per_block_optin'.

    Rebinding mlp.forward to the eager GptOssMLP.forward makes the router
    module execute again, so output_router_logits=True capture works.

    NOTE: only the forward is rebound here. If the MXFP4 auto-quantizer also
    swapped the experts module for Mxfp4GptOssExperts (same replace step), a
    pure forward rebind is NOT sufficient — the eager forward calls
    experts(hidden, router_indices, router_scores) which mismatches the fused
    signature; load bf16-dequantized instead so experts stay eager
    GptOssExperts.

    Returns the number of layers rebound (0 = no-op / nothing to do).
    """
    from types import MethodType

    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssMLP
    except Exception as exc:  # noqa: BLE001
        log.warning("force_eager_gpt_oss: cannot import transformers gpt_oss modeling: %s", exc)
        return 0

    if not isinstance(model, GptOssForCausalLM):
        log.info("force_eager_gpt_oss: model is %s (not GptOssForCausalLM) — no-op", type(model).__name__)
        return 0

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        log.warning("force_eager_gpt_oss: no model.model.layers found — no-op")
        return 0

    n_rebound = 0
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        fwd_mod = getattr(mlp.forward, "__module__", "") or ""
        if not fwd_mod.startswith("transformers.integrations.mxfp4"):
            continue
        experts_cls = type(getattr(mlp, "experts", None)).__name__
        if experts_cls == "Mxfp4GptOssExperts":
            log.warning(
                "force_eager_gpt_oss: layer %d experts is %s — eager GptOssMLP.forward calls "
                "experts(hidden, router_indices, router_scores) which mismatches the fused "
                "signature; load bf16-dequantized so experts stay eager GptOssExperts",
                li, experts_cls,
            )
        mlp.forward = MethodType(GptOssMLP.forward, mlp)
        n_rebound += 1
        log.info(
            "force_eager_gpt_oss: rebound layer %d mlp.forward %s -> eager GptOssMLP.forward (experts=%s)",
            li, fwd_mod, experts_cls,
        )
    log.info("force_eager_gpt_oss: rebound %d/%d gpt-oss MLP layers", n_rebound, len(layers))
    return n_rebound


def load_model_and_tokenizer(cfg: BiasStudyConfig) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.model_family == "dbrx":
        _patch_dbrx_dynamic_cache()
        _patch_dbrx_config_cache(cfg.model_id)

    log.info("Loading tokenizer + model: %s (family=%s)", cfg.model_id, cfg.model_family)
    # alpindale/dbrx-instruct is missing tokenizer.json; Xenova/dbrx-instruct-tokenizer
    # is a HF-compatible port that provides it. Load tokenizer from there, weights from cfg.model_id.
    tokenizer_id = "Xenova/dbrx-instruct-tokenizer" if cfg.model_family == "dbrx" else cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        trust_remote_code=cfg.trust_remote_code,
        dtype=torch_dtype(cfg.torch_dtype),
        device_map=cfg.device_map,
    )
    if cfg.model_family == "gpt-oss" and cfg.device_map == "auto":
        # device_map="auto" sizes the map from the on-disk (uint8 MXFP4 packed)
        # param sizes, but Mxfp4Config(dequantize=True) materializes bf16
        # weights (2x memory) during load -> first GPU OOMs without explicit
        # per-device caps. Cap each GPU at ~72% of its VRAM and let CPU take
        # the rest (node mem: 512G for the dequantized-bf16 120B capture).
        # PYTORCH_CUDA_ALLOC_CONF expandable_segments:True reduces the
        # fragmentation that the load-hit-OOM crash showed.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        max_memory: dict = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                max_memory[i] = int(torch.cuda.get_device_properties(i).total_memory * 0.72)
            max_memory["cpu"] = "500GiB"
            kwargs["max_memory"] = max_memory
    if cfg.model_family == "gpt-oss" and cfg.force_eager_moe:
        # openai/gpt-oss-* are pre-quantized MXFP4 checkpoints
        # (config.quantization_config.quant_method == "mxfp4"). Loading them
        # without an explicit config makes transformers>=5.13 auto-attach the
        # fused MXFP4 hub kernel: replace_with_mxfp4_linear swaps every
        # GptOssExperts module for Mxfp4GptOssExperts (forward signature needs
        # scatter_idx; crashes on torch 2.6) and bypasses the GptOssTopKRouter
        # module, so router hooks/capture stay empty (all-zero phi).
        # Passing Mxfp4Config(dequantize=True) dequantizes the checkpoint back
        # to bf16 during load (get_weight_conversions -> Mxfp4Dequantize) and
        # leaves the model fully eager: dimensions/params stay
        # GptOssExperts, forward-hooks and output_router_logits capture work.
        from transformers import Mxfp4Config
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    elif cfg.load_in_8bit or cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
        )

    # Pre-load config so we can patch missing attributes before model __init__
    # runs. Some community models (e.g. alpindale/dbrx-instruct) have a broken
    # custom DbrxConfig that never sets pad_token_id, causing AttributeError.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if not hasattr(model_config, "pad_token_id"):
        model_config.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, config=model_config, **kwargs)
    model.eval()
    if cfg.model_family == "gpt-oss" and cfg.force_eager_moe:
        force_eager_gpt_oss(model)
    return model, tokenizer
