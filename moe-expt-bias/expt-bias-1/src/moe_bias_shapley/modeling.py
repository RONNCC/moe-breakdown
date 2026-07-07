"""Shared model/tokenizer loading for all study driver scripts.

Factored out of scripts/run_bias_study.py so Experiment 3 (interactions) and
Experiment 4 (ablation cross-check) driver scripts can reuse the exact same
loading path (dtype handling, quantization, trust_remote_code) instead of
duplicating it.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

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
    if cfg.load_in_8bit or cfg.load_in_4bit:
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
    return model, tokenizer
