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


def load_model_and_tokenizer(cfg: BiasStudyConfig) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer + model: %s (family=%s)", cfg.model_id, cfg.model_family)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
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

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    return model, tokenizer
