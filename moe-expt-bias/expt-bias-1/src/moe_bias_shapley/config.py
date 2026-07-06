"""Config dataclasses and YAML loading for the bias-Shapley study."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SlurmConfig:
    partition: str = "ice-gpu"
    account: Optional[str] = None
    qos: str = "coc-ice"
    time: str = "04:00:00"
    cpus_per_task: int = 8
    gpus_per_node: int = 1
    gpu_type: Optional[str] = "a100"
    use_gres: bool = True
    mem: str = "80G"
    modules: List[str] = field(default_factory=lambda: ["gcc/12.3.0", "python/3.11", "cuda/12.1.1"])
    workdir: str = "/home/hice1/sghose7/scratch/moe-breakdown"
    uv_env_dir: Optional[str] = None
    sbatch_script: str = "run_bias_study.sbatch"
    extra_sbatch_args: List[str] = field(default_factory=list)


@dataclass
class BiasStudyConfig:
    # ── identity ──────────────────────────────────────────────────────────────
    study_name: str = "unnamed-bias-study"

    # ── model ─────────────────────────────────────────────────────────────────
    model_id: str = "allenai/OLMoE-1B-7B"
    # Family tag used to pick the right hooking strategy:
    #   olmoe   → allenai/OLMoE-1B-7B (top-1, 64 experts)
    #   mixtral → mistralai/Mixtral-8x7B-v0.1 (top-2, 8 experts)
    #   qwen3   → Qwen/Qwen3-30B-A3B (top-8, 128 experts)
    #   olmo    → allenai/OLMo-7B (dense; FFN-layer attribution)
    #   dense   → generic dense model (FFN-layer attribution)
    model_family: str = "olmoe"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"   # bfloat16 | float16 | float32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "auto"

    # ── experiment ────────────────────────────────────────────────────────────
    benchmarks: List[str] = field(default_factory=lambda: ["stereoset"])
    # shapley_method:
    #   routing_contrast  — contrastive routing weights (fast, 2 fwd passes/pair)
    #   exact             — true Shapley via expert ablation (slow, 2^K passes/pair/layer)
    shapley_method: str = "routing_contrast"
    # For exact Shapley: which layers to run ablation on (empty = all MoE layers).
    ablation_layer_indices: List[int] = field(default_factory=list)
    max_prompts: Optional[int] = None   # None = use all prompts
    batch_size: int = 1                 # prompts per GPU batch (keep at 1 for now)
    seed: int = 42

    # ── output ────────────────────────────────────────────────────────────────
    output_root: str = "~/scratch/moe-breakdown-bias-runs"

    # ── slurm ─────────────────────────────────────────────────────────────────
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # ── derived (filled at load time) ─────────────────────────────────────────
    num_experts: int = 0   # filled from model config at runtime if 0
    topk: int = 0          # filled from model config at runtime if 0


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def _parse_slurm(raw: dict) -> SlurmConfig:
    return SlurmConfig(**{k: v for k, v in raw.items() if hasattr(SlurmConfig, k) or k in SlurmConfig.__dataclass_fields__})


def load_bias_study_config(path: str | Path) -> BiasStudyConfig:
    path = Path(path)
    raw: dict = yaml.safe_load(path.read_text()) or {}

    slurm_raw = raw.pop("slurm", {})
    slurm = SlurmConfig(**slurm_raw) if slurm_raw else SlurmConfig()

    # Only forward known fields to avoid dataclass complaints.
    known = {f for f in BiasStudyConfig.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in known}
    filtered["slurm"] = slurm

    return BiasStudyConfig(**filtered)
