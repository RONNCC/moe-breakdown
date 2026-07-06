"""moe_bias_shapley — Expert-Shapley bias attribution for MoE and dense LLMs."""
from .config import BiasStudyConfig, SlurmConfig, load_bias_study_config
from .benchmarks import PromptPair, load_stereoset, load_bbq
from .hooks import RouterCaptureState, attach_router_hooks, detach_hooks
from .shapley import compute_routing_contrast, compute_dense_layer_contrast
from .metrics import compute_concentration_metrics
from .reporting import save_results, load_results

__all__ = [
    "BiasStudyConfig",
    "SlurmConfig",
    "load_bias_study_config",
    "PromptPair",
    "load_stereoset",
    "load_bbq",
    "RouterCaptureState",
    "attach_router_hooks",
    "detach_hooks",
    "compute_routing_contrast",
    "compute_dense_layer_contrast",
    "compute_concentration_metrics",
    "save_results",
    "load_results",
]
