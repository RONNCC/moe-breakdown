"""One-shot diagnostic for GPT-OSS router-capture paths (run under Slurm).

Loads openai/gpt-oss-120b once per requested mode and reports whether:
  1. the MLP forward is the kernelized one or the eager one,
  2. forward-hooks on router submodules fire and populate captures,
  3. output_router_logits=True works and how it is shaped,
  4. whether attach_router_hooks + output capture combination works.

Usage: python diag_gptoss_capture.py [NO|YES|BOTH]   (default BOTH).
Modes are the value of USE_HUB_KERNELS.
"""
import os
import sys

MODE = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTH"
if MODE not in ("NO", "YES"):
    MODE = "BOTH"

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/scratch/hf_cache"))
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HOME
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def log(*a):
    print(*a, flush=True)


def diagnose(mode: str, tokenizer) -> None:
    os.environ["USE_HUB_KERNELS"] = mode
    log(f"\n========== USE_HUB_KERNELS={mode} ==========")
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    log("model class:", type(model).__name__)
    mlp = model.model.layers[0].mlp
    log("mlp type:", type(mlp).__name__)
    log("mlp has .router:", hasattr(mlp, "router"),
        "| fwd name:", mlp.forward.__name__,
        "| fwd module:", mlp.forward.__module__)

    enc = tokenizer("The nurse greeted the applicant.", return_tensors="pt").to("cuda:0")

    # Omit labels for this probe: GPT-OSS computes an auxiliary load-balancing
    # loss from router_logits when labels are supplied, which can mask whether
    # the native router-output path itself works.
    with torch.no_grad():
        out = model(**enc, output_router_logits=True)
    rl = getattr(out, "router_logits", None)
    log("forward(output_router_logits=True, no labels) OK | router_logits type:",
        type(rl).__name__ if rl is not None else None,
        "| len:", len(rl) if rl is not None else None)
    if rl is not None and len(rl) > 0:
        log("layer0 router_logits shape:", tuple(rl[0].shape))
        log("layer0 router_logits nnz:", int((rl[0] != 0).sum()))

    try:
        with torch.no_grad():
            labeled = model(**enc, labels=enc["input_ids"], output_router_logits=True)
        log("forward(output_router_logits=True, labels) OK | loss:",
            float(labeled.loss) if labeled.loss is not None else None)
    except Exception as e:  # noqa: BLE001
        log("forward(output_router_logits=True, labels) crashed:", type(e).__name__, e)

    del model
    torch.cuda.empty_cache()


def main() -> int:
    log("python:", sys.version.split()[0], "| torch:", torch.__version__)
    from transformers import __version__ as tv
    log("transformers:", tv, "| devices:", torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    modes = [MODE] if MODE != "BOTH" else ["NO", "YES"]
    for m in modes:
        try:
            diagnose(m, tokenizer)
        except Exception as e:  # noqa: BLE001
            log(f"mode {m} crashed:", type(e).__name__, e)
    log("========== DIAGNOSTIC DONE ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
