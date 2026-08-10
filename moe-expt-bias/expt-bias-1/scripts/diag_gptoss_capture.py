"""One-shot diagnostic for GPT-OSS router-capture paths (run under Slurm).

Loads openai/gpt-oss-120b once per requested mode and reports whether:
  1. the MLP forward is the kernelized one or the eager one,
  2. forward-hooks on router submodules fire and populate captures,
  3. output_router_logits=True works and how it is shaped,
  4. whether attach_router_hooks + output capture combination works,
  5. (EAGER) whether force_eager_gpt_oss rebinds the fused MXFP4 mlp.forward
     back to the eager GptOssMLP.forward so the study's output_router_logits
     capture path works (router_logits len=36, layer0=(n_tokens, 128), nnz>0).

Usage: python diag_gptoss_capture.py [NO|YES|EAGER|BOTH]   (default BOTH).
Modes are the value of USE_HUB_KERNELS; EAGER uses YES (buggy kernel state)
then force_eager_gpt_oss to rebind.
"""
import os
import sys
from pathlib import Path

MODE = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTH"
if MODE not in ("NO", "YES", "EAGER"):
    MODE = "BOTH"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def diagnose_eager(tokenizer) -> None:
    from moe_bias_shapley.modeling import force_eager_gpt_oss

    # Reproduce the buggy state the study hits: USE_HUB_KERNELS=YES + the auto
    # MXFP4 load attach the fused kernel forward to every GptOssMLP; then
    # force_eager_gpt_oss rebinds to the eager GptOssMLP.forward.
    os.environ["USE_HUB_KERNELS"] = "YES"
    log("\n========== EAGER (force_eager_gpt_oss) ==========")
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-120b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        log("model class:", type(model).__name__)
        mlp = model.model.layers[0].mlp
        log("pre-patch  mlp fwd:", getattr(mlp.forward, "__module__", "?"),
            "/", getattr(mlp.forward, "__name__", "?"),
            "| experts class:", type(getattr(mlp, "experts", None)).__name__)
        n_rebound = force_eager_gpt_oss(model)
        log("force_eager_gpt_oss rebound:", n_rebound)
        log("post-patch mlp fwd:", getattr(mlp.forward, "__module__", "?"),
            "/", getattr(mlp.forward, "__name__", "?"))

        enc = tokenizer("The nurse greeted the applicant.", return_tensors="pt").to("cuda:0")
        n_tokens = int(enc["input_ids"].shape[1])
        log("capture probe: n_tokens =", n_tokens)

        # Exact study capture path (shapley._with_router_capture, router_capture=
        # "outputs"): teacher-forced forward with output_router_logits=True, NO labels.
        with torch.no_grad():
            out = model(**enc, output_router_logits=True)
        rl = getattr(out, "router_logits", None)
        log("forward(output_router_logits=True, no labels) OK | router_logits type:",
            type(rl).__name__ if rl is not None else None,
            "| len:", len(rl) if rl is not None else None)
        if rl is None or len(rl) == 0:
            log("CAPTURE FAIL: router_logits empty — eager router still not executing")
            return
        ok_len = len(rl) == 36
        layer0 = rl[0]
        shape = tuple(layer0.shape)
        nnz = int((layer0 != 0).sum()) if layer0 is not None else 0
        ok_shape = shape == (n_tokens, 128)
        ok_nnz = nnz > 0
        log("layer0 router_logits shape:", shape, "| nnz:", nnz)
        log("checks: len(rl)==36:", ok_len, "| layer0 == (n_tokens, 128):", ok_shape,
            "| nnz>0:", ok_nnz)
        if ok_len and ok_shape and ok_nnz:
            log("EAGER CAPTURE OK — study output_router_logits path will populate state.captured")
        else:
            log("EAGER CAPTURE PARTIAL/FAIL: ok_len=%s ok_shape=%s ok_nnz=%s", ok_len, ok_shape, ok_nnz)
    except Exception as e:  # noqa: BLE001
        log("EAGER capture path crashed:", type(e).__name__, e)
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> int:
    log("python:", sys.version.split()[0], "| torch:", torch.__version__)
    from transformers import __version__ as tv
    log("transformers:", tv, "| devices:", torch.cuda.device_count())
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    modes = {"NO": ["NO"], "YES": ["YES"], "EAGER": ["EAGER"], "BOTH": ["NO", "YES"]}[MODE]
    for m in modes:
        try:
            if m == "EAGER":
                diagnose_eager(tokenizer)
            else:
                diagnose(m, tokenizer)
        except Exception as e:  # noqa: BLE001
            log(f"mode {m} crashed:", type(e).__name__, e)
    log("========== DIAGNOSTIC DONE ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
