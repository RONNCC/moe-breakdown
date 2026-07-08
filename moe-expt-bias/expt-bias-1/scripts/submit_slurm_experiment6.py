#!/usr/bin/env python3
"""Submit Experiment 6 (ladder-wide ablation + controls) jobs to Slurm.

Submits one job per model in the sparsity ladder. Uses each model's existing
Exp1/2 config for GPU resources and derives result_dir from output_root/study_name.

Usage:
  # Submit all four models:
  python3 scripts/submit_slurm_experiment6.py

  # Single model:
  python3 scripts/submit_slurm_experiment6.py --config configs/study.olmoe.concentration.yaml

  # Dry run (print commands, don't submit):
  python3 scripts/submit_slurm_experiment6.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moe_bias_shapley.config import load_bias_study_config  # noqa: E402

# Models to include in the ladder-wide ablation, with tuned max-pairs per model.
# Smaller max-pairs for large models to stay within QOS GPU-min limits.
# gpu_type_override / gpus_override: ICE uses model-name GRES labels (l40s, h100, h200),
# not memory-size labels (40gb, 80gb) as set in the shared configs — override here.
LADDER = [
    # (config_path, max_pairs, routing_freq_pairs, time_limit, gpu_type_override, gpus_override)
    # OLMoE 1B-7B: ~14GB total, 1×L40S (48GB)
    ("configs/study.olmoe.concentration.yaml",         60, 200, "03:00:00", "l40s", 1),
    # Mixtral 8x7B: 4×H100 (config already uses h100, no change needed)
    ("configs/study.mixtral-8x7b.concentration.yaml",  30, 100, "02:00:00", "h100", 4),
    # Phi-3.5-MoE: 42B bf16 ~84GB → 2×H100 (80GB each)
    ("configs/study.phi3.5-moe.concentration.yaml",    30, 100, "04:00:00", "h100", 2),
    # OLMo-7B dense: ~14GB, 1×L40S
    ("configs/study.olmo-7b.dense-baseline.yaml",      60,   0, "02:00:00", "l40s", 1),
]


def submit_one(config_path: str, max_pairs: int, routing_freq_pairs: int, time_limit: str,
               gpu_type_override: str | None, gpus_override: int | None, dry_run: bool) -> None:
    cfg_path = ROOT / config_path
    cfg = load_bias_study_config(cfg_path)
    slurm = cfg.slurm
    if gpu_type_override is not None:
        slurm.gpu_type = gpu_type_override
    if gpus_override is not None:
        slurm.gpus_per_node = gpus_override

    result_dir = Path(cfg.output_root).expanduser() / cfg.study_name
    out_dir = result_dir / "experiment6"
    logs_dir = out_dir / "slurm-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    workdir = str(Path(slurm.workdir).expanduser())
    uv_env_dir = slurm.uv_env_dir or ""

    extra_args = f"--result-dir {result_dir} --max-pairs {max_pairs} --routing-freq-pairs {routing_freq_pairs}"

    export_bits = {
        "ALL": None,
        "STUDY_CONFIG": str(cfg_path.resolve()),
        "SCRIPT": "scripts/run_experiment6_ablation.py",
        "EXTRA_ARGS": extra_args,
        "OUT_DIR": str(out_dir.resolve()),
        "WORKDIR": workdir,
        "MODULES": " ".join(slurm.modules),
    }
    if uv_env_dir and "$" not in uv_env_dir:
        export_bits["UV_ENV_DIR"] = uv_env_dir
    export_arg = ",".join([k if v is None else f"{k}={v}" for k, v in export_bits.items()])

    sbatch_script = ROOT / "slurm" / "run_experiment_script.sbatch"
    job_name = f"bias-exp6-{cfg.study_name}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--nodes=1",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.mem}",
        f"--time={time_limit}",
        f"--output={logs_dir / '%x-%j.out'}",
        f"--error={logs_dir / '%x-%j.err'}",
    ]
    if slurm.use_gres:
        if slurm.gpu_type:
            cmd.append(f"--gres=gpu:{slurm.gpu_type}:{slurm.gpus_per_node}")
        else:
            cmd.append(f"--gres=gpu:{slurm.gpus_per_node}")
    else:
        cmd.append(f"--gpus-per-node={slurm.gpus_per_node}")
    if slurm.partition:
        cmd.extend(["--partition", slurm.partition])
    if slurm.qos:
        cmd.extend(["--qos", slurm.qos])
    cmd.extend(slurm.extra_sbatch_args)
    cmd.append(f"--export={export_arg}")
    cmd.append(str(sbatch_script.resolve()))

    print(f"[submit_exp6] {cfg.study_name} — {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Submit Experiment 6 ablation jobs to Slurm")
    p.add_argument("--config", default=None, help="Submit a single config (default: all ladder models)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    targets = [(cfg, mp, rfp, tl, gt, ng) for cfg, mp, rfp, tl, gt, ng in LADDER
               if args.config is None or Path(cfg).name == Path(args.config).name or cfg == args.config]

    if not targets:
        print(f"No matching configs for --config={args.config}")
        return 1

    for cfg_path, max_pairs, rfp, time_limit, gtype, ngpus in targets:
        submit_one(cfg_path, max_pairs, rfp, time_limit, gtype, ngpus, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
