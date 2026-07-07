#!/usr/bin/env python3
"""Submit an Experiment 4 (ablation cross-check) job to Slurm.

Reuses an existing Experiment 1/2 study config for model/GPU parameters,
derives the result_dir from output_root/study_name, and submits via
run_experiment_script.sbatch with run_experiment4_ablation.py as SCRIPT.

Usage:
  python3 scripts/submit_slurm_experiment4.py --config configs/study.olmoe.concentration.yaml
  python3 scripts/submit_slurm_experiment4.py --config configs/study.mixtral-8x7b.concentration.yaml --max-pairs 60
  python3 scripts/submit_slurm_experiment4.py --config configs/study.olmoe.concentration.yaml --dry-run
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Submit Experiment 4 (ablation cross-check) to Slurm")
    p.add_argument("--config", required=True, help="Experiment 1/2 config that produced the result dir")
    p.add_argument("--max-pairs", type=int, default=60,
                   help="Number of prompt pairs for the ablation curve (default: 60)")
    p.add_argument("--result-dir", default=None,
                   help="Override result dir (default: output_root/study_name from config)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    cfg = load_bias_study_config(config_path)
    slurm = cfg.slurm

    result_dir = Path(args.result_dir).expanduser() if args.result_dir else (
        Path(cfg.output_root).expanduser() / cfg.study_name
    )
    out_dir = result_dir / "experiment4"
    logs_dir = out_dir / "slurm-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    workdir = str(Path(slurm.workdir).expanduser())
    uv_env_dir = slurm.uv_env_dir or ""

    extra_args = f"--result-dir {result_dir} --max-pairs {args.max_pairs}"

    export_bits = {
        "ALL": None,
        "STUDY_CONFIG": str(config_path.resolve()),
        "SCRIPT": "scripts/run_experiment4_ablation.py",
        "EXTRA_ARGS": extra_args,
        "OUT_DIR": str(out_dir.resolve()),
        "WORKDIR": workdir,
        "MODULES": " ".join(slurm.modules),
    }
    if uv_env_dir and "$" not in uv_env_dir:
        export_bits["UV_ENV_DIR"] = uv_env_dir
    export_arg = ",".join([k if v is None else f"{k}={v}" for k, v in export_bits.items()])

    sbatch_script = ROOT / "slurm" / "run_experiment_script.sbatch"
    job_name = f"bias-exp4-{cfg.study_name}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--nodes=1",
        f"--cpus-per-task={slurm.cpus_per_task}",
        f"--mem={slurm.mem}",
        f"--time={slurm.time}",
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

    print("[submit_exp4]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
