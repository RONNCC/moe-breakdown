#!/usr/bin/env python3
"""Submit a single bias-Shapley study to Slurm.

Usage:
  python3 scripts/submit_slurm_study.py --config configs/study.olmoe.concentration.yaml
  python3 scripts/submit_slurm_study.py --config configs/study.olmoe.concentration.yaml --dry-run
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


def _add_if(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def build_sbatch_command(config_path: Path) -> list[str]:
    cfg = load_bias_study_config(config_path)
    slurm = cfg.slurm

    out_dir = Path(cfg.output_root).expanduser() / cfg.study_name
    logs_dir = out_dir / "slurm-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name=bias-{cfg.study_name}",
        f"--nodes=1",
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
    _add_if(cmd, "--partition", slurm.partition)
    _add_if(cmd, "--account", slurm.account)
    _add_if(cmd, "--qos", slurm.qos)
    cmd.extend(slurm.extra_sbatch_args)

    workdir = str(Path(slurm.workdir).expanduser())
    uv_env_dir = slurm.uv_env_dir or ""

    export_bits = {
        "ALL": None,
        "STUDY_CONFIG": str(config_path.resolve()),
        "OUT_DIR": str(out_dir.resolve()),
        "WORKDIR": workdir,
        "MODULES": " ".join(slurm.modules),
    }
    if uv_env_dir and "$" not in uv_env_dir:
        export_bits["UV_ENV_DIR"] = uv_env_dir
    export_arg = ",".join([k if v is None else f"{k}={v}" for k, v in export_bits.items()])
    cmd.append(f"--export={export_arg}")
    cmd.append(str((ROOT / "slurm" / slurm.sbatch_script).resolve()))
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Submit a bias-Shapley study to Slurm")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cmd = build_sbatch_command(Path(args.config))
    print("[submit]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
