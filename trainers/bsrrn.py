"""Wrapper around the URGENT 2026 baseline BSRNN trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _ensure_exp_layout(exp_dir: Path) -> None:
    (exp_dir / "train" / "checkpoints").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="BSRNN training wrapper")
    parser.add_argument("--config", required=True, help="Path to base config")
    parser.add_argument("--curated_list", required=True, help="Curated CSV to train on")
    parser.add_argument("--exp.name", dest="exp_name", default=None, help="Override experiment name")
    args, unknown = parser.parse_known_args()

    cfg: DictConfig = OmegaConf.load(args.config)
    if args.exp_name:
        cfg.exp.name = args.exp_name

    exp_root = Path("experiments")
    exp_dir = exp_root / cfg.exp.name
    _ensure_exp_layout(exp_dir)

    config_path = exp_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    baseline_repo = Path(cfg.paths.baseline_repo)
    trainer_entry = baseline_repo / "baseline_code" / "train_se.py"
    train_dir = exp_dir / "train"

    cmd = [
        sys.executable,
        str(trainer_entry),
        "--config_file",
        str(cfg.train.bsrrn_config),
        "--curated_list",
        str(args.curated_list),
        "--exp_dir",
        str(train_dir),
    ]
    cmd.extend(unknown)

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
