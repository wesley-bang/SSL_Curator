"""Wrapper around the URGENT 2026 baseline FlowSE trainer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import logging


def _ensure_layout(exp_dir: Path) -> None:
    (exp_dir / "train" / "checkpoints").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="FlowSE training wrapper")
    parser.add_argument("--config", required=True, help="Path to base config")
    parser.add_argument("--curated_list", required=True, help="Curated CSV to train on")
    parser.add_argument("--exp.name", dest="exp_name", default=None, help="Override experiment name")
    args, unknown = parser.parse_known_args()

    cfg: DictConfig = OmegaConf.load(args.config)
    if args.exp_name:
        cfg.exp.name = args.exp_name

    exp_dir = Path("experiments") / cfg.exp.name
    _ensure_layout(exp_dir)

    config_snapshot = exp_dir / "config.yaml"
    OmegaConf.save(cfg, config_snapshot)

    baseline_repo = Path(cfg.paths.baseline_repo)
    entry_script = baseline_repo / cfg.train.flowse_entry
    train_dir = exp_dir / "train"

    if not entry_script.exists():
        logging.warning("FlowSE entry script %s not found, skipping subprocess call.", entry_script)
        return

    cmd = [
        sys.executable,
        str(entry_script),
        "--config_file",
        str(cfg.train.flowse_config),
        "--curated_list",
        str(args.curated_list),
        "--exp_dir",
        str(train_dir),
    ]
    cmd.extend(unknown)

    subprocess.check_call(cmd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
