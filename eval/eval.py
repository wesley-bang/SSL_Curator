"""Evaluation CLI covering official metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf


def _resolve_ckpt(cfg: DictConfig, override: str | None) -> str:
    if override:
        return override
    if cfg.eval.ckpt:
        return str(cfg.eval.ckpt)
    raise ValueError("Checkpoint path must be provided via --ckpt or eval.ckpt in config.")


def _write_plan(
    out_dir: Path,
    ckpt: str,
    manifest: str,
    curated: str | None,
    metrics: List[Dict[str, Any]],
) -> None:
    plan = {
        "ckpt": ckpt,
        "manifest": manifest,
        "curated": curated,
        "metrics": metrics,
    }
    plan_path = out_dir / "plan.json"
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    logging.info("Wrote eval plan to %s", plan_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation stub CLI")
    parser.add_argument("--config", required=True, help="Path to evaluation config")
    parser.add_argument("--ckpt", help="Checkpoint to evaluate (overrides config)")
    args = parser.parse_args()

    cfg: DictConfig = OmegaConf.load(args.config)
    ckpt = _resolve_ckpt(cfg, args.ckpt)

    out_dir = Path(cfg.eval.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = out_dir / "eval_config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)

    manifest = str(cfg.eval.data.manifest)
    curated = cfg.eval.data.get("curated")
    metrics: List[Dict[str, Any]] = OmegaConf.to_container(cfg.eval.metrics, resolve=True)  # type: ignore[arg-type]

    _write_plan(out_dir, ckpt, manifest, curated, metrics)

    for metric in metrics:
        script = metric.get("script")
        name = metric.get("name")
        logging.info("Would run %s via %s", name, script)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
