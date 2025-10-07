"""Selector CLI dispatcher."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from omegaconf import OmegaConf

# Import selector implementations to populate the registry.
from . import coverage_only, cvar, mos_threshold  # noqa: F401
from .registry import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Selector dispatcher")
    parser.add_argument("--config", required=True, help="Path to selector config YAML")
    parser.add_argument("--scores", required=True, help="Path to scores parquet")
    parser.add_argument("--quotas", help="Optional quotas config path")
    parser.add_argument("--out", required=True, help="Path to write curated CSV")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.quotas:
        cfg.select.quotas = args.quotas

    df = pd.read_parquet(args.scores)
    select_kwargs = dict(OmegaConf.to_container(cfg.select, resolve=True))
    select_name = select_kwargs.pop("name")
    selector = build(select_name, **select_kwargs)
    df_selected = selector.select(df)

    df_selected[["utt_id"]].to_csv(args.out, index=False)
    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scores_path": args.scores,
        "n_selected": int(df_selected.shape[0]),
    }
    with open(args.out.replace(".csv", "_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
