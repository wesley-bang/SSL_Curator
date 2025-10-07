"""Scoring CLI that converts manifests to scores parquet."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator

import pandas as pd
from omegaconf import DictConfig, OmegaConf

REQUIRED_COLUMNS = [
    "utt_id",
    "path",
    "duration_sec",
    "sr",
    "language",
    "distortion",
    "loss_proxy",
    "ssl_embed",
]


def _iter_manifest(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _create_row(record: Dict[str, object]) -> Dict[str, object]:
    return {
        "utt_id": record["utt_id"],
        "path": record["path"],
        "duration_sec": float(record["duration_sec"]),
        "sr": int(record.get("sr", 16000)),
        "language": record.get("language", "unknown"),
        "distortion": record.get("distortion", "unknown"),
        "loss_proxy": float(record.get("loss_proxy", 0.0)),
        "ssl_embed": record.get("ssl_embed", [0.0, 0.0, 0.0]),
    }


def build_dataframe(manifest_path: Path) -> pd.DataFrame:
    rows = [_create_row(rec) for rec in _iter_manifest(manifest_path)]
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring entrypoint")
    parser.add_argument("--config", required=True, help="Path to scoring config")
    parser.add_argument("--manifest", help="Path to manifest JSONL override")
    parser.add_argument("--out", required=True, help="Output parquet path")
    args, extras = parser.parse_known_args()

    manifest_override = args.manifest
    for item in extras:
        if item.startswith("data.manifest="):
            manifest_override = item.split("=", 1)[1]
            break

    cfg: DictConfig = OmegaConf.load(args.config)
    if manifest_override:
        manifest_path = Path(manifest_override).expanduser()
    else:
        manifest_cfg = cfg.get("data", {}).get("manifest")
        if not manifest_cfg:
            raise ValueError("Manifest path not provided via --manifest or data.manifest override.")
        manifest_path = Path(manifest_cfg).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = build_dataframe(manifest_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logging.info("Wrote %s rows to %s", df.shape[0], args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
