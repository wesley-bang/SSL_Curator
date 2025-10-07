from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from .registry import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Datasets orchestration CLI")
    parser.add_argument(
        "--data_cfg",
        default="configs/data.yaml",
        help="Path to the global data configuration",
    )
    parser.add_argument(
        "--dataset_cfg",
        required=True,
        help="Path to the specific dataset configuration",
    )
    parser.add_argument(
        "--stage",
        choices=["download", "prepare", "verify", "manifest", "all"],
        default="all",
        help="Which pipeline stage to run",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Subset to build manifests for",
    )
    args = parser.parse_args()

    dataset_cfg = OmegaConf.load(args.dataset_cfg)
    data_cfg = OmegaConf.load(args.data_cfg)

    dataset = build_dataset(dataset_cfg, data_cfg)

    if args.stage in {"download", "all"}:
        dataset.download()
    if args.stage in {"prepare", "all"}:
        dataset.prepare()
    if args.stage in {"verify", "all"}:
        dataset.verify()
    if args.stage in {"manifest", "all"}:
        dataset.build_manifests(split=args.split)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
