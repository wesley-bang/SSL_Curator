from __future__ import annotations

"""Simple evaluation utilities for curated subsets."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from selector.utils import normalize

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute proxy summary metrics for a curated selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True, help="scores.parquet path")
    parser.add_argument("--selected", type=Path, required=True, help="curated CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report output path")
    return parser.parse_args()


def load_data(scores_path: Path, selected_path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    selected = pd.read_csv(selected_path)
    merged = selected.merge(scores, on="utt_id", suffixes=("_sel", "_score"), how="left")
    return merged


def pick_series(df: pd.DataFrame, base: str) -> pd.Series:
    for candidate in (base, f"{base}_sel", f"{base}_score"):
        if candidate in df.columns:
            return df[candidate]
    raise KeyError(f"Column '{base}' not found in dataframe")


def compute_tail_hardness(series: pd.Series) -> Tuple[float, float]:
    norm = normalize(series.to_numpy(dtype=np.float32))
    hardness = 1.0 - norm
    p10 = float(np.percentile(hardness, 10))
    p25 = float(np.percentile(hardness, 25))
    return p10, p25


def generate_report(df: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    proxies = ["dnsmos", "nisqa", "squim_sdr_proxy"]
    overall_means = {proxy: float(pick_series(df, proxy).mean()) for proxy in proxies}
    tails = {proxy: compute_tail_hardness(pick_series(df, proxy)) for proxy in proxies}

    slice_counts = (
        df.groupby("slice_key")
        .agg(selected_count=("utt_id", "count"), avg_loss=("loss_proxy", "mean"))
        .reset_index()
    )
    slice_counts["avg_loss"] = slice_counts["avg_loss"].astype(float)

    metrics["overall_means"] = overall_means
    metrics["tails"] = tails
    metrics["slice_counts"] = slice_counts
    return metrics


def format_markdown(metrics: Dict[str, object]) -> str:
    lines = ["# Curated Selection Report", ""]

    lines.append("## Overall Proxy Means")
    lines.append("| Proxy | Mean |")
    lines.append("|-------|------|")
    for key, value in metrics["overall_means"].items():
        lines.append(f"| {key} | {value:.4f} |")
    lines.append("")

    lines.append("## Tail Hardness (1 - proxy normalization)")
    lines.append("| Proxy | p10 | p25 |")
    lines.append("|-------|-----|------|")
    for key, (p10, p25) in metrics["tails"].items():
        lines.append(f"| {key} | {p10:.4f} | {p25:.4f} |")
    lines.append("")

    lines.append("## Per-Slice Coverage")
    lines.append("| Slice Key | Selected | Avg Loss |")
    lines.append("|-----------|----------|----------|")
    slice_df: pd.DataFrame = metrics["slice_counts"]  # type: ignore[assignment]
    for row in slice_df.itertuples(index=False):
        lines.append(f"| {row.slice_key} | {row.selected_count} | {row.avg_loss:.4f} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    df = load_data(args.scores, args.selected)
    if df.empty:
        raise ValueError("Selected dataframe is empty; nothing to evaluate")

    metrics = generate_report(df)
    report = format_markdown(metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    LOGGER.info("Wrote evaluation report to %s", args.out)
    print(report)


if __name__ == "__main__":
    main()