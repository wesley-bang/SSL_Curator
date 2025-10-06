from __future__ import annotations

"""Quota loading and slice utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import yaml

SLICE_DELIMITER = "|"


@dataclass(frozen=True)
class SliceQuota:
    distortion: str
    lang: str
    sr: int
    minimum: int

    @property
    def key(self) -> str:
        return make_slice_key(self.distortion, self.lang, self.sr)


def make_slice_key(distortion: str, lang: str, sr: int) -> str:
    return f"{distortion}{SLICE_DELIMITER}{lang}{SLICE_DELIMITER}{int(sr)}"


def parse_slice_key(key: str) -> Tuple[str, str, int]:
    parts = key.split(SLICE_DELIMITER)
    if len(parts) != 3:
        raise ValueError(f"Invalid slice key '{key}', expected 3 parts")
    distortion, lang, sr_str = parts
    return distortion, lang, int(sr_str)


def load_quotas(path: Path) -> Dict[str, SliceQuota]:
    """Load slice quotas from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Quota file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    quotas: Dict[str, SliceQuota] = {}
    for key, value in data.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Quota for {key} must be a non-negative integer")
        distortion, lang, sr = parse_slice_key(str(key))
        quotas[str(key)] = SliceQuota(distortion, lang, sr, int(value))
    return quotas


def compute_slice_keys(df: pd.DataFrame) -> pd.Series:
    """Compute slice keys for each row of the dataframe."""
    required = {"distortion", "lang", "sr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    return (
        df["distortion"].astype(str)
        + SLICE_DELIMITER
        + df["lang"].astype(str)
        + SLICE_DELIMITER
        + df["sr"].astype(int).astype(str)
    )


def assess_quota_feasibility(
    df: pd.DataFrame,
    quotas: Dict[str, SliceQuota],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """Return per-slice pools and deficits relative to quotas."""
    if "slice_key" not in df.columns:
        df = df.assign(slice_key=compute_slice_keys(df))

    pools: Dict[str, pd.DataFrame] = {}
    deficits: Dict[str, int] = {}
    for key, quota in quotas.items():
        slice_df = df[df["slice_key"] == key]
        pools[key] = slice_df
        deficit = max(0, quota.minimum - len(slice_df))
        if deficit > 0:
            deficits[key] = deficit
    return pools, deficits


def summarise_quota_deficits(deficits: Dict[str, int]) -> str:
    if not deficits:
        return "All quotas feasible"
    parts = [f"{key}:-{value}" for key, value in sorted(deficits.items())]
    return ", ".join(parts)