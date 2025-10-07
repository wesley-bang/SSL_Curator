"""Coherent CVaR-based selector implementation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .base import BaseSelector
from .registry import register_selector


def _compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


@register_selector("cvar")
class CvarSelector(BaseSelector):
    """Naive CVaR selector with diversity gating."""

    def __init__(
        self,
        K_hours: float,
        alpha: float,
        quotas: Optional[Dict[str, Any]] = None,
        diversity: Optional[Dict[str, Any]] = None,
        penalties: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            quotas=quotas,
            diversity=diversity,
            penalties=penalties,
            filters=filters,
            **kwargs,
        )
        self.K_hours = float(K_hours)
        self.alpha = float(alpha)
        self.quotas = quotas or {}
        self.diversity = diversity or {"min_cosine": 0.15}
        self.penalties = penalties or {"quota_violation": 10.0}
        self.filters = filters or {}

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df
        mapping = {
            "language": "languages",
            "sr": "sample_rates",
            "distortion": "distortions",
        }
        for column, key in mapping.items():
            allowed = self.filters.get(key)
            if allowed:
                filtered = filtered[filtered[column].isin(allowed)]
        return filtered

    def _within_diversity(self, row: pd.Series, picked: List[pd.Series]) -> bool:
        if not picked:
            return True
        min_drop = 1.0 - float(self.diversity.get("min_cosine", 0.15))
        current = np.asarray(row["ssl_embed"], dtype=float)
        for candidate in picked[-50:]:
            ref = np.asarray(candidate["ssl_embed"], dtype=float)
            cos = _compute_cosine(current, ref)
            if cos > 1.0 - min_drop:
                return False
        return True

    def _update_hours(self, subset: List[pd.Series]) -> float:
        return float(sum(r["hours"] for r in subset))

    def select(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        df = self._apply_filters(df_scores.copy())
        if df.empty:
            return df

        df["hours"] = df["duration_sec"] / 3600.0
        df = df.sort_values("loss_proxy", ascending=False)

        picked: List[pd.Series] = []
        total_hours = 0.0
        for _, row in df.iterrows():
            if total_hours >= self.K_hours:
                break
            if not isinstance(row.get("ssl_embed"), Iterable):
                continue
            if self._within_diversity(row, picked):
                picked.append(row)
                total_hours += row["hours"]

        return pd.DataFrame(picked).reset_index(drop=True)


__all__ = ["CvarSelector"]
