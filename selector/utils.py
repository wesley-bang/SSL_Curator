from __future__ import annotations

"""Utility helpers for selection logic (CVaR, normalisation, logging)."""

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


def cvar(values: Sequence[float], alpha: float, higher_is_worse: bool = True) -> float:
    """Compute the Conditional Value at Risk (CVaR) of a sequence.

    Args:
        values: Sequence of numeric values representing losses or scores.
        alpha: Fraction in (0, 1] specifying tail size.
        higher_is_worse: Whether larger numbers represent worse outcomes. Set to
            ``False`` to operate on the lower tail instead of the upper tail.

    Returns:
        Mean of the worst ``ceil(alpha * n)`` values.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute CVaR of an empty sequence")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    k = max(1, int(np.ceil(alpha * arr.size)))
    sorted_arr = np.sort(arr)
    tail = sorted_arr[-k:] if higher_is_worse else sorted_arr[:k]
    return float(np.mean(tail))


def normalize(vec: Sequence[float], eps: float = 1e-8, method: str = "minmax") -> np.ndarray:
    """Normalise a vector using either min-max or z-score scaling."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot normalise an empty sequence")

    if method == "minmax":
        vmin = arr.min()
        vmax = arr.max()
        scale = max(vmax - vmin, eps)
        normed = (arr - vmin) / scale
    elif method == "zscore":
        mean = arr.mean()
        std = max(arr.std(), eps)
        normed = (arr - mean) / std
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

    return normed.astype(np.float32)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a consistent formatter."""
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


@contextmanager
def log_timer(message: str) -> Iterator[None]:
    """Context manager that logs execution time when exiting."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info("%s finished in %.3fs", message, elapsed)


def batched(iterable: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    """Yield successive batches from a sequence."""
    if size <= 0:
        raise ValueError("size must be positive")
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]