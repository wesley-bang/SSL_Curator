from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from selector.utils import cvar


def test_cvar_tail_mean() -> None:
    values = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    assert cvar(values, alpha=0.4) == pytest.approx(3.5)


def test_cvar_monotonicity() -> None:
    values = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    smaller = cvar(values, alpha=0.2, higher_is_worse=False)
    medium = cvar(values, alpha=0.4, higher_is_worse=False)
    larger = cvar(values, alpha=0.8, higher_is_worse=False)
    assert smaller <= medium <= larger
