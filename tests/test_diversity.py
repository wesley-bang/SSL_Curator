from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from selector.diversity import KCenterResult, k_center_farthest


def test_k_center_rejects_near_duplicates() -> None:
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = base + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vectors = np.stack([base, near, far])

    result: KCenterResult = k_center_farthest(vectors, k=2, min_cos_margin=0.05)

    assert set(result.selected) == {0, 2}
    assert result.rejections >= 1
