"""Tests for the CVaR selector behaviour."""

import pandas as pd

from selector.cvar import CvarSelector


def _build_scores(num: int = 5) -> pd.DataFrame:
    rows = []
    for idx in range(num):
        rows.append(
            {
                "utt_id": f"utt_{idx}",
                "path": f"/tmp/{idx}.wav",
                "duration_sec": 30.0 + idx * 5.0,
                "sr": 16000,
                "language": "en",
                "distortion": "noise",
                "loss_proxy": 1.0 - idx * 0.05,
                "ssl_embed": [0.1 + idx * 0.01, 0.2, 0.3],
            }
        )
    return pd.DataFrame(rows)


def test_cvar_selection_monotonicity():
    df = _build_scores()
    small = CvarSelector(K_hours=0.02, alpha=0.1)
    large = CvarSelector(K_hours=0.04, alpha=0.1)

    small_sel = small.select(df)
    large_sel = large.select(df)

    assert len(large_sel) >= len(small_sel)
    assert set(small_sel["utt_id"]).issubset(set(large_sel["utt_id"]))
