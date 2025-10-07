"""Quota feasibility tests."""

import pandas as pd

from selector.cvar import CvarSelector


def test_quota_feasibility_no_violation():
    df = pd.DataFrame(
        [
            {
                "utt_id": "x",
                "path": "x.wav",
                "duration_sec": 60.0,
                "sr": 16000,
                "language": "en",
                "distortion": "noise",
                "loss_proxy": 0.95,
                "ssl_embed": [0.2, 0.1, 0.3],
            },
            {
                "utt_id": "y",
                "path": "y.wav",
                "duration_sec": 55.0,
                "sr": 16000,
                "language": "en",
                "distortion": "reverb",
                "loss_proxy": 0.9,
                "ssl_embed": [0.3, 0.3, 0.3],
            },
        ]
    )

    quotas = {"distortion": {"noise": 0.5, "reverb": 0.5}}
    selector = CvarSelector(K_hours=0.04, alpha=0.1, quotas=quotas)

    selected = selector.select(df)
    assert not selected.empty
    assert selected["duration_sec"].sum() <= 0.04 * 3600 + 1e-6
