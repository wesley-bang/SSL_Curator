"""Reproducibility tests for selector pipeline."""

import pandas as pd

from selector.cvar import CvarSelector


def test_reproducibility_seed_control():
    df = pd.DataFrame(
        [
            {
                "utt_id": "a",
                "path": "a.wav",
                "duration_sec": 10.0,
                "sr": 16000,
                "language": "en",
                "distortion": "noise",
                "loss_proxy": 0.9,
                "ssl_embed": [0.1, 0.2, 0.3],
            },
            {
                "utt_id": "b",
                "path": "b.wav",
                "duration_sec": 12.0,
                "sr": 16000,
                "language": "en",
                "distortion": "reverb",
                "loss_proxy": 0.8,
                "ssl_embed": [0.3, 0.2, 0.1],
            },
        ]
    )

    selector = CvarSelector(K_hours=0.01, alpha=0.1)
    first = selector.select(df)
    second = selector.select(df)

    pd.testing.assert_frame_equal(first, second)
