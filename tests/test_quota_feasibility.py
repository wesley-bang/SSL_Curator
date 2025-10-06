from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from selector.quotas import SliceQuota
from selector.select import (
    SelectionState,
    coverage_fill,
    group_indices_by_slice,
    prepare_dataframe,
    compute_loss_proxy,
)


def make_dummy_df() -> pd.DataFrame:
    base_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    data = {
        "utt_id": [f"utt{i}" for i in range(6)],
        "wav_path": ["dummy.wav"] * 6,
        "speaker_id": [f"spk{i}" for i in range(6)],
        "lang": ["en", "en", "en", "en", "en", "en"],
        "distortion": ["noise", "noise", "reverb", "reverb", "noise", "reverb"],
        "sr": [16000] * 6,
        "snr_db": [20.0] * 6,
        "device": ["android"] * 6,
        "room": ["studio"] * 6,
        "duration_s": [5.0] * 6,
        "dnsmos": np.linspace(3.0, 4.5, 6),
        "nisqa": np.linspace(3.2, 4.7, 6),
        "utmos": np.linspace(3.1, 4.6, 6),
        "squim_sdr_proxy": np.linspace(7.0, 10.0, 6),
        "ssl_content_vec": base_vectors.tolist(),
        "ssl_speaker_vec": base_vectors[:, ::-1].tolist(),
    }
    return pd.DataFrame(data)


def test_coverage_fill_meets_quotas() -> None:
    df = prepare_dataframe(make_dummy_df())
    compute_loss_proxy(df, weights=[1.0, 1.0, 1.0], uncert_beta=0.0)
    slice_groups = group_indices_by_slice(df)

    state = SelectionState(
        budget_seconds=10_000.0,
        diversity_margin=0.0,
        allow_speaker_repeat=True,
        deny_speakers=set(),
        embeddings=np.asarray(df["ssl_content_vec"].tolist(), dtype=np.float32),
        durations=df["duration_s"].to_numpy(dtype=np.float32),
        speakers=df["speaker_id"].tolist(),
        slice_keys=df["slice_key"].tolist(),
    )

    quotas = {
        "noise|en|16000": SliceQuota("noise", "en", 16000, 2),
        "reverb|en|16000": SliceQuota("reverb", "en", 16000, 2),
    }

    deficits, _ = coverage_fill(quotas, slice_groups, state)

    assert deficits == {}
    assert state.slice_counts["noise|en|16000"] >= 2
    assert state.slice_counts["reverb|en|16000"] >= 2
    assert len(state.selected_indices) >= 4
