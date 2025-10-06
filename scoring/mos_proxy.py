from __future__ import annotations

"""Deterministic MOS proxy scorers for the scoring pipeline."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _clip_unit(value: float) -> float:
    """Clamp a scalar into the [0, 1] range."""
    return float(np.clip(value, 0.0, 1.0))


def basic_audio_stats(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Compute lightweight deterministic audio statistics.

    Args:
        waveform: 1-D float32 waveform array.
        sample_rate: Sampling rate in Hz.

    Returns:
        Mapping of statistic name to scalar value (some normalised to [0, 1]).
    """
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional")

    wf = waveform.astype(np.float32)
    wf = np.clip(wf, -1.0, 1.0)

    rms = float(np.sqrt(np.mean(np.square(wf)) + 1e-10))
    peak = float(np.max(np.abs(wf)))
    crest = peak / (rms + 1e-8)

    zc = float(np.mean(np.abs(np.diff(np.signbit(wf)))))

    n_fft = int(min(4096, len(wf)))
    if n_fft % 2 == 1:
        n_fft += 1
    spectrum = np.abs(np.fft.rfft(wf, n=n_fft)) + 1e-8
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    centroid = float((freqs * spectrum).sum() / (spectrum.sum() + 1e-8))
    centroid_norm = centroid / max(sample_rate / 2.0, 1.0)

    log_spec = np.log(spectrum)
    spectral_flatness = float(np.exp(log_spec.mean()) / (spectrum.mean() + 1e-8))

    abs_wf = np.abs(wf)
    dynamic_range = float(np.percentile(abs_wf, 95) - np.percentile(abs_wf, 5))

    stats = {
        "rms": rms,
        "rms_norm": _clip_unit(rms / 0.2),
        "peak": peak,
        "crest": crest,
        "crest_norm": _clip_unit((crest - 1.0) / 9.0),
        "zcr": _clip_unit(zc),
        "flatness": _clip_unit(spectral_flatness),
        "bandwidth_norm": _clip_unit(dynamic_range / 0.8),
        "centroid_norm": _clip_unit(centroid_norm),
    }
    stats["clarity_norm"] = _clip_unit(1.0 - stats["flatness"])
    stats["low_centroid_bonus"] = _clip_unit(1.0 - abs(stats["centroid_norm"] - 0.35) / 0.35)
    stats["dynamic_range_norm"] = _clip_unit(dynamic_range / 0.5)

    return stats


@dataclass
class MOSProxy:
    """Callable deterministic MOS proxy for a given method name."""

    method: str

    def __post_init__(self) -> None:
        self.method = self.method.lower()
        self._validate()

    def _validate(self) -> None:
        if self.method not in {"dnsmos", "nisqa", "utmos"}:
            raise ValueError(f"Unsupported MOS proxy method: {self.method}")

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Compute the MOS proxy score.

        Args:
            waveform: 1-D array of audio samples in [-1, 1].
            sample_rate: Sampling rate.

        Returns:
            Pseudo-MOS score in the [1, 5] range.
        """
        stats = basic_audio_stats(waveform, sample_rate)
        if self.method == "dnsmos":
            value = (
                0.55 * stats["rms_norm"]
                + 0.25 * stats["clarity_norm"]
                + 0.20 * (1.0 - stats["zcr"])
            )
        elif self.method == "nisqa":
            value = (
                0.40 * stats["dynamic_range_norm"]
                + 0.35 * stats["clarity_norm"]
                + 0.25 * stats["low_centroid_bonus"]
            )
        else:  # utmos
            value = (
                0.50 * stats["rms_norm"]
                + 0.30 * stats["low_centroid_bonus"]
                + 0.20 * (1.0 - stats["crest_norm"])
            )

        return float(1.0 + 4.0 * _clip_unit(value))


def compute_mos_proxies(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Compute all supported MOS proxies at once.

    Args:
        waveform: Audio samples.
        sample_rate: Sampling rate.

    Returns:
        Dict mapping method name to MOS score.
    """
    return {
        name: MOSProxy(name)(waveform, sample_rate)
        for name in ("dnsmos", "nisqa", "utmos")
    }