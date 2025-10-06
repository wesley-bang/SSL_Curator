from __future__ import annotations

"""Compute deterministic proxy scores and embeddings for a manifest."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .mos_proxy import basic_audio_stats, compute_mos_proxies
from .ssl_embed import compute_ssl_embeddings

try:  # Optional heavy deps
    import soundfile as sf
except Exception:  # pragma: no cover - when dependency missing
    sf = None

try:  # pragma: no cover - optional
    import librosa
except Exception:  # pragma: no cover - when dependency missing
    librosa = None

LOGGER = logging.getLogger(__name__)


MANIFEST_COLUMNS = [
    "utt_id",
    "wav_path",
    "speaker_id",
    "lang",
    "distortion",
    "sr",
    "snr_db",
    "device",
    "room",
    "duration_s",
]

DISTORTIONS = [
    "reverb",
    "noise",
    "clipping",
    "codec",
    "bandlimit",
    "babble",
    "low_snr",
]

LANGUAGES = ["en", "zh", "ja"]
SAMPLE_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
DEVICES = ["android", "ios", "pc", "tablet", "smart_speaker"]
ROOMS = ["studio", "office", "hall", "car", "kitchen", "lab"]
SPEAKERS = [f"spk{i:03d}" for i in range(1, 181)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute proxy scores, embeddings, and metadata for a manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    parser.add_argument(
        "--ssl",
        type=str,
        default="hubert_base",
        help="SSL model identifier from s3prl.hub (fallback to deterministic)",
    )
    parser.add_argument(
        "--layers-content",
        nargs="+",
        type=int,
        default=[9, 10, 11, 12],
        help="Layer indices to pool for content embeddings",
    )
    parser.add_argument(
        "--layers-speaker",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6],
        help="Layer indices to pool for speaker embeddings",
    )
    parser.add_argument(
        "--mos",
        choices=["dnsmos", "nisqa", "utmos"],
        default="dnsmos",
        help="Primary MOS proxy focus (all proxies are computed regardless)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed")
    parser.add_argument(
        "--regenerate-manifest",
        action="store_true",
        help="Force regeneration of the toy manifest if present",
    )
    return parser.parse_args()


def ensure_manifest(manifest_path: Path, seed: int, force: bool = False) -> None:
    """Create the toy manifest if it does not exist."""
    if manifest_path.exists() and not force:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_toy_manifest(seed=seed)
    df.to_csv(manifest_path, index=False)
    LOGGER.info("Generated toy manifest with %d rows at %s", len(df), manifest_path)


def build_toy_manifest(seed: int, total_rows: int = 120) -> pd.DataFrame:
    """Construct a deterministic toy manifest covering many slices."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    quota_slices: List[Tuple[str, str, int]] = [
        ("reverb", "en", 16000),
        ("noise", "en", 16000),
        ("codec", "zh", 16000),
        ("bandlimit", "ja", 8000),
        ("low_snr", "en", 48000),
        ("clipping", "ja", 22050),
        ("codec", "en", 44100),
        ("bandlimit", "zh", 16000),
        ("babble", "en", 24000),
        ("reverb", "zh", 32000),
        ("noise", "ja", 16000),
        ("low_snr", "zh", 48000),
        ("babble", "ja", 48000),
        ("clipping", "en", 16000),
        ("noise", "zh", 24000),
        ("codec", "ja", 32000),
    ]

    def add_row(idx: int, distortion: str, lang: str, sr: int) -> Dict[str, object]:
        speaker = SPEAKERS[idx % len(SPEAKERS)]
        duration = float(rng.uniform(3.0, 10.0))
        snr = float(rng.uniform(-3.0, 5.0)) if distortion == "low_snr" else float(rng.uniform(8.0, 35.0))
        device = rng.choice(DEVICES)
        room = rng.choice(ROOMS)
        utt_id = f"{distortion}_{lang}_{sr}_{idx:03d}"
        wav_rel = Path("data") / "audio" / f"{utt_id}.wav"
        return {
            "utt_id": utt_id,
            "wav_path": str(wav_rel),
            "speaker_id": speaker,
            "lang": lang,
            "distortion": distortion,
            "sr": int(sr),
            "snr_db": round(snr, 2),
            "device": device,
            "room": room,
            "duration_s": round(duration, 3),
        }

    idx = 0
    for distortion, lang, sr in quota_slices:
        for _ in range(6):
            rows.append(add_row(idx, distortion, lang, sr))
            idx += 1

    while len(rows) < total_rows:
        distortion = rng.choice(DISTORTIONS)
        lang = rng.choice(LANGUAGES)
        sr = int(rng.choice(SAMPLE_RATES))
        rows.append(add_row(idx, distortion, lang, sr))
        idx += 1

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df


def synthesise_waveform(
    utt_id: str,
    duration_s: float,
    sample_rate: int,
    snr_db: float,
    distortion: str,
) -> np.ndarray:
    """Synthesize a deterministic waveform for the given utterance."""
    sr = int(sample_rate)
    num_samples = max(1, int(round(duration_s * sr)))
    base_seed = abs(hash((utt_id, "base"))) % (2**32)
    rng = np.random.default_rng(base_seed)
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)

    freq1 = rng.uniform(120.0, 1200.0)
    freq2 = freq1 * rng.uniform(1.5, 3.0)
    freq3 = rng.uniform(40.0, 180.0)

    waveform = (
        0.6 * np.sin(2 * np.pi * freq1 * t + rng.uniform(0, 2 * np.pi))
        + 0.3 * np.sin(2 * np.pi * freq2 * t)
        + 0.1 * np.sin(2 * np.pi * freq3 * t)
    ).astype(np.float32)

    waveform = apply_distortion(waveform, distortion, rng, sr)

    signal_rms = float(np.sqrt(np.mean(np.square(waveform)) + 1e-10))
    if np.isfinite(snr_db):
        snr_ratio = 10 ** (snr_db / 20.0)
        noise_rms = signal_rms / max(snr_ratio, 1e-2)
    else:
        noise_rms = 0.01
    noise = rng.normal(0.0, noise_rms, size=waveform.shape).astype(np.float32)
    waveform = waveform + noise
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform.astype(np.float32)


def apply_distortion(
    waveform: np.ndarray,
    distortion: str,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """Apply lightweight synthetic distortions to the waveform."""
    wf = waveform.copy()
    if distortion == "reverb":
        tail = np.exp(-np.linspace(0.0, 3.0, num=256, dtype=np.float32))
        tail += 0.1 * rng.standard_normal(tail.shape).astype(np.float32)
        wf = np.convolve(wf, tail, mode="same")
    elif distortion == "noise":
        wf += 0.03 * rng.standard_normal(wf.shape).astype(np.float32)
    elif distortion == "clipping":
        threshold = rng.uniform(0.25, 0.45)
        wf = np.clip(wf, -threshold, threshold)
    elif distortion == "codec":
        down_factor = 2
        wf = wf[::down_factor]
        kernel = np.ones(down_factor, dtype=np.float32) / down_factor
        wf = np.convolve(wf, kernel, mode="full")
        wf = np.interp(np.linspace(0, len(wf) - 1, len(waveform)), np.arange(len(wf)), wf)
    elif distortion == "bandlimit":
        kernel_size = 33
        window = np.hanning(kernel_size).astype(np.float32)
        window /= window.sum()
        wf = np.convolve(wf, window, mode="same")
    elif distortion == "babble":
        babble = sum(
            rng.normal(0.0, 0.05, size=wf.shape).astype(np.float32)
            for _ in range(3)
        )
        wf = 0.6 * wf + 0.4 * babble
    elif distortion == "low_snr":
        wf += 0.05 * rng.standard_normal(wf.shape).astype(np.float32)

    wf = np.clip(wf, -1.0, 1.0)
    return wf.astype(np.float32)


def load_waveform(
    path: Path,
    duration_s: float,
    sample_rate: int,
    utt_id: str,
    snr_db: float,
    distortion: str,
) -> np.ndarray:
    """Load or synthesise a waveform for the manifest row."""
    if path.exists() and sf is not None:
        try:  # pragma: no cover - dependent on local audio
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != sample_rate:
                if librosa is None:
                    raise RuntimeError("librosa not available for resampling")
                data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
            return data.astype(np.float32)
        except Exception as exc:
            LOGGER.debug("Falling back to synthetic audio for %s: %s", path, exc)

    return synthesise_waveform(utt_id, duration_s, sample_rate, snr_db, distortion)


def compute_squim_proxy(waveform: np.ndarray, sample_rate: int) -> float:
    """Compute a heuristic SQUIM/SDR style proxy (higher is better)."""
    stats = basic_audio_stats(waveform, sample_rate)
    value = (
        10.0 * stats["rms_norm"]
        + 8.0 * stats["clarity_norm"]
        + 6.0 * stats["dynamic_range_norm"]
        - 5.0 * stats["zcr"]
    )
    return float(value)


def run(args: argparse.Namespace) -> pd.DataFrame:
    ensure_manifest(args.manifest, args.seed, force=args.regenerate_manifest)

    df_manifest = pd.read_csv(args.manifest)
    missing = [col for col in MANIFEST_COLUMNS if col not in df_manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    results: List[Dict[str, object]] = []

    tqdm_iter = tqdm(df_manifest.itertuples(index=False), total=len(df_manifest), desc="scoring")

    for row in tqdm_iter:
        wav_path = Path(row.wav_path)
        waveform = load_waveform(
            path=wav_path,
            duration_s=float(row.duration_s),
            sample_rate=int(row.sr),
            utt_id=str(row.utt_id),
            snr_db=float(row.snr_db),
            distortion=str(row.distortion),
        )

        mos_scores = compute_mos_proxies(waveform, int(row.sr))
        squim = compute_squim_proxy(waveform, int(row.sr))
        content_vec, speaker_vec = compute_ssl_embeddings(
            waveform=waveform,
            sample_rate=int(row.sr),
            model_name=args.ssl,
            content_layers=tuple(args.layers_content),
            speaker_layers=tuple(args.layers_speaker),
            utt_id=str(row.utt_id),
        )

        proxy_values = np.array(list(mos_scores.values()), dtype=np.float32)
        sigma = float(np.std(proxy_values))

        record: Dict[str, object] = {
            "utt_id": row.utt_id,
            "wav_path": str(wav_path),
            "speaker_id": row.speaker_id,
            "lang": row.lang,
            "distortion": row.distortion,
            "sr": int(row.sr),
            "snr_db": float(row.snr_db),
            "device": row.device,
            "room": row.room,
            "duration_s": float(row.duration_s),
            "dnsmos": mos_scores["dnsmos"],
            "nisqa": mos_scores["nisqa"],
            "utmos": mos_scores["utmos"],
            "squim_sdr_proxy": squim,
            "proxy_sigma": sigma,
            "ssl_content_vec": content_vec.tolist(),
            "ssl_speaker_vec": speaker_vec.tolist(),
        }
        results.append(record)

    df_scores = pd.DataFrame(results)
    return df_scores


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    df_scores = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_parquet(args.out, index=False)
    LOGGER.info("Wrote %d scored utterances to %s", len(df_scores), args.out)


if __name__ == "__main__":
    main()






