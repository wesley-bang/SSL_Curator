from __future__ import annotations

"""Generate a manifest CSV from a LibriSpeech-style directory tree."""

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "soundfile is required to generate the manifest. Install it via 'pip install soundfile'."
    ) from exc

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".flac", ".wav", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manifest CSV from LibriSpeech directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Root directory of LibriSpeech (train-clean-100, etc.)")
    parser.add_argument("out", type=Path, help="Output CSV path")
    parser.add_argument(
        "--relative-to",
        type=Path,
        help="If provided, store wav_path relative to this directory (defaults to root)",
    )
    parser.add_argument(
        "--device",
        default="studio_mic",
        help="Device label to populate the manifest",
    )
    parser.add_argument(
        "--room",
        default="studio",
        help="Room/environment label to populate the manifest",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=60.0,
        help="SNR (dB) placeholder to assign to every utterance",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code to assign to every utterance",
    )
    return parser.parse_args()


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def infer_distortion(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "clean" in parts:
        return "clean"
    if "other" in parts or "noisy" in parts:
        return "noisy"
    return "unknown"


def build_row(
    audio_path: Path,
    base_dir: Path,
    device: str,
    room: str,
    lang: str,
    snr_db: float,
) -> Optional[List[str]]:
    try:
        info = sf.info(str(audio_path))
    except RuntimeError as exc:
        LOGGER.warning("Failed to read %s (%s); skipping", audio_path, exc)
        return None

    utt_id = audio_path.stem
    if "-" in utt_id:
        speaker_id = utt_id.split("-")[0]
    else:
        speaker_id = audio_path.parent.name

    rel_path = audio_path.relative_to(base_dir)
    duration_s = info.frames / max(info.samplerate, 1)

    return [
        utt_id,
        str(rel_path).replace("\\", "/"),
        speaker_id,
        lang,
        infer_distortion(audio_path),
        str(info.samplerate),
        f"{snr_db:.2f}",
        device,
        room,
        f"{duration_s:.6f}",
    ]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    base_dir = (args.relative_to or root).resolve()
    rows: List[List[str]] = []

    LOGGER.info("Scanning %s for audio files", root)
    for path in sorted(iter_audio_files(root)):
        row = build_row(path, base_dir=base_dir, device=args.device, room=args.room, lang=args.lang, snr_db=args.snr_db)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No audio files found. Check the root path or supported extensions.")

    header = [
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    LOGGER.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()