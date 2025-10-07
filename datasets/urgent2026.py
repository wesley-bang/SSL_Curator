from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import glob
import os
import subprocess

import soundfile as sf

from .base import BaseDataset

_LANGUAGE_TAGS: Tuple[str, ...] = ("en", "zh", "ja", "de", "fr", "es", "it", "ko", "ru")
_DISTORTION_TAGS: Tuple[str, ...] = (
    "reverb",
    "noise",
    "clip",
    "codec",
    "packetloss",
    "bandlimit",
    "device",
    "clean",
)


def _iter_wavs(pattern: str) -> Iterable[str]:
    for match in glob.glob(pattern, recursive=True):
        if match.lower().endswith(".wav"):
            yield match


def _infer_label(path: str, keys: Tuple[str, ...]) -> Dict[str, str]:
    normalized = path.replace("\\", "/").lower()
    labels: Dict[str, str] = {}
    if "language" in keys:
        for lang in _LANGUAGE_TAGS:
            if f"/{lang}/" in normalized or f"_{lang}_" in normalized:
                labels["language"] = lang
                break
    if "distortion" in keys:
        for distortion in _DISTORTION_TAGS:
            if f"/{distortion}/" in normalized or f"_{distortion}_" in normalized:
                labels["distortion"] = distortion
                break
    return labels


class Urgent2026Dataset(BaseDataset):
    def download(self) -> None:  # pragma: no cover - manual step
        return

    def prepare(self) -> None:
        sim_cfg = self.dcfg.dataset.simulation
        if not bool(sim_cfg.use):
            return
        repo = Path(sim_cfg.baseline_repo)
        out_dir = Path(sim_cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if getattr(sim_cfg, "ffmpeg_path", None):
            env["FFMPEG_BIN"] = str(sim_cfg.ffmpeg_path)
        subprocess.check_call(
            [
                "python",
                str(Path(repo, "simulation", "simulate_data_from_param.py")),
                "--param_json",
                str(Path(sim_cfg.params_json)),
                "--out_dir",
                str(out_dir),
            ],
            env=env,
        )

    def verify(self) -> None:  # pragma: no cover - manual step
        return

    def build_manifests(self, split: str = "all") -> None:
        dataset_cfg = self.dcfg.dataset
        globs_cfg = dataset_cfg.globs
        infer_cfg = dataset_cfg.infer

        keys: Tuple[str, ...] = tuple(
            name
            for name, enabled in (
                ("language", bool(getattr(infer_cfg, "language_from_path", False))),
                ("distortion", bool(getattr(infer_cfg, "distortion_from_path", False))),
            )
            if enabled
        )

        splits = ("train", "dev", "test") if split == "all" else (split,)
        for sp in splits:
            pattern = getattr(globs_cfg, sp)
            rows = []
            for wav_path in _iter_wavs(pattern):
                try:
                    info = sf.info(wav_path)
                except Exception:
                    continue
                row = {
                    "utt_id": Path(wav_path).stem,
                    "path": str(Path(wav_path).resolve()),
                    "duration_sec": float(info.duration),
                    "sr": int(info.samplerate),
                    "split": sp,
                }
                if keys:
                    row.update(_infer_label(wav_path, keys))
                rows.append(row)
            out_path = self._write_manifest(rows, sp)
            print(f"[urgent2026] Wrote {len(rows)} rows -> {out_path}")


__all__ = ["Urgent2026Dataset"]
