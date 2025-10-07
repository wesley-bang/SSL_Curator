from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import shutil
import tarfile
import zipfile

import requests


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest for the given file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, overwrite: bool = False, timeout: Optional[int] = None) -> Path:
    """Download a URL to a destination file, streaming to disk."""
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    return dest


def extract(archive: Path, dest: Path, overwrite: bool = False) -> Path:
    """Extract a tar or zip archive to the destination directory."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix in {".zip"}:
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(dest)
    elif archive.suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        mode = "r" if archive.suffix == ".tar" else "r:*"
        with tarfile.open(archive, mode) as tar_ref:
            tar_ref.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    return dest


__all__ = ["sha256_file", "download", "extract"]
