from __future__ import annotations

"""Self-supervised embedding helpers with deterministic fallbacks."""

from functools import lru_cache
import logging
from typing import Iterable, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    from s3prl.hub import hub

    _HAS_S3PRL = True
except Exception:  # pragma: no cover - executed when unavailable
    torch = None
    hub = None
    _HAS_S3PRL = False


def _seed_from_key(utt_id: str, flavor: str) -> int:
    return abs(hash((utt_id, flavor))) % (2**32)


def deterministic_embedding(utt_id: str, flavor: str, dim: int = 256) -> np.ndarray:
    """Create a deterministic pseudo-random embedding for the utterance."""
    rng = np.random.default_rng(_seed_from_key(utt_id, flavor))
    vec = rng.standard_normal(dim, dtype=np.float32)
    norm = float(np.linalg.norm(vec) + 1e-6)
    return (vec / norm).astype(np.float32)


def _pool_layers(hidden_states: Sequence["torch.Tensor"], layers: Iterable[int]) -> "torch.Tensor":
    selected = []
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(f"Layer index {layer} out of bounds for {len(hidden_states)} hidden states")
        selected.append(hidden_states[layer])
    stacked = torch.stack(selected, dim=0)  # type: ignore[arg-type]
    return stacked.mean(dim=0).mean(dim=0)


@lru_cache(maxsize=2)
def _load_ssl_model(model_name: str):  # pragma: no cover - heavy optional path
    if not _HAS_S3PRL:
        raise RuntimeError("S3PRL is not available; cannot load SSL model")
    if not hasattr(hub, model_name):
        raise ValueError(f"Unknown S3PRL model: {model_name}")
    model = getattr(hub, model_name)()
    model.eval()
    return model


def compute_ssl_embeddings(
    waveform: np.ndarray,
    sample_rate: int,
    model_name: str,
    content_layers: Sequence[int],
    speaker_layers: Sequence[int],
    utt_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute content and speaker embeddings for a waveform.

    Falls back to deterministic pseudo-random vectors when S3PRL (and torch) are
    unavailable.

    Args:
        waveform: 1-D audio samples in float32.
        sample_rate: Sample rate in Hz.
        model_name: S3PRL model identifier.
        content_layers: Layer indices to pool for content embeddings.
        speaker_layers: Layer indices to pool for speaker embeddings.
        utt_id: Utterance identifier used to seed deterministic fallbacks.

    Returns:
        Tuple ``(content_vec, speaker_vec)`` with ``np.float32`` arrays.
    """
    if not _HAS_S3PRL:
        return (
            deterministic_embedding(utt_id, "content"),
            deterministic_embedding(utt_id, "speaker"),
        )

    assert torch is not None

    try:  # pragma: no cover - exercised in environments with S3PRL
        model = _load_ssl_model(model_name)
        with torch.no_grad():
            tensor = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
            outputs = model(tensor, sample_rate)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("S3PRL embedding failed (%s); using deterministic fallback", exc)
        return (
            deterministic_embedding(utt_id, "content"),
            deterministic_embedding(utt_id, "speaker"),
        )

    hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        LOGGER.warning("S3PRL outputs missing hidden_states; using fallback")
        return (
            deterministic_embedding(utt_id, "content"),
            deterministic_embedding(utt_id, "speaker"),
        )

    content = _pool_layers(hidden_states, content_layers)
    speaker = _pool_layers(hidden_states, speaker_layers)

    content_vec = content.cpu().numpy().astype(np.float32)
    speaker_vec = speaker.cpu().numpy().astype(np.float32)

    c_norm = float(np.linalg.norm(content_vec) + 1e-6)
    s_norm = float(np.linalg.norm(speaker_vec) + 1e-6)

    return (content_vec / c_norm, speaker_vec / s_norm)