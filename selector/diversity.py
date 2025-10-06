from __future__ import annotations

"""Embedding diversity utilities with FAISS fallbacks."""

from dataclasses import dataclass
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional path
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - executed when faiss missing
    faiss = None
    _HAS_FAISS = False


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return mat / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    normed = _normalise_rows(matrix)
    return normed @ normed.T


@dataclass
class KCenterResult:
    selected: List[int]
    rejections: int


def k_center_farthest(
    vectors: np.ndarray,
    k: int,
    min_cos_margin: float = 0.0,
) -> KCenterResult:
    """Select ``k`` indices using farthest-point traversal with cosine guard."""
    if k <= 0:
        return KCenterResult([], 0)

    normed = _normalise_rows(vectors)
    n = normed.shape[0]
    if n == 0:
        return KCenterResult([], 0)

    selected: List[int] = [0]
    nearest_sim = normed @ normed[0]
    rejections = 0

    while len(selected) < min(k, n):
        candidate = None
        best_score = 2.0  # bigger than max cosine
        for idx in range(n):
            if idx in selected:
                continue
            sim = float(nearest_sim[idx])
            if sim >= 1.0 - min_cos_margin:
                rejections += 1
                continue
            if sim < best_score:
                best_score = sim
                candidate = idx
        if candidate is None:
            break
        selected.append(candidate)
        sims = normed @ normed[candidate]
        nearest_sim = np.maximum(nearest_sim, sims)

    return KCenterResult(selected, rejections)


class CosineANNIndex:
    """Simple cosine similarity index with optional FAISS backend."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self._vectors: Optional[np.ndarray] = None
        if _HAS_FAISS:  # pragma: no cover - not hit without faiss
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = None

    def add(self, vectors: np.ndarray) -> None:
        normed = _normalise_rows(vectors)
        self._vectors = normed
        if self._index is not None:
            self._index.reset()
            self._index.add(normed.astype(np.float32))

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            raise RuntimeError("Index is empty, call add() first")
        q = _normalise_rows(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if self._index is not None:
            distances, indices = self._index.search(q, k)
            return distances[0], indices[0]
        sims = (self._vectors @ q[0]).astype(np.float32)
        idx = np.argsort(sims)[::-1][:k]
        return sims[idx], idx


def cosine_to_selected(
    vectors: np.ndarray,
    selected_indices: Sequence[int],
    candidate_index: int,
) -> float:
    """Return cosine similarity between candidate and nearest selected vector."""
    if not selected_indices:
        return 0.0
    normed = _normalise_rows(vectors)
    candidate = normed[candidate_index]
    selected = normed[list(selected_indices)]
    sims = selected @ candidate
    return float(np.max(sims))