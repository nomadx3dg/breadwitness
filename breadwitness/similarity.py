"""Vector similarity calculations (pure NumPy)."""

from __future__ import annotations

import numpy as np

from .config import SIMILARITY_CHUNK_SIZE


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (zero rows are left as zeros)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_scores(
    matrix: np.ndarray, query: np.ndarray, chunk_size: int = SIMILARITY_CHUNK_SIZE
) -> np.ndarray:
    """Cosine similarity between ``query`` and every row of ``matrix``.

    ``matrix`` may be float16 (as stored); rows are converted to float32 one
    chunk at a time so memory use stays low. Rows are assumed to be unit length.
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("query vector has zero length")
    if matrix.ndim != 2 or matrix.shape[1] != q.size:
        raise ValueError(
            f"dimension mismatch: index vectors have {matrix.shape[-1]} dims, "
            f"query has {q.size}"
        )
    q = q / norm
    out = np.empty(matrix.shape[0], dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk_size):
        out[start : start + chunk_size] = (
            matrix[start : start + chunk_size].astype(np.float32) @ q
        )
    return out
