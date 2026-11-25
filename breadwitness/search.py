"""Exact, perceptual, embedding and hybrid image search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DEFAULT_LIMIT, DEFAULT_MAX_DISTANCE, HASH_BITS
from .database import Database
from .embedding import EmbeddingBackend
from .embedding_store import EmbeddingStore
from .hashing import hamming_distances
from .ingestion import ingest_image
from .models import MatchMethod, SearchMode, SearchResponse, SearchResult
from .preprocessing import load_image
from .ranking import embedding_score, rank_results, similarity_score
from .similarity import cosine_scores


def search_image(
    db: Database,
    query_path: Path,
    *,
    limit: int = DEFAULT_LIMIT,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    exclude_self: bool = False,
    mode: SearchMode | str = SearchMode.HASH,
    backend: EmbeddingBackend | None = None,
    min_similarity: float | None = None,
) -> SearchResponse:
    """Search the index for images matching ``query_path``.

    Every result is classified by the strongest evidence for it:

    * EXACT: identical file bytes (same SHA-256), in every mode.
    * PERCEPTUAL: pHash Hamming distance <= ``max_distance`` (modes hash, hybrid).
    * SEMANTIC: embedding cosine similarity >= ``min_similarity`` (modes
      embedding, hybrid). Needs ``backend`` and embeddings built for it.

    The default mode (HASH) behaves exactly like Breadwitness 0.1. Raises
    InvalidImageError if the query is not a valid image.
    """
    mode = SearchMode(mode)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 0 <= max_distance <= HASH_BITS:
        raise ValueError(f"max_distance must be between 0 and {HASH_BITS}")
    if mode is not SearchMode.HASH and backend is None:
        raise ValueError(f"mode '{mode.value}' requires an embedding backend")

    use_hash = mode in (SearchMode.HASH, SearchMode.HYBRID)
    use_embedding = mode in (SearchMode.EMBEDDING, SearchMode.HYBRID)
    model_id = backend.model_id if backend is not None else None

    query = ingest_image(query_path)
    candidates = db.all_images()
    if exclude_self:
        candidates = [c for c in candidates if c.path != query.path]
    n = len(candidates)
    if n == 0:
        return SearchResponse(query, [], 0, mode=mode, model_id=model_id)

    p_dist = hamming_distances(query.phash, [c.phash for c in candidates])
    d_dist = hamming_distances(query.dhash, [c.dhash for c in candidates])
    a_dist = hamming_distances(query.ahash, [c.ahash for c in candidates])
    is_exact = np.array([c.sha256 == query.sha256 for c in candidates])

    cosine = np.full(n, np.nan, dtype=np.float32)
    if backend is not None:
        ids, matrix = EmbeddingStore(db).load_matrix(backend.model_id)
        if matrix.shape[0] > 0:
            query_vec = backend.embed_images([load_image(query_path).image])[0]
            by_id = dict(zip(ids.tolist(), cosine_scores(matrix, query_vec).tolist()))
            for i, c in enumerate(candidates):
                value = by_id.get(c.id)
                if value is not None:
                    cosine[i] = value
    has_cosine = ~np.isnan(cosine)
    unembedded = int((~has_cosine).sum()) if backend is not None else 0

    perceptual_ok = (p_dist <= max_distance) if use_hash else np.zeros(n, dtype=bool)
    if use_embedding:
        threshold = (
            min_similarity if min_similarity is not None else backend.default_min_similarity
        )
        with np.errstate(invalid="ignore"):
            semantic_ok = has_cosine & (cosine >= threshold)
    else:
        semantic_ok = np.zeros(n, dtype=bool)

    results: list[SearchResult] = []
    for i in np.flatnonzero(is_exact | perceptual_ok | semantic_ok):
        p, d, a = int(p_dist[i]), int(d_dist[i]), int(a_dist[i])
        cos = float(cosine[i]) if has_cosine[i] else None
        if is_exact[i]:
            method, score, basis = MatchMethod.EXACT, 1.0, "sha256"
        elif perceptual_ok[i]:
            method, score, basis = (
                MatchMethod.PERCEPTUAL, similarity_score(p, d, a), "hash",
            )
        else:
            method, score, basis = (
                MatchMethod.SEMANTIC, embedding_score(cos), "embedding",
            )
        results.append(
            SearchResult(
                record=candidates[i], method=method,
                phash_distance=p, dhash_distance=d, ahash_distance=a,
                score=score, cosine=cos, score_basis=basis,
            )
        )

    return SearchResponse(
        query=query,
        results=rank_results(results)[:limit],
        candidates_scanned=n,
        mode=mode,
        model_id=model_id,
        unembedded=unembedded,
    )
