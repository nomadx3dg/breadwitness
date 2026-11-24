"""Exact and perceptual image search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DEFAULT_LIMIT, DEFAULT_MAX_DISTANCE, HASH_BITS
from .database import Database
from .hashing import hamming_distances
from .ingestion import ingest_image
from .models import MatchMethod, SearchResponse, SearchResult
from .ranking import rank_results, similarity_score


def search_image(
    db: Database,
    query_path: Path,
    *,
    limit: int = DEFAULT_LIMIT,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    exclude_self: bool = False,
) -> SearchResponse:
    """Search the index for images matching ``query_path``.

    A result is EXACT when the file bytes are identical (same SHA-256). It is
    PERCEPTUAL when its pHash Hamming distance to the query is at most
    ``max_distance``. Raises InvalidImageError if the query is not a valid image.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not 0 <= max_distance <= HASH_BITS:
        raise ValueError(f"max_distance must be between 0 and {HASH_BITS}")

    query = ingest_image(query_path)
    candidates = db.all_images()
    if exclude_self:
        candidates = [c for c in candidates if c.path != query.path]
    if not candidates:
        return SearchResponse(query=query, results=[], candidates_scanned=0)

    p_dist = hamming_distances(query.phash, [c.phash for c in candidates])
    d_dist = hamming_distances(query.dhash, [c.dhash for c in candidates])
    a_dist = hamming_distances(query.ahash, [c.ahash for c in candidates])
    is_exact = np.array([c.sha256 == query.sha256 for c in candidates])

    results: list[SearchResult] = []
    for i in np.flatnonzero(is_exact | (p_dist <= max_distance)):
        exact = bool(is_exact[i])
        p, d, a = int(p_dist[i]), int(d_dist[i]), int(a_dist[i])
        results.append(
            SearchResult(
                record=candidates[i],
                method=MatchMethod.EXACT if exact else MatchMethod.PERCEPTUAL,
                phash_distance=p,
                dhash_distance=d,
                ahash_distance=a,
                score=1.0 if exact else similarity_score(p, d, a),
            )
        )

    ranked = rank_results(results)[:limit]
    return SearchResponse(query=query, results=ranked, candidates_scanned=len(candidates))
