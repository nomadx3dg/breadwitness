"""Scoring and ordering of search results.

Scores are heuristic similarity values in [0, 1] derived from Hamming
distances. They are NOT probabilities and must not be presented as such.
"""

from __future__ import annotations

from .config import HASH_BITS
from .models import MatchMethod, SearchResult

# Relative weights of the pHash, dHash and aHash distances.
HASH_WEIGHTS: tuple[float, float, float] = (0.5, 0.3, 0.2)

# 1.0 is reserved for byte-identical files; approximate matches stay below it.
PERCEPTUAL_MAX_SCORE = 0.99


def similarity_score(phash_d: int, dhash_d: int, ahash_d: int) -> float:
    """Weighted similarity for approximate matches, in [0, PERCEPTUAL_MAX_SCORE]."""
    weighted = (
        HASH_WEIGHTS[0] * phash_d
        + HASH_WEIGHTS[1] * dhash_d
        + HASH_WEIGHTS[2] * ahash_d
    )
    return max(0.0, min(PERCEPTUAL_MAX_SCORE, 1.0 - weighted / HASH_BITS))


def embedding_score(cosine: float) -> float:
    """Cosine similarity clipped to [0, 1] for display."""
    return max(0.0, min(1.0, cosine))


_TIER = {MatchMethod.EXACT: 0, MatchMethod.PERCEPTUAL: 1, MatchMethod.SEMANTIC: 2}


def sort_key(result: SearchResult) -> tuple[int, int, float, float, str]:
    """Order results by evidence tier, then by the signal that defines the tier.

    1. exact (identical bytes)
    2. perceptual (pHash distance, smaller first; embedding cosine breaks ties)
    3. semantic (embedding cosine, larger first)

    Scores computed from different signals are never compared with each other;
    the tier decides first. Without embeddings this reduces to the 0.1 order.
    """
    tier = _TIER[result.method]
    phash = result.phash_distance if tier < 2 else 0
    cosine = result.cosine if result.cosine is not None else -2.0
    return (tier, phash, -cosine, -result.score, result.record.path)


def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    return sorted(results, key=sort_key)
