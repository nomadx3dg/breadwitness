"""Plain data models shared across Breadwitness modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MatchMethod(str, Enum):
    """How a search result was matched to the query image."""

    EXACT = "exact"  # byte-identical file (same SHA-256)
    PERCEPTUAL = "perceptual"  # visually similar (perceptual hash distance)
    SEMANTIC = "semantic"  # similar by embedding (cosine similarity)


class SearchMode(str, Enum):
    """Which signals a search uses to decide what counts as a match."""

    HASH = "hash"  # exact + perceptual hashes (the 0.1 behavior)
    EMBEDDING = "embedding"  # exact + embedding similarity
    HYBRID = "hybrid"  # all three


@dataclass(frozen=True)
class PerceptualHashes:
    """64-bit perceptual hashes encoded as 16-character hex strings."""

    phash: str
    dhash: str
    ahash: str


@dataclass(frozen=True)
class ImageRecord:
    """Metadata and fingerprints for a single image file."""

    path: str
    filename: str
    sha256: str
    phash: str
    dhash: str
    ahash: str
    width: int
    height: int
    format: str
    file_size: int
    mtime_ns: int
    indexed_at: str = ""
    id: int | None = None


@dataclass(frozen=True)
class SearchResult:
    """One ranked search hit."""

    record: ImageRecord
    method: MatchMethod
    phash_distance: int
    dhash_distance: int
    ahash_distance: int
    score: float  # heuristic similarity in [0, 1]; NOT a probability
    cosine: float | None = None  # embedding cosine similarity, if computed
    score_basis: str = "hash"  # which signal `score` is derived from


@dataclass
class SearchResponse:
    """Everything a search returns."""

    query: ImageRecord
    results: list[SearchResult]
    candidates_scanned: int
    mode: SearchMode = SearchMode.HASH
    model_id: str | None = None
    unembedded: int = 0  # candidates without a current embedding for model_id


@dataclass
class IndexReport:
    """Summary of one indexing run."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class EmbedReport:
    """Summary of one embedding run."""

    model_id: str
    total: int = 0  # images in the index
    embedded: int = 0  # embeddings written in this run
    up_to_date: int = 0  # images that already had a current embedding
    purged: int = 0  # rows removed for other models
    skipped: list[tuple[str, str]] = field(default_factory=list)
