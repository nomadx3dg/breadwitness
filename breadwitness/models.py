"""Plain data models shared across Breadwitness modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MatchMethod(str, Enum):
    """How a search result was matched to the query image."""

    EXACT = "exact"  # byte-identical file (same SHA-256)
    PERCEPTUAL = "perceptual"  # visually similar (perceptual hash distance)


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


@dataclass
class SearchResponse:
    """Everything a search returns."""

    query: ImageRecord
    results: list[SearchResult]
    candidates_scanned: int


@dataclass
class IndexReport:
    """Summary of one indexing run."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
