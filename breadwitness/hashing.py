"""Cryptographic and perceptual hashing, and Hamming-distance helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from .config import FILE_HASH_CHUNK_SIZE, HASH_SIZE
from .models import PerceptualHashes


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(FILE_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def compute_perceptual_hashes(image: Image.Image) -> PerceptualHashes:
    """Compute pHash, dHash and aHash (64 bits each) for a PIL image."""
    return PerceptualHashes(
        phash=str(imagehash.phash(image, hash_size=HASH_SIZE)),
        dhash=str(imagehash.dhash(image, hash_size=HASH_SIZE)),
        ahash=str(imagehash.average_hash(image, hash_size=HASH_SIZE)),
    )


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two hex-encoded hashes."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def hamming_distances(query: str, candidates: Sequence[str]) -> np.ndarray:
    """Vectorized Hamming distances from one hex hash to many."""
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.int64)
    q = np.uint64(int(query, 16))
    values = np.fromiter(
        (int(c, 16) for c in candidates), dtype=np.uint64, count=len(candidates)
    )
    xor = np.bitwise_xor(values, q)
    bits = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64)
    return bits.sum(axis=1, dtype=np.int64)
