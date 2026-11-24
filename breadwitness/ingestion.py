"""Discovering image files and turning them into ImageRecords."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS
from .hashing import compute_perceptual_hashes, sha256_file
from .models import ImageRecord
from .preprocessing import InvalidImageError, load_image


def is_supported(path: Path) -> bool:
    """True if the file extension is a supported image type."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_image_files(root: Path) -> Iterator[Path]:
    """Yield absolute paths of supported images under ``root`` (recursive).

    ``root`` may also be a single image file. Output order is deterministic.
    """
    root = Path(root).expanduser()
    if root.is_file():
        if is_supported(root):
            yield Path(os.path.abspath(root))
        return
    if not root.is_dir():
        raise FileNotFoundError(f"No such file or directory: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            if is_supported(candidate):
                yield Path(os.path.abspath(candidate))


def ingest_image(path: Path) -> ImageRecord:
    """Read, validate and fingerprint one image file.

    Raises InvalidImageError if the file is unreadable or corrupted.
    """
    path = Path(os.path.abspath(path))
    try:
        stat = path.stat()
        sha256 = sha256_file(path)
    except OSError as exc:
        raise InvalidImageError(f"{path}: {exc}") from exc
    loaded = load_image(path)
    hashes = compute_perceptual_hashes(loaded.image)
    return ImageRecord(
        path=str(path),
        filename=path.name,
        sha256=sha256,
        phash=hashes.phash,
        dhash=hashes.dhash,
        ahash=hashes.ahash,
        width=loaded.width,
        height=loaded.height,
        format=loaded.format,
        file_size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )
