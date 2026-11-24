"""Building and incrementally updating the image index."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from .database import Database
from .ingestion import ingest_image, iter_image_files
from .models import IndexReport
from .preprocessing import InvalidImageError

ProgressCallback = Callable[[Path, str], None]

COMMIT_BATCH_SIZE = 100


def index_directory(
    db: Database,
    root: Path,
    *,
    force: bool = False,
    prune: bool = True,
    on_progress: ProgressCallback | None = None,
) -> IndexReport:
    """Index every supported image under ``root``.

    A file is skipped as unchanged when its size and modification time match
    the stored record (unless ``force``). If ``prune`` is true and ``root`` is
    a directory, records under it whose files no longer exist are removed.
    """
    root = Path(os.path.abspath(Path(root).expanduser()))
    report = IndexReport()
    seen: set[str] = set()

    def notify(path: Path, status: str) -> None:
        if on_progress is not None:
            on_progress(path, status)

    for path in iter_image_files(root):
        key = str(path)
        seen.add(key)
        report.scanned += 1
        existing = db.get_by_path(key)

        try:
            stat = path.stat()
        except OSError as exc:
            report.skipped.append((key, str(exc)))
            notify(path, "skipped")
            continue

        if (
            existing is not None
            and not force
            and existing.file_size == stat.st_size
            and existing.mtime_ns == stat.st_mtime_ns
        ):
            report.unchanged += 1
            notify(path, "unchanged")
            continue

        try:
            record = ingest_image(path)
        except InvalidImageError as exc:
            if existing is not None:
                db.delete_path(key)  # stale record for a now-broken file
            report.skipped.append((key, str(exc)))
            notify(path, "skipped")
            continue

        db.upsert(record)
        if existing is None:
            report.added += 1
            notify(path, "added")
        elif existing.sha256 != record.sha256:
            report.updated += 1
            notify(path, "updated")
        else:
            report.unchanged += 1  # same content; metadata refreshed
            notify(path, "unchanged")

        if report.scanned % COMMIT_BATCH_SIZE == 0:
            db.commit()

    if prune and root.is_dir():
        for stale in db.paths_under(str(root)):
            if stale not in seen:
                db.delete_path(stale)
                report.removed += 1
                notify(Path(stale), "removed")

    db.commit()
    return report
