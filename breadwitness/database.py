"""SQLite persistence for image metadata and fingerprints."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from .config import SCHEMA_VERSION
from .models import ImageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT NOT NULL UNIQUE,
    filename   TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    phash      TEXT NOT NULL,
    dhash      TEXT NOT NULL,
    ahash      TEXT NOT NULL,
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    format     TEXT NOT NULL,
    file_size  INTEGER NOT NULL,
    mtime_ns   INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images (sha256);
"""

_COLUMNS = (
    "id, path, filename, sha256, phash, dhash, ahash, width, height, "
    "format, file_size, mtime_ns, indexed_at"
)


class DatabaseError(RuntimeError):
    """Raised when the index database is unusable."""


def _row_to_record(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=row["id"],
        path=row["path"],
        filename=row["filename"],
        sha256=row["sha256"],
        phash=row["phash"],
        dhash=row["dhash"],
        ahash=row["ahash"],
        width=row["width"],
        height=row["height"],
        format=row["format"],
        file_size=row["file_size"],
        mtime_ns=row["mtime_ns"],
        indexed_at=row["indexed_at"],
    )


class Database:
    """Thin wrapper around a SQLite index.

    Write methods do not commit; call ``commit()`` or use the object as a
    context manager (commits on success, rolls back on error).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Cannot open index database {path}: {exc}") from exc

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
        elif int(row["value"]) > SCHEMA_VERSION:
            raise DatabaseError(
                f"Index schema v{row['value']} is newer than this Breadwitness "
                f"(supports v{SCHEMA_VERSION}). Please upgrade."
            )

    # -- lifecycle ---------------------------------------------------------
    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()

    # -- writes ------------------------------------------------------------
    def upsert(self, record: ImageRecord) -> None:
        """Insert a record, or replace the existing one with the same path."""
        indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            """
            INSERT INTO images (path, filename, sha256, phash, dhash, ahash,
                                width, height, format, file_size, mtime_ns,
                                indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                filename   = excluded.filename,
                sha256     = excluded.sha256,
                phash      = excluded.phash,
                dhash      = excluded.dhash,
                ahash      = excluded.ahash,
                width      = excluded.width,
                height     = excluded.height,
                format     = excluded.format,
                file_size  = excluded.file_size,
                mtime_ns   = excluded.mtime_ns,
                indexed_at = excluded.indexed_at
            """,
            (
                record.path, record.filename, record.sha256, record.phash,
                record.dhash, record.ahash, record.width, record.height,
                record.format, record.file_size, record.mtime_ns, indexed_at,
            ),
        )

    def delete_path(self, path: str) -> bool:
        cur = self._conn.execute("DELETE FROM images WHERE path = ?", (path,))
        return cur.rowcount > 0

    # -- reads -------------------------------------------------------------
    def get_by_path(self, path: str) -> ImageRecord | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM images WHERE path = ?", (path,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def find_by_sha256(self, sha256: str) -> list[ImageRecord]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM images WHERE sha256 = ? ORDER BY path", (sha256,)
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def all_images(self) -> list[ImageRecord]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM images ORDER BY path"
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def paths_under(self, root: str) -> list[str]:
        """Indexed paths located inside the directory ``root``."""
        prefix = root.rstrip("/\\") + ("\\" if "\\" in root else "/")
        rows = self._conn.execute("SELECT path FROM images").fetchall()
        return [r["path"] for r in rows if r["path"].startswith(prefix)]

    def duplicate_groups(self) -> list[list[ImageRecord]]:
        """Groups of indexed files with byte-identical content."""
        shas = [
            r[0]
            for r in self._conn.execute(
                "SELECT sha256 FROM images GROUP BY sha256 "
                "HAVING COUNT(*) > 1 ORDER BY sha256"
            )
        ]
        return [self.find_by_sha256(sha) for sha in shas]

    def stats(self) -> dict[str, Any]:
        total, size = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM images"
        ).fetchone()
        formats = {
            r["format"]: r["n"]
            for r in self._conn.execute(
                "SELECT format, COUNT(*) AS n FROM images "
                "GROUP BY format ORDER BY n DESC, format"
            )
        }
        groups, redundant = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(n - 1), 0) FROM ("
            "SELECT COUNT(*) AS n FROM images GROUP BY sha256 HAVING n > 1)"
        ).fetchone()
        last = self._conn.execute("SELECT MAX(indexed_at) FROM images").fetchone()[0]
        return {
            "total_images": int(total),
            "total_bytes": int(size),
            "formats": formats,
            "duplicate_groups": int(groups),
            "redundant_copies": int(redundant),
            "last_indexed_at": last,
        }
