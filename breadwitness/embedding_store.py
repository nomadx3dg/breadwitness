"""Persistent storage of embeddings (in the same SQLite file as the index).

Vectors are stored as float16 BLOBs (half the size of float32, negligible
loss for cosine similarity). A vector is *current* only while its stored
SHA-256 equals the image's SHA-256, so a changed file never matches with a
stale vector.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np

from .database import Database


class PendingImage(NamedTuple):
    image_id: int
    path: str
    sha256: str
    file_size: int
    mtime_ns: int


class EmbeddingStore:
    def __init__(self, db: Database) -> None:
        self._conn = db.connection

    def upsert_many(
        self, model_id: str, rows: Iterable[tuple[int, str, np.ndarray]]
    ) -> int:
        """Store (image_id, sha256, vector) rows; returns how many were written."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data = [
            (
                image_id,
                model_id,
                int(np.asarray(vec).shape[0]),
                sha256,
                np.asarray(vec, dtype=np.float32).astype(np.float16).tobytes(),
                now,
            )
            for image_id, sha256, vec in rows
        ]
        self._conn.executemany(
            """
            INSERT INTO embeddings (image_id, model_id, dim, sha256, vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id, model_id) DO UPDATE SET
                dim = excluded.dim, sha256 = excluded.sha256,
                vector = excluded.vector, created_at = excluded.created_at
            """,
            data,
        )
        return len(data)

    def pending(self, model_id: str, *, force: bool = False) -> list[PendingImage]:
        """Images lacking a current embedding for ``model_id`` (all if ``force``)."""
        if force:
            sql = (
                "SELECT id, path, sha256, file_size, mtime_ns FROM images ORDER BY id"
            )
            rows = self._conn.execute(sql).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT i.id, i.path, i.sha256, i.file_size, i.mtime_ns
                FROM images i
                LEFT JOIN embeddings e ON e.image_id = i.id AND e.model_id = ?
                WHERE e.image_id IS NULL OR e.sha256 != i.sha256
                ORDER BY i.id
                """,
                (model_id,),
            ).fetchall()
        return [PendingImage(*tuple(r)) for r in rows]

    def count_fresh(self, model_id: str) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM embeddings e JOIN images i ON i.id = e.image_id "
                "WHERE e.model_id = ? AND e.sha256 = i.sha256",
                (model_id,),
            ).fetchone()[0]
        )

    def load_matrix(self, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (image_ids, matrix) of current embeddings; matrix is float16.

        Rows are streamed into one preallocated array to keep memory low.
        """
        n = self.count_fresh(model_id)
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)
        cursor = self._conn.execute(
            """
            SELECT e.image_id, e.dim, e.vector
            FROM embeddings e JOIN images i ON i.id = e.image_id
            WHERE e.model_id = ? AND e.sha256 = i.sha256
            ORDER BY e.image_id
            """,
            (model_id,),
        )
        ids = np.empty(n, dtype=np.int64)
        matrix: np.ndarray | None = None
        filled = 0
        for row in cursor:
            vec = np.frombuffer(row["vector"], dtype=np.float16)
            if matrix is None:
                matrix = np.empty((n, int(row["dim"])), dtype=np.float16)
            if vec.size != matrix.shape[1] or filled >= n:
                raise ValueError(f"inconsistent embedding data for model {model_id}")
            ids[filled] = row["image_id"]
            matrix[filled] = vec
            filled += 1
        assert matrix is not None
        return ids[:filled], matrix[:filled]

    def model_counts(self) -> dict[str, dict[str, int]]:
        """Per-model counts: current and stale embeddings, and dimensions."""
        rows = self._conn.execute(
            """
            SELECT e.model_id AS model_id,
                   SUM(e.sha256 = i.sha256) AS fresh,
                   SUM(e.sha256 != i.sha256) AS stale,
                   MAX(e.dim) AS dim
            FROM embeddings e JOIN images i ON i.id = e.image_id
            GROUP BY e.model_id ORDER BY e.model_id
            """
        ).fetchall()
        return {
            r["model_id"]: {
                "fresh": int(r["fresh"]), "stale": int(r["stale"]), "dim": int(r["dim"]),
            }
            for r in rows
        }

    def delete_model(self, model_id: str) -> int:
        return self._conn.execute(
            "DELETE FROM embeddings WHERE model_id = ?", (model_id,)
        ).rowcount

    def delete_other_models(self, keep_model_id: str) -> int:
        return self._conn.execute(
            "DELETE FROM embeddings WHERE model_id != ?", (keep_model_id,)
        ).rowcount
