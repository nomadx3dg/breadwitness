import sqlite3

import numpy as np

from breadwitness.database import Database
from breadwitness.embedding_store import EmbeddingStore
from breadwitness.indexing import index_directory
from breadwitness.similarity import l2_normalize

LEGACY_V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL, sha256 TEXT NOT NULL, phash TEXT NOT NULL,
    dhash TEXT NOT NULL, ahash TEXT NOT NULL, width INTEGER NOT NULL,
    height INTEGER NOT NULL, format TEXT NOT NULL, file_size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL, indexed_at TEXT NOT NULL
);
INSERT INTO meta VALUES ('schema_version', '1');
INSERT INTO images (path, filename, sha256, phash, dhash, ahash, width, height,
    format, file_size, mtime_ns, indexed_at)
VALUES ('/old/a.png', 'a.png', 'abc', '0000000000000000', '0000000000000000',
    '0000000000000000', 10, 10, 'PNG', 1, 1, '2026-01-01T00:00:00+00:00');
"""


def _vectors(n, dim=4, seed=0):
    return l2_normalize(np.random.default_rng(seed).normal(size=(n, dim)))


def test_roundtrip_float16_precision(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        store = EmbeddingStore(db)
        images = db.all_images()
        vecs = _vectors(len(images))
        store.upsert_many("m", [(r.id, r.sha256, v) for r, v in zip(images, vecs)])
        ids, matrix = store.load_matrix("m")
        assert ids.tolist() == [r.id for r in images]
        assert matrix.dtype == np.float16 and matrix.shape == (3, 4)
        assert np.allclose(matrix.astype(np.float32), vecs, atol=1e-3)


def test_persistence_across_restarts(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        rec = db.all_images()[0]
        EmbeddingStore(db).upsert_many("m", [(rec.id, rec.sha256, _vectors(1)[0])])
    with Database(db_path) as db:
        assert EmbeddingStore(db).count_fresh("m") == 1


def test_pending_and_stale_detection(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        store = EmbeddingStore(db)
        images = db.all_images()
        assert len(store.pending("m")) == 3
        store.upsert_many("m", [(r.id, r.sha256, _vectors(1)[0]) for r in images])
        assert store.pending("m") == []
        assert len(store.pending("m", force=True)) == 3
        # a changed file (new sha256) makes its embedding stale
        db.connection.execute("UPDATE images SET sha256 = 'changed' WHERE id = ?", (images[0].id,))
        assert [p.image_id for p in store.pending("m")] == [images[0].id]
        assert store.count_fresh("m") == 2
        assert images[0].id not in store.load_matrix("m")[0].tolist()
        assert store.model_counts()["m"] == {"fresh": 2, "stale": 1, "dim": 4}


def test_models_are_stored_separately_and_purgeable(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        store = EmbeddingStore(db)
        rows = [(r.id, r.sha256, _vectors(1)[0]) for r in db.all_images()]
        store.upsert_many("a", rows)
        store.upsert_many("b", rows)
        assert set(store.model_counts()) == {"a", "b"}
        assert store.delete_other_models("a") == 3
        assert set(store.model_counts()) == {"a"}
        assert store.delete_model("a") == 3
        assert store.model_counts() == {}


def test_pruned_images_lose_their_embeddings(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        store = EmbeddingStore(db)
        store.upsert_many("m", [(r.id, r.sha256, _vectors(1)[0]) for r in db.all_images()])
        (image_dir / "b.jpg").unlink()
        index_directory(db, image_dir)
        assert store.count_fresh("m") == 2
        total = db.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert total == 2  # removed by ON DELETE CASCADE, not just hidden


def test_empty_model_loads_empty_matrix(db_path):
    with Database(db_path) as db:
        ids, matrix = EmbeddingStore(db).load_matrix("none")
        assert ids.size == 0 and matrix.shape[0] == 0


def test_v1_database_is_migrated_without_losing_data(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_V1_SCHEMA)
    conn.commit()
    conn.close()
    with Database(path) as db:
        assert db.count() == 1
        assert db.get_by_path("/old/a.png").sha256 == "abc"
        version = db.connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "2"
        store = EmbeddingStore(db)
        rec = db.all_images()[0]
        store.upsert_many("m", [(rec.id, rec.sha256, _vectors(1)[0])])
        assert store.count_fresh("m") == 1
