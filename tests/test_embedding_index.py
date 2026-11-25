import os

import pytest

from breadwitness.database import Database
from breadwitness.embedding import ColorHistogramBackend
from breadwitness.embedding_index import build_embeddings
from breadwitness.embedding_store import EmbeddingStore
from breadwitness.indexing import index_directory

from .conftest import make_image


@pytest.fixture
def db(image_dir, db_path):
    with Database(db_path) as database:
        index_directory(database, image_dir)
        yield database


def test_first_run_embeds_everything(db, mean_backend):
    report = build_embeddings(db, mean_backend)
    assert (report.embedded, report.up_to_date, report.total) == (3, 0, 3)
    assert EmbeddingStore(db).count_fresh("test-mean") == 3


def test_second_run_skips_unchanged_files(db, mean_backend):
    build_embeddings(db, mean_backend)
    mean_backend.batches.clear()
    report = build_embeddings(db, mean_backend)
    assert (report.embedded, report.up_to_date) == (0, 3)
    assert mean_backend.batches == []  # the model was not called at all


def test_changed_file_is_re_embedded_only(db, image_dir, mean_backend):
    build_embeddings(db, mean_backend)
    target = image_dir / "a.png"
    make_image(77).save(target)
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    index_directory(db, image_dir)  # refreshes the hash index; embedding is now stale
    mean_backend.batches.clear()
    report = build_embeddings(db, mean_backend)
    assert (report.embedded, report.up_to_date) == (1, 2)
    assert mean_backend.batches == [1]


def test_batches_respect_batch_size(db, mean_backend):
    build_embeddings(db, mean_backend, batch_size=2)
    assert mean_backend.batches == [2, 1]


def test_progress_callback(db, mean_backend):
    calls = []
    build_embeddings(db, mean_backend, batch_size=2, on_progress=lambda d, t: calls.append((d, t)))
    assert calls == [(2, 3), (3, 3)]


def test_rebuild_re_embeds_everything(db, mean_backend):
    build_embeddings(db, mean_backend)
    report = build_embeddings(db, mean_backend, force=True)
    assert report.embedded == 3 and report.up_to_date == 0


def test_model_change_embeds_under_new_id_and_keeps_old(db, mean_backend):
    build_embeddings(db, mean_backend)
    report = build_embeddings(db, ColorHistogramBackend())
    assert report.embedded == 3
    assert set(EmbeddingStore(db).model_counts()) == {"test-mean", "color-hist"}


def test_purge_other_models(db, mean_backend):
    build_embeddings(db, mean_backend)
    report = build_embeddings(db, ColorHistogramBackend(), purge_other_models=True)
    assert report.purged == 3
    assert set(EmbeddingStore(db).model_counts()) == {"color-hist"}


def test_missing_corrupt_and_modified_files_are_skipped(db, image_dir, mean_backend):
    (image_dir / "a.png").unlink()
    (image_dir / "b.jpg").write_bytes(b"corrupted after indexing")
    report = build_embeddings(db, mean_backend)
    assert report.embedded == 1 and len(report.skipped) == 2
    reasons = " ".join(r for _, r in report.skipped)
    assert "not found" in reasons and "changed" in reasons


def test_invalid_batch_size(db, mean_backend):
    with pytest.raises(ValueError):
        build_embeddings(db, mean_backend, batch_size=0)
