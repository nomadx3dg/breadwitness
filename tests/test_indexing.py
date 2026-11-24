import os
import shutil

from breadwitness.database import Database
from breadwitness.indexing import index_directory

from .conftest import make_image


def test_first_index_adds_valid_and_skips_corrupt(image_dir, db_path):
    with Database(db_path) as db:
        report = index_directory(db, image_dir)
        assert report.added == 3
        assert len(report.skipped) == 1 and "corrupt.jpg" in report.skipped[0][0]
        assert db.count() == 3


def test_incremental_run_is_unchanged(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        report = index_directory(db, image_dir)
        assert (report.added, report.updated, report.unchanged) == (0, 0, 3)


def test_changed_file_is_detected(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        target = image_dir / "a.png"
        make_image(50).save(target)
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        report = index_directory(db, image_dir)
        assert report.updated == 1 and db.count() == 3


def test_deleted_file_is_pruned(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        (image_dir / "b.jpg").unlink()
        report = index_directory(db, image_dir)
        assert report.removed == 1 and db.count() == 2


def test_no_prune_keeps_missing_files(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        (image_dir / "b.jpg").unlink()
        report = index_directory(db, image_dir, prune=False)
        assert report.removed == 0 and db.count() == 3


def test_file_that_becomes_corrupt_is_dropped(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        (image_dir / "a.png").write_bytes(b"broken now")
        report = index_directory(db, image_dir)
        assert db.count() == 2 and len(report.skipped) == 2


def test_force_reprocesses_but_reports_unchanged_content(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        report = index_directory(db, image_dir, force=True)
        assert report.unchanged == 3 and report.updated == 0


def test_duplicate_detection_and_stats(image_dir, db_path):
    shutil.copy(image_dir / "a.png", image_dir / "nested" / "a_copy.png")
    with Database(db_path) as db:
        index_directory(db, image_dir)
        groups = db.duplicate_groups()
        assert len(groups) == 1 and len(groups[0]) == 2
        info = db.stats()
        assert info["total_images"] == 4
        assert info["duplicate_groups"] == 1 and info["redundant_copies"] == 1


def test_index_persists_across_restarts(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
    with Database(db_path) as db:
        assert db.count() == 3
        assert db.get_by_path(str(image_dir / "a.png")) is not None
