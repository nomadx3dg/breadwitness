import shutil

import pytest

from breadwitness.database import Database
from breadwitness.indexing import index_directory
from breadwitness.models import MatchMethod
from breadwitness.preprocessing import InvalidImageError
from breadwitness.search import search_image

from .conftest import make_image


@pytest.fixture
def indexed_db(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
    db = Database(db_path)
    yield db
    db.close()


def test_exact_match(indexed_db, image_dir, tmp_path):
    query = tmp_path / "query.png"
    shutil.copy(image_dir / "a.png", query)
    response = search_image(indexed_db, query)
    top = response.results[0]
    assert top.method is MatchMethod.EXACT
    assert top.record.filename == "a.png"
    assert top.phash_distance == 0 and top.score == 1.0


def test_perceptual_match_on_recompressed_resized_copy(indexed_db, tmp_path):
    query = tmp_path / "query.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    response = search_image(indexed_db, query)
    top = response.results[0]
    assert top.method is MatchMethod.PERCEPTUAL
    assert top.record.filename == "a.png"
    assert 0 <= top.phash_distance <= 10
    assert 0.0 < top.score < 1.0


def test_unrelated_image_has_no_matches(indexed_db, tmp_path):
    query = tmp_path / "other.png"
    make_image(999).save(query)
    assert search_image(indexed_db, query).results == []


def test_limit_and_ordering(indexed_db, image_dir, tmp_path):
    shutil.copy(image_dir / "a.png", image_dir / "a_copy.png")
    with Database(indexed_db.path) as db:
        index_directory(db, image_dir)
    query = tmp_path / "q.png"
    shutil.copy(image_dir / "a.png", query)
    everything = search_image(indexed_db, query, max_distance=64, limit=10)
    assert [r.method for r in everything.results[:2]] == [MatchMethod.EXACT] * 2
    assert len(everything.results) == 4
    distances = [r.phash_distance for r in everything.results[2:]]
    assert distances == sorted(distances)
    assert len(search_image(indexed_db, query, max_distance=64, limit=1).results) == 1


def test_exclude_self(indexed_db, image_dir):
    results = search_image(indexed_db, image_dir / "a.png", exclude_self=True).results
    assert all(r.record.filename != "a.png" for r in results)


def test_invalid_query_and_arguments(indexed_db, image_dir):
    with pytest.raises(InvalidImageError):
        search_image(indexed_db, image_dir / "corrupt.jpg")
    with pytest.raises(ValueError):
        search_image(indexed_db, image_dir / "a.png", limit=0)
    with pytest.raises(ValueError):
        search_image(indexed_db, image_dir / "a.png", max_distance=65)


def test_empty_index_returns_no_results(db_path, image_dir):
    with Database(db_path) as db:
        response = search_image(db, image_dir / "a.png")
    assert response.results == [] and response.candidates_scanned == 0
