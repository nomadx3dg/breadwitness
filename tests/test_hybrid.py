import shutil

import pytest

from breadwitness.database import Database
from breadwitness.embedding import ColorHistogramBackend
from breadwitness.embedding_index import build_embeddings
from breadwitness.indexing import index_directory
from breadwitness.models import ImageRecord, MatchMethod, SearchMode, SearchResult
from breadwitness.ranking import rank_results
from breadwitness.search import search_image

from .conftest import make_image


def _result(name, method, p=0, cosine=None, score=0.5):
    rec = ImageRecord(name, name, "s", "0" * 16, "0" * 16, "0" * 16, 1, 1, "PNG", 1, 1)
    return SearchResult(rec, method, p, p, p, score, cosine, "x")


def test_ranking_orders_by_tier_not_by_raw_scores():
    results = [
        _result("semantic_high.png", MatchMethod.SEMANTIC, p=30, cosine=0.99, score=0.99),
        _result("perceptual.png", MatchMethod.PERCEPTUAL, p=4, cosine=0.5, score=0.6),
        _result("exact.png", MatchMethod.EXACT, p=0, cosine=0.1, score=1.0),
        _result("semantic_low.png", MatchMethod.SEMANTIC, p=20, cosine=0.85, score=0.85),
    ]
    names = [r.record.filename for r in rank_results(results)]
    assert names == ["exact.png", "perceptual.png", "semantic_high.png", "semantic_low.png"]


def test_ranking_perceptual_ties_broken_by_cosine_then_missing_last():
    results = [
        _result("no_cos.png", MatchMethod.PERCEPTUAL, p=2, cosine=None),
        _result("low.png", MatchMethod.PERCEPTUAL, p=2, cosine=0.5),
        _result("high.png", MatchMethod.PERCEPTUAL, p=2, cosine=0.9),
        _result("closer.png", MatchMethod.PERCEPTUAL, p=1, cosine=0.1),
    ]
    names = [r.record.filename for r in rank_results(results)]
    assert names == ["closer.png", "high.png", "low.png", "no_cos.png"]


@pytest.fixture
def embedded_db(image_dir, db_path):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        build_embeddings(db, ColorHistogramBackend())
    db = Database(db_path)
    yield db
    db.close()


def test_hash_mode_is_unchanged_from_0_1(embedded_db, tmp_path):
    query = tmp_path / "q.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    response = search_image(embedded_db, query)  # default mode
    assert response.mode is SearchMode.HASH
    assert [r.method for r in response.results] == [MatchMethod.PERCEPTUAL]
    assert response.results[0].cosine is None


def test_hybrid_shows_all_signals(embedded_db, tmp_path):
    query = tmp_path / "q.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    response = search_image(
        embedded_db, query, mode="hybrid", backend=ColorHistogramBackend()
    )
    top = response.results[0]
    assert top.method is MatchMethod.PERCEPTUAL and top.record.filename == "a.png"
    assert top.cosine is not None and top.cosine > 0.95
    assert top.score_basis == "hash"
    assert response.model_id == "color-hist" and response.unembedded == 0


def test_exact_match_wins_in_every_mode(embedded_db, image_dir, tmp_path):
    query = tmp_path / "q.png"
    shutil.copy(image_dir / "a.png", query)
    for mode in ("hash", "embedding", "hybrid"):
        backend = None if mode == "hash" else ColorHistogramBackend()
        top = search_image(embedded_db, query, mode=mode, backend=backend).results[0]
        assert top.method is MatchMethod.EXACT and top.score_basis == "sha256"


def test_embedding_mode_returns_semantic_matches_ordered_by_cosine(embedded_db, tmp_path):
    query = tmp_path / "q.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    response = search_image(
        embedded_db, query, mode="embedding", backend=ColorHistogramBackend(),
        min_similarity=-1.0,
    )
    assert response.results[0].method is MatchMethod.SEMANTIC  # no hash tier in this mode
    assert response.results[0].record.filename == "a.png"
    cosines = [r.cosine for r in response.results]
    assert cosines == sorted(cosines, reverse=True) and len(cosines) == 3
    assert all(r.score_basis == "embedding" for r in response.results)


def test_threshold_filters_semantic_matches(embedded_db, tmp_path):
    query = tmp_path / "q.png"
    make_image(999).save(query)
    backend = ColorHistogramBackend()
    strict = search_image(embedded_db, query, mode="embedding", backend=backend)
    assert strict.results == []
    loose = search_image(embedded_db, query, mode="embedding", backend=backend, min_similarity=-1.0)
    assert len(loose.results) == 3


def test_hybrid_combines_tiers(embedded_db, tmp_path):
    query = tmp_path / "q.jpg"
    make_image(1).resize((128, 128)).save(query, quality=60)
    response = search_image(
        embedded_db, query, mode="hybrid", backend=ColorHistogramBackend(),
        min_similarity=-1.0, limit=10,
    )
    methods = [r.method for r in response.results]
    assert methods == [MatchMethod.PERCEPTUAL, MatchMethod.SEMANTIC, MatchMethod.SEMANTIC]


def test_images_without_embeddings_are_counted_and_still_hash_searchable(
    image_dir, db_path, tmp_path
):
    with Database(db_path) as db:
        index_directory(db, image_dir)
        backend = ColorHistogramBackend()
        build_embeddings(db, backend)
        make_image(1).save(image_dir / "new.png")  # indexed later, not embedded yet
        index_directory(db, image_dir)
        query = tmp_path / "q.jpg"
        make_image(1).resize((128, 128)).save(query, quality=60)
        response = search_image(db, query, mode="hybrid", backend=backend)
    assert response.unembedded == 1
    assert {r.record.filename for r in response.results} >= {"a.png", "new.png"}


def test_embedding_modes_require_a_backend(embedded_db, image_dir):
    with pytest.raises(ValueError, match="requires an embedding backend"):
        search_image(embedded_db, image_dir / "a.png", mode="hybrid")
