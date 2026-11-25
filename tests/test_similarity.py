import numpy as np
import pytest

from breadwitness.similarity import cosine_scores, l2_normalize


def test_l2_normalize_and_zero_rows():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], [0, 0])


def test_cosine_matches_naive_computation_across_chunks():
    rng = np.random.default_rng(0)
    matrix = l2_normalize(rng.normal(size=(50, 8)))
    query = rng.normal(size=8)
    expected = matrix @ (query / np.linalg.norm(query))
    got = cosine_scores(matrix.astype(np.float16), query, chunk_size=7)
    assert np.allclose(got, expected, atol=2e-3)


def test_cosine_identical_and_opposite():
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    assert np.allclose(cosine_scores(matrix, np.array([2.0, 0.0])), [1, -1, 0])


def test_cosine_rejects_bad_input():
    with pytest.raises(ValueError, match="zero length"):
        cosine_scores(np.ones((2, 2), dtype=np.float16), np.zeros(2))
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_scores(np.ones((2, 3), dtype=np.float16), np.ones(2))
