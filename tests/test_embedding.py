import sys

import numpy as np
import pytest

from breadwitness.embedding import (
    ColorHistogramBackend,
    EmbeddingBackendError,
    get_backend,
    resolve_model_spec,
)
from breadwitness.config import DEFAULT_MODEL

from .conftest import make_image


def test_color_hist_vectors_are_unit_length_and_deterministic():
    backend = ColorHistogramBackend()
    vectors = backend.embed_images([make_image(1), make_image(2), make_image(1)])
    assert vectors.shape == (3, backend.dim)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vectors[0], vectors[2])


def test_color_hist_similar_vs_different_images():
    backend = ColorHistogramBackend()
    original = backend.embed_images([make_image(1)])[0]
    resized = backend.embed_images([make_image(1).resize((120, 120))])[0]
    other = backend.embed_images([make_image(2)])[0]
    assert float(original @ resized) > 0.95
    assert float(original @ other) < backend.default_min_similarity


def test_registry_resolution(monkeypatch):
    assert get_backend("color-hist").model_id == "color-hist"
    assert get_backend("test-mean").model_id == "test-mean"
    monkeypatch.delenv("BREADWITNESS_MODEL", raising=False)
    assert resolve_model_spec(None) == DEFAULT_MODEL
    monkeypatch.setenv("BREADWITNESS_MODEL", "color-hist")
    assert resolve_model_spec(None) == "color-hist"
    assert resolve_model_spec("test-mean") == "test-mean"


def test_unknown_and_malformed_specs():
    with pytest.raises(EmbeddingBackendError, match="Unknown model"):
        get_backend("nonsense")
    with pytest.raises(EmbeddingBackendError, match="takes no arguments"):
        get_backend("color-hist:extra")
    with pytest.raises(EmbeddingBackendError, match="openclip:<model>/<pretrained>"):
        get_backend("openclip:ViT-B-32")


def test_openclip_missing_package_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "open_clip", None)  # makes the import fail
    with pytest.raises(EmbeddingBackendError, match="breadwitness\\[embeddings\\]"):
        get_backend("openclip:ViT-B-32/laion2b_s34b_b79k")
