"""Embedding backends: turn images into L2-normalized vectors.

A backend has a ``model_id`` (stored with every vector, so changing model
never mixes incompatible vectors), a ``dim``, a ``default_min_similarity``
and an ``embed_images`` method.

Built in:

* ``color-hist``: a tiny, dependency-free color-layout embedding. No download,
  no extra packages. It measures color similarity, not meaning.
* ``openclip:<model>/<pretrained>``: semantic embeddings from OpenCLIP
  (requires ``pip install "breadwitness[embeddings]"``; weights are
  downloaded on first use and cached).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL, MODEL_CACHE_ENV_VAR, MODEL_ENV_VAR
from .similarity import l2_normalize


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot be created or used."""


class EmbeddingBackend(Protocol):
    model_id: str
    dim: int
    default_min_similarity: float

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-length vectors."""
        ...


class ColorHistogramBackend:
    """Spatial color-histogram embedding (256 dimensions, no model download).

    Each image is reduced to 64x64, split into a 2x2 grid, and each cell gets
    a 4x4x4 RGB histogram. Square roots of the normalized histograms make the
    cosine similarity a Bhattacharyya coefficient.
    """

    model_id = "color-hist"
    GRID = 2
    LEVELS = 4
    dim = GRID * GRID * LEVELS**3
    default_min_similarity = 0.85

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self._embed_one(image)
        return l2_normalize(out)

    def _embed_one(self, image: Image.Image) -> np.ndarray:
        small = np.asarray(
            image.convert("RGB").resize((64, 64), Image.Resampling.BOX), dtype=np.uint8
        )
        q = (small >> 6).astype(np.int64)  # 0..3 per channel
        bins = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        step = 64 // self.GRID
        cells = []
        for gy in range(self.GRID):
            for gx in range(self.GRID):
                cell = bins[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
                hist = np.bincount(cell.ravel(), minlength=64).astype(np.float32)
                cells.append(np.sqrt(hist / hist.sum()))
        return np.concatenate(cells) / self.GRID


class OpenCLIPBackend:
    """OpenCLIP image embeddings, always run on the CPU."""

    default_min_similarity = 0.80  # heuristic; tune with --min-similarity

    def __init__(self, model_name: str, pretrained: str, cache_dir: str | None = None):
        try:
            import open_clip  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise EmbeddingBackendError(
                "OpenCLIP is not installed. Install it with:\n"
                '  pip install "breadwitness[embeddings]"\n'
                "For a smaller CPU-only PyTorch, first run:\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu\n"
                "Or use the built-in model: --model color-hist"
            ) from exc
        self.model_id = f"openclip:{model_name}/{pretrained}"
        self._torch = torch
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device="cpu", cache_dir=cache_dir
            )
        except Exception as exc:  # open_clip raises many different error types
            raise EmbeddingBackendError(
                f"Could not load OpenCLIP model {self.model_id}: {exc}"
            ) from exc
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self.dim = int(self.embed_images([Image.new("RGB", (64, 64))]).shape[1])

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images])
        with torch.inference_mode():
            features = self._model.encode_image(batch)
        return l2_normalize(features.float().cpu().numpy())


# -- registry ----------------------------------------------------------------

BackendFactory = Callable[[str], EmbeddingBackend]


def _make_color_hist(arg: str) -> EmbeddingBackend:
    if arg:
        raise EmbeddingBackendError("color-hist takes no arguments")
    return ColorHistogramBackend()


def _make_openclip(arg: str) -> EmbeddingBackend:
    name, _, pretrained = arg.partition("/")
    if not name or not pretrained:
        raise EmbeddingBackendError(
            "OpenCLIP models are written openclip:<model>/<pretrained>, "
            f"for example {DEFAULT_MODEL}"
        )
    return OpenCLIPBackend(name, pretrained, os.environ.get(MODEL_CACHE_ENV_VAR))


_FACTORIES: dict[str, BackendFactory] = {
    "color-hist": _make_color_hist,
    "openclip": _make_openclip,
}


def register_backend(prefix: str, factory: BackendFactory) -> None:
    """Register a backend. ``factory`` receives the text after ``prefix:``."""
    _FACTORIES[prefix] = factory


def resolve_model_spec(spec: str | None = None) -> str:
    """Pick the model: explicit spec, then $BREADWITNESS_MODEL, then default."""
    return spec or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def get_backend(spec: str | None = None) -> EmbeddingBackend:
    """Create the backend for a model spec such as ``color-hist``."""
    spec = resolve_model_spec(spec)
    prefix, _, arg = spec.partition(":")
    factory = _FACTORIES.get(prefix)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES))
        raise EmbeddingBackendError(f"Unknown model '{spec}'. Known backends: {known}")
    return factory(arg)


def describe_models() -> list[tuple[str, str, bool]]:
    """(spec, description, available) for the models Breadwitness documents."""
    openclip_ok = importlib.util.find_spec("open_clip") is not None
    return [
        (
            DEFAULT_MODEL,
            "semantic, 512 dims, ~600 MB download, needs breadwitness[embeddings]",
            openclip_ok,
        ),
        ("color-hist", "color layout only, 256 dims, built in, no download", True),
    ]
