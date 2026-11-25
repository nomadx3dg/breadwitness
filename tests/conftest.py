"""Shared test helpers: deterministic synthetic images."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from breadwitness.embedding import register_backend
from breadwitness.similarity import l2_normalize


def make_image(seed: int, size: tuple[int, int] = (256, 256)) -> Image.Image:
    """A smooth, deterministic, structured image: blurred random color blocks."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks, "RGB").resize(size, Image.Resampling.BICUBIC)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A small library: 3 valid images, 1 corrupted file, 1 non-image file."""
    root = tmp_path / "library"
    (root / "nested").mkdir(parents=True)
    make_image(1).save(root / "a.png")
    make_image(2).save(root / "b.jpg", quality=95)
    make_image(3).save(root / "nested" / "c.bmp")
    (root / "corrupt.jpg").write_bytes(b"this is not a real jpeg")
    (root / "notes.txt").write_text("ignore me")
    return root


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "index.db"


class MeanColorBackend:
    """Tiny deterministic test model: centered mean RGB (3 dims)."""

    model_id = "test-mean"
    dim = 3
    default_min_similarity = 0.9

    def __init__(self):
        self.batches: list[int] = []

    def embed_images(self, images):
        self.batches.append(len(images))
        rows = [
            np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0) - 127.5
            for im in images
        ]
        return l2_normalize(np.stack(rows))


register_backend("test-mean", lambda arg: MeanColorBackend())


@pytest.fixture
def mean_backend() -> MeanColorBackend:
    return MeanColorBackend()
