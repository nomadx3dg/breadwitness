"""Shared test helpers: deterministic synthetic images."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


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
