"""Build a tiny synthetic library, index it, and search it with the Python API.

Run from the repository root:  python examples/quickstart.py
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from breadwitness.database import Database
from breadwitness.indexing import index_directory
from breadwitness.search import search_image


def make_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks, "RGB").resize((256, 256), Image.Resampling.BICUBIC)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    library = root / "library"
    library.mkdir()
    for seed in range(5):
        make_image(seed).save(library / f"image_{seed}.png")

    query = root / "query.jpg"  # a resized, recompressed copy of image_3
    make_image(3).resize((120, 120)).save(query, quality=50)

    with Database(root / "index.db") as db:
        report = index_directory(db, library)
        print(f"Indexed {report.added} images")
        response = search_image(db, query, limit=3)

    for result in response.results:
        print(
            f"{result.record.filename}: {result.method.value}, "
            f"pHash distance {result.phash_distance}, score {result.score:.3f}"
        )
