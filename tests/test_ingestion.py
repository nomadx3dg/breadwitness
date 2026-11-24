import pytest
from PIL import Image

from breadwitness.ingestion import ingest_image, iter_image_files
from breadwitness.preprocessing import InvalidImageError, load_image

from .conftest import make_image


def test_iter_image_files_is_recursive_and_filters_extensions(image_dir):
    names = [p.name for p in iter_image_files(image_dir)]
    assert set(names) == {"a.png", "b.jpg", "c.bmp", "corrupt.jpg"}
    assert "notes.txt" not in names


def test_extensions_are_case_insensitive(tmp_path):
    make_image(1).save(tmp_path / "UPPER.JPG")
    assert [p.name for p in iter_image_files(tmp_path)] == ["UPPER.JPG"]


def test_supported_formats_ingest(tmp_path):
    for name in ("a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp"):
        make_image(5).save(tmp_path / name)
        record = ingest_image(tmp_path / name)
        assert (record.width, record.height) == (256, 256)
        assert record.file_size > 0 and len(record.sha256) == 64


def test_corrupt_file_raises(image_dir):
    with pytest.raises(InvalidImageError):
        ingest_image(image_dir / "corrupt.jpg")


def test_truncated_image_raises(tmp_path):
    good = tmp_path / "good.png"
    make_image(4).save(good)
    bad = tmp_path / "truncated.png"
    bad.write_bytes(good.read_bytes()[:50])
    with pytest.raises(InvalidImageError):
        ingest_image(bad)


def test_exif_orientation_is_normalized(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    make_image(1, size=(200, 100)).save(path, exif=exif)
    loaded = load_image(path)
    assert (loaded.width, loaded.height) == (100, 200)


def test_transparent_png_loads(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(path)
    assert ingest_image(path).format == "PNG"
