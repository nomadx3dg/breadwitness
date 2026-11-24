"""Image loading, validation and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(Exception):
    """Raised when a file cannot be read as a valid image."""


@dataclass
class LoadedImage:
    """A validated, orientation-normalized RGB image plus source metadata."""

    image: Image.Image
    width: int
    height: int
    format: str


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, compositing transparency onto white."""
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def load_image(path: Path) -> LoadedImage:
    """Validate and load an image, applying EXIF orientation.

    Reported width/height are those of the orientation-corrected image.
    Raises InvalidImageError for corrupted or unreadable files.
    """
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as img:
            fmt = (img.format or "UNKNOWN").upper()
            if fmt == "MPO":  # multi-picture JPEG variant
                fmt = "JPEG"
            img.load()
            oriented = ImageOps.exif_transpose(img)
            rgb = _to_rgb(oriented)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        EOFError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError(f"{path}: {exc}") from exc
    width, height = rgb.size
    return LoadedImage(image=rgb, width=width, height=height, format=fmt)
