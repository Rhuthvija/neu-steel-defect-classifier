"""Check uploaded files before they reach the model.

Two kinds of feedback:
* errors   (InvalidImageError) -> we refuse the file and say why.
* warnings (list of strings)   -> we still predict, but tell the user the result
  may be unreliable (e.g. a colour photo, which is nothing like NEU data).
Only Pillow + numpy are needed, so this is fast to test.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "MPO"}  # MPO = JPEG from some phone cameras
MIN_SIDE = 32
MAX_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """Raised with a message that is safe to show to the user."""


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v * (1 / 256)).convert("L")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def validate_image_bytes(data: bytes, filename: str, max_mb: float = 10) -> tuple[Image.Image, list[str]]:
    """Return (RGB image, warnings) or raise InvalidImageError."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        shown = ext or "(no extension)"
        raise InvalidImageError(
            f"Unsupported file type {shown}. Please upload a JPG, PNG, BMP or TIFF image."
        )
    if len(data) == 0:
        raise InvalidImageError("The file is empty (0 bytes). Please choose another image.")
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidImageError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {max_mb:g} MB.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # checks the file structure without decoding all pixels
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise InvalidImageError(f"The image is {img.width}x{img.height} pixels, which is too large to process.")
        img.load()
    except InvalidImageError:
        raise
    except Image.DecompressionBombError:
        raise InvalidImageError("The image is too large to process safely.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise InvalidImageError(
            "This file could not be opened as an image. It may be corrupted, or it may be "
            "a different kind of file renamed to look like an image."
        ) from None

    if img.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"The file is really a {img.format} image (despite the {ext} name). "
            "Please upload a JPG, PNG, BMP or TIFF image."
        )

    warnings: list[str] = []
    if getattr(img, "n_frames", 1) > 1:
        warnings.append("The file has several frames/pages; only the first one is used.")

    img = ImageOps.exif_transpose(img) or img
    if min(img.size) < MIN_SIDE:
        raise InvalidImageError(
            f"The image is only {img.width}x{img.height} pixels. Please upload at least {MIN_SIDE}x{MIN_SIDE}."
        )

    rgb = _to_rgb(img)
    warnings.extend(distribution_warnings(rgb))
    return rgb, warnings


def distribution_warnings(rgb: Image.Image) -> list[str]:
    """Cheap checks for images that look nothing like the NEU training data."""
    warnings = []
    arr = np.asarray(rgb.resize((128, 128))).astype(np.float32)
    channel_spread = np.abs(arr - arr.mean(axis=2, keepdims=True)).mean()
    if channel_spread > 8:
        warnings.append(
            "This looks like a colour image. The model was trained only on grayscale "
            "steel-surface close-ups, so this prediction may not be meaningful."
        )
    if arr.std() < 4:
        warnings.append("The image is almost a single flat colour; there may be nothing to classify.")
    ratio = max(rgb.size) / min(rgb.size)
    if ratio > 2:
        warnings.append(
            f"The image is very non-square ({rgb.width}x{rgb.height}) and will be stretched to a square, "
            "which can distort defects."
        )
    return warnings


def load_and_validate_path(path: str | Path, max_mb: float = 10) -> tuple[Image.Image, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"File not found: {path}")
    return validate_image_bytes(path.read_bytes(), path.name, max_mb)
