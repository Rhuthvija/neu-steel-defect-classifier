"""Upload validation: bad files must be rejected with a clear message."""
import io

import numpy as np
import pytest
from PIL import Image

from src.validation import InvalidImageError, validate_image_bytes


def _png_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _steel_like(size=(200, 200)) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(60, 200, size=size[::-1], dtype=np.uint8), mode="L")


def test_valid_grayscale_png_is_accepted_and_converted_to_rgb():
    img, warnings = validate_image_bytes(_png_bytes(_steel_like()), "ok.png")
    assert img.mode == "RGB" and img.size == (200, 200)
    assert warnings == []


@pytest.mark.parametrize("fmt,ext", [("JPEG", ".jpg"), ("BMP", ".bmp"), ("TIFF", ".tif")])
def test_other_supported_formats(fmt, ext):
    img, _ = validate_image_bytes(_png_bytes(_steel_like(), fmt), f"x{ext}")
    assert img.mode == "RGB"


def test_unsupported_extension_rejected():
    with pytest.raises(InvalidImageError, match="Unsupported file type .gif"):
        validate_image_bytes(b"GIF89a....", "anim.gif")


def test_missing_extension_rejected():
    with pytest.raises(InvalidImageError, match="no extension"):
        validate_image_bytes(_png_bytes(_steel_like()), "noext")


def test_empty_file_rejected():
    with pytest.raises(InvalidImageError, match="empty"):
        validate_image_bytes(b"", "empty.jpg")


def test_text_file_renamed_to_jpg_rejected():
    with pytest.raises(InvalidImageError, match="could not be opened"):
        validate_image_bytes(b"this is not an image at all", "fake.jpg")


def test_truncated_image_rejected():
    data = _png_bytes(_steel_like(), "JPEG")
    with pytest.raises(InvalidImageError):
        validate_image_bytes(data[: len(data) // 3], "broken.jpg")


def test_real_format_mismatch_rejected():
    gif = io.BytesIO()
    _steel_like().save(gif, format="GIF")
    with pytest.raises(InvalidImageError, match="really a GIF"):
        validate_image_bytes(gif.getvalue(), "sneaky.png")


def test_too_small_rejected():
    with pytest.raises(InvalidImageError, match="at least"):
        validate_image_bytes(_png_bytes(_steel_like((10, 10))), "tiny.png")


def test_too_big_file_rejected():
    with pytest.raises(InvalidImageError, match="limit"):
        validate_image_bytes(b"\x00" * (2 * 1024 * 1024), "big.png", max_mb=1)


def test_colour_photo_gives_warning_not_error():
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[..., 0] = 220  # strongly red
    arr[..., 2] = np.linspace(0, 255, 200, dtype=np.uint8)
    img, warnings = validate_image_bytes(_png_bytes(Image.fromarray(arr)), "photo.png")
    assert img.mode == "RGB"
    assert any("colour" in w for w in warnings)


def test_flat_image_warns():
    _, warnings = validate_image_bytes(_png_bytes(Image.new("L", (200, 200), 128)), "flat.png")
    assert any("flat" in w for w in warnings)


def test_rgba_is_flattened_to_rgb():
    rgba = _steel_like().convert("RGBA")
    img, _ = validate_image_bytes(_png_bytes(rgba), "alpha.png")
    assert img.mode == "RGB"


def test_16bit_png_is_handled():
    arr = (np.random.default_rng(1).integers(0, 65535, (200, 200))).astype(np.uint16)
    img16 = Image.fromarray(arr)  # uint16 array -> 16-bit image
    assert img16.mode.startswith("I")
    img, _ = validate_image_bytes(_png_bytes(img16), "deep.png")
    assert img.mode == "RGB"
