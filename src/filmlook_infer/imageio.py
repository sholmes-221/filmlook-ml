"""Bounded, validated image loading (identical in training and inference)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MIN_SIDE = 256
MAX_PIXELS = 60_000_000          # decompression-bomb guard
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2


class ImageValidationError(ValueError):
    pass


def limit_size(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale only if needed (Lanczos)."""
    if max(img.size) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def load_image(path, max_side: int | None = None) -> Image.Image:
    """Open, verify, apply EXIF orientation, convert to RGB. Raises ImageValidationError.

    ``path`` may be a filesystem path or a binary file-like object.
    ``max_side``: decode JPEGs at reduced DCT scale (``draft``) then downscale to this long side:
    bounds memory and matches the training resolution.
    """
    try:
        if hasattr(path, "read"):
            data = path.read()
            import io
            opener = lambda: Image.open(io.BytesIO(data))  # noqa: E731
        else:
            p = Path(path)
            opener = lambda: Image.open(p)  # noqa: E731
        with opener() as im:
            im.verify()
        im = opener()
        if im.width * im.height > MAX_PIXELS:
            raise ImageValidationError(f"too large: {im.size}")
        if max_side and im.format == "JPEG" and max(im.size) > 2 * max_side:
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im)
        im.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise ImageValidationError(f"corrupt or unreadable: {e}") from e
    if min(im.size) < MIN_SIDE:
        raise ImageValidationError(f"too small: {im.size}")
    if im.mode not in ("RGB", "L", "RGBA", "P", "CMYK", "I;16", "LA"):
        raise ImageValidationError(f"unsupported mode {im.mode}")
    im = im.convert("RGB")
    return limit_size(im, max_side) if max_side else im
