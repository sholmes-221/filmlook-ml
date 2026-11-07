"""Torch-free DINOv2 preprocessing (identical in training and deployment)."""
from __future__ import annotations

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def dinov2_pixels(img: Image.Image, size: int = 224, hflip: bool = False) -> np.ndarray:
    """Resize shortest edge to size*8/7 (bicubic) -> center crop -> ImageNet normalise. (3, size, size)."""
    resize_short = round(size * 256 / 224)
    img = img.convert("RGB")
    w, h = img.size
    s = resize_short / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1).copy()
