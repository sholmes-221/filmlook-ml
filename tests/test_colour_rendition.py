"""Colour-rendition descriptor: deterministic, resolution-robust, texture-insensitive, responsive to colour/tone edits."""
import numpy as np
from PIL import Image

from filmlook_infer.colour_rendition import DIM, NAMES, colour_rendition


def _img(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[1], 0:size[0]]
    base = np.stack([x * 0.7, y * 0.9, 255 - x * 0.5], 2) % 256
    return Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))


def test_shape_names_and_determinism():
    im = _img()
    v = colour_rendition(im)
    assert v.shape == (DIM,) and len(NAMES) == DIM == 80
    assert np.array_equal(v, colour_rendition(im))
    assert abs(v[NAMES.index("hue_bin0"):NAMES.index("hue_bin0") + 12].sum() - 1) < 1e-4


def test_resolution_and_grain_robust_but_colour_sensitive():
    im = _img()
    v = colour_rendition(im)
    small = im.resize((160, 120), Image.BILINEAR)
    assert np.abs(colour_rendition(small) - v).max() < 0.05                   # resolution
    rng = np.random.default_rng(1)
    grainy = Image.fromarray(np.clip(np.asarray(im, np.int16) + rng.normal(0, 12, (240, 320, 3)).astype(np.int16), 0, 255).astype(np.uint8))
    d_grain = np.abs(colour_rendition(grainy) - v).max()
    warm = Image.fromarray(np.clip(np.asarray(im, np.float32) * np.array([1.15, 1.0, 0.85]), 0, 255).astype(np.uint8))
    d_warm = np.abs(colour_rendition(warm) - v).max()
    assert d_warm > 3 * d_grain                                               # colour shift dominates grain
    assert colour_rendition(warm)[NAMES.index("cr_band2")] > v[NAMES.index("cr_band2")]
