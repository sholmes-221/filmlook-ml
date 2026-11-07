"""Controlled edit families applied to a base photograph.

Each edit produces a variant whose effect on specific attributes is known *relative to
the other variants of the same photograph*. That gives us pairwise supervision without
any third-party labels. All variants of one base share a ``family_id`` and are always
placed in the same split.

Caveat: ``film_emulation`` is our own approximation of film rendering. Training only on
it would teach the model "our emulation = film". Synthetic orderings therefore get a
lower loss weight than human labels and supervise *only* the attributes each edit
clearly affects.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _luma(a: np.ndarray) -> np.ndarray:
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def filmic_curve(x: np.ndarray, toe: float = 0.03, shoulder: float = 2.2, white: float = 0.97) -> np.ndarray:
    """Soft shoulder + slight toe lift. Maps [0,1] -> [toe, white]; never clips."""
    y = (1 - np.exp(-shoulder * np.clip(x, 0, 1))) / (1 - np.exp(-shoulder))
    return toe + (white - toe) * y


def film_grain(shape, rng: np.random.Generator, strength: float, size: float = 0.9) -> np.ndarray:
    g = rng.standard_normal(shape[:2]).astype(np.float32)
    g = ndimage.gaussian_filter(g, size)  # clumped, spatially correlated
    return g / (g.std() + 1e-6) * strength


def _unsharp(a: np.ndarray, amount: float, radius: float) -> np.ndarray:
    blur = ndimage.gaussian_filter(a, (radius, radius, 0))
    return a + amount * (a - blur)


def edit_film_emulation(img: Image.Image, rng: np.random.Generator, mono: bool = False) -> Image.Image:
    a = _arr(img)
    a = ndimage.gaussian_filter(a, (0.6, 0.6, 0))           # gentle acutance
    if mono:
        y = _luma(a)
        a = np.repeat(y[..., None], 3, -1)
    a = filmic_curve(a)
    y = _luma(a)
    # halation: faint red bloom around bright areas
    bright = np.clip((y - 0.85) * 6, 0, 1)
    if not mono:
        a[..., 0] += 0.06 * ndimage.gaussian_filter(bright, 6)
    # grain peaks in mid-tones, monochromatic (dye clouds overlap across layers)
    mid_w = 0.35 + 0.65 * (1 - np.abs(y - 0.5) * 2)
    g = film_grain(a.shape, rng, 0.035 if mono else 0.022)
    a = a + (g * mid_w)[..., None]
    return _img(a)


def edit_digital_clean_sharp(img, rng):
    a = _arr(img) * 1.18            # push exposure: hard highlight clipping
    return _img(_unsharp(np.clip(a, 0, 1), 0.8, 1.2))


def edit_hdr_overprocessed(img, rng):
    a = _arr(img)
    y = _luma(a)
    base = ndimage.gaussian_filter(y, 15)
    detail = y - base
    y2 = 0.5 + 0.45 * (base - base.mean()) + 2.4 * detail      # compress global, boost local
    a = a * (y2 / (y + 1e-3))[..., None]
    m = a.mean(-1, keepdims=True)
    a = m + 1.7 * (a - m)                                       # oversaturate
    return _img(a)


def edit_fake_grain_filter(img, rng):
    a = _arr(img)
    a = a * np.array([1.08, 1.0, 0.86]) + np.array([0.03, 0.01, 0.0])  # warm + "orange" highlights
    a = 0.08 + 0.92 * a                                                  # faded blacks
    a = a + rng.normal(0, 0.045, a.shape)                               # per-pixel RGB noise
    return _img(a)


def edit_posterized(img, rng):
    a = _arr(img)
    return _img(np.round(a * 20) / 20)


def edit_oversharpened(img, rng):
    return _img(_unsharp(_arr(img), 2.5, 1.0))


def edit_bw_harsh_digital(img, rng):
    y = _luma(_arr(img)) * 1.2
    a = np.repeat(np.clip(y, 0, 1)[..., None], 3, -1)
    return _img(_unsharp(a, 1.2, 1.0))


EDITS = {
    "film_emulation": lambda im, r: edit_film_emulation(im, r, mono=False),
    "bw_film_emulation": lambda im, r: edit_film_emulation(im, r, mono=True),
    "digital_clean_sharp": edit_digital_clean_sharp,
    "hdr_overprocessed": edit_hdr_overprocessed,
    "fake_grain_filter": edit_fake_grain_filter,
    "posterized": edit_posterized,
    "oversharpened": edit_oversharpened,
    "bw_harsh_digital": edit_bw_harsh_digital,
}

# Partial absolute targets for attributes each edit clearly determines (v1 rubric).
EDIT_TARGETS: dict[str, dict[str, float]] = {
    "film_emulation": {"highlight_rolloff": 0.85, "microcontrast": 0.8, "grain_naturalness": 0.8},
    "bw_film_emulation": {"highlight_rolloff": 0.85, "microcontrast": 0.8, "grain_naturalness": 0.8},
    "digital_clean_sharp": {"highlight_rolloff": 0.25, "microcontrast": 0.4},
    "hdr_overprocessed": {"highlight_rolloff": 0.2, "tonal_depth": 0.15, "color_response": 0.2,
                          "film_coherence": 0.1},
    "fake_grain_filter": {"grain_naturalness": 0.15, "color_response": 0.45, "film_coherence": 0.3},
    "posterized": {"highlight_rolloff": 0.15, "tonal_depth": 0.2},
    "oversharpened": {"microcontrast": 0.1},
    "bw_harsh_digital": {"highlight_rolloff": 0.25, "microcontrast": 0.3, "tonal_depth": 0.3},
}

# (winner_edit, loser_edit, attribute) - relative judgements within one family. Edits are NOT
# assumed better than originals in general: only degradations are ordered against the original.
EDIT_PAIRS: list[tuple[str, str, str]] = [
    ("film_emulation", "digital_clean_sharp", "highlight_rolloff"),
    ("film_emulation", "digital_clean_sharp", "microcontrast"),
    ("film_emulation", "hdr_overprocessed", "highlight_rolloff"),
    ("film_emulation", "hdr_overprocessed", "tonal_depth"),
    ("film_emulation", "hdr_overprocessed", "color_response"),
    ("film_emulation", "hdr_overprocessed", "film_score"),
    ("film_emulation", "fake_grain_filter", "grain_naturalness"),
    ("film_emulation", "fake_grain_filter", "film_score"),
    ("film_emulation", "posterized", "highlight_rolloff"),
    ("film_emulation", "oversharpened", "microcontrast"),
    ("film_emulation", "oversharpened", "film_score"),
    ("bw_film_emulation", "bw_harsh_digital", "highlight_rolloff"),
    ("bw_film_emulation", "bw_harsh_digital", "microcontrast"),
    ("bw_film_emulation", "bw_harsh_digital", "film_score"),
    ("original", "posterized", "highlight_rolloff"),
    ("original", "posterized", "tonal_depth"),
    ("original", "oversharpened", "microcontrast"),
    ("original", "hdr_overprocessed", "tonal_depth"),
    ("original", "hdr_overprocessed", "highlight_rolloff"),
]

# Mapping edit types -> evaluation categories from the spec.
EVAL_CATEGORY = {
    "original": "source_photo (capture medium unknown)",
    "film_emulation": "subtle film-like digital (our emulation)",
    "bw_film_emulation": "black-and-white analog aesthetic (our emulation)",
    "digital_clean_sharp": "digital, clipped + sharpened",
    "hdr_overprocessed": "overprocessed HDR",
    "fake_grain_filter": "digital film-filter with artificial grain",
    "posterized": "banding / posterisation",
    "oversharpened": "over-sharpened digital",
    "bw_harsh_digital": "harsh digital black-and-white",
}


def apply_edit(img: Image.Image, edit: str, seed: int) -> Image.Image:
    if edit == "original":
        return img.convert("RGB")
    return EDITS[edit](img, np.random.default_rng(seed))
