"""Deterministic film-inspired rendering engine (film-synth-v1). NumPy + OpenCV, float32 sRGB-encoded values in [0, 1].

Every operation is driven by one documented, dimensionless parameter, and a parameter of 0 (or False) is a no-op.
Given the same image, parameters and seed, the output is bit-identical.

Order: exposure (nuisance) -> white balance -> monochrome mix -> tone (black point, shadow lift, contrast, highlight
compression) -> saturation -> split toning -> local contrast -> softness -> halation / bloom -> vignette -> grain.

| parameter | range | effect at the extreme |
|---|---|---|
| temperature_shift | [-1, 1] | R gain x(1 + 0.12 t), B gain x(1 - 0.12 t) |
| tint | [-1, 1] | G gain x(1 - 0.08 t) (positive = magenta) |
| saturation_adjustment | [-1, 1] | chroma x(1 + 0.7 s) about luma |
| contrast_adjustment | [-1, 1] | > 0 blends toward a logistic S-curve (gain 10); < 0 linear compression about 0.5 by up to 45 % |
| shadow_lift | [0, 1] | output black raised to 0.12 l |
| black_point | [0, 1] | input values below 0.06 b crushed to 0 (rescaled) |
| highlight_compression | [0, 1] | soft shoulder above knee 1 - 0.35 h, so the output never reaches 1 |
| local_contrast | [-1, 1] | x + 0.5 lc (x - blur_sigma=2 % of the short side) |
| softness | [0, 1] | Gaussian blur sigma 0.3 + 1.2 s px (applied when > 0.02) |
| halation | [0, 1] | red-orange glow from luma > 0.8, sigma 1.5 % of the short side, strength 0.6 h |
| bloom | [0, 1] | neutral glow of highlights, strength 0.35 b |
| vignette | [0, 1] | x(1 - 0.35 v r^2), r = 1 at the corners |
| shadow/highlight_tint_strength, _hue | [0, 1], deg | additive tint of up to 0.06 weighted by (1 - L)^2 or L^2 |
| grain_strength | [0, 1] | luma noise standard deviation 0.045 g |
| grain_size | [0, 1] | noise blurred with sigma 0.35 + 1.4 z px, then renormalised |
| chroma_grain | [0, 1] | per-channel noise at 0.5 c x the luma sigma |
| grain_midtone_weight | [0, 1] | amplitude x((1 - w) + 4 w L(1 - L)) |
| monochrome | bool | luma with randomised channel weights (filter-like) |
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

PARAM_NAMES = ["temperature_shift", "tint", "saturation_adjustment", "contrast_adjustment", "shadow_lift", "black_point",
               "highlight_compression", "local_contrast", "softness", "halation", "bloom", "vignette",
               "shadow_tint_hue", "shadow_tint_strength", "highlight_tint_hue", "highlight_tint_strength",
               "grain_strength", "grain_size", "chroma_grain", "grain_midtone_weight"]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luma(x):
    return x @ LUMA


def _blur(x, sigma):
    return cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


def _hue_rgb(deg):
    h = np.deg2rad(deg)
    v = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)], dtype=np.float32)
    return v / (np.abs(v).max() + 1e-6)


def render(img_u8: np.ndarray, p: dict, seed: int, exposure_ev: float = 0.0) -> np.ndarray:
    """Apply parameters ``p`` (missing = neutral) to an RGB uint8 image. Returns RGB uint8."""
    g = lambda k, d=0.0: float(p.get(k, d))  # noqa: E731
    rng = np.random.default_rng(seed)
    x = img_u8.astype(np.float32) / 255.0
    short = min(x.shape[:2])
    if exposure_ev:
        x = np.clip(x * (2.0 ** exposure_ev), 0, 1)
    # white balance
    t, ti = g("temperature_shift"), g("tint")
    if t or ti:
        x = x * np.array([1 + 0.12 * t, 1 - 0.08 * ti, 1 - 0.12 * t], dtype=np.float32)
    if p.get("monochrome"):
        w = rng.dirichlet([6, 8, 3]).astype(np.float32)            # filter-like channel mix, varies per variant
        x = np.repeat((x @ w)[..., None], 3, axis=2)
    x = np.clip(x, 0, 1)
    # tone
    b = g("black_point")
    if b:
        bp = 0.06 * b
        x = np.clip((x - bp) / (1 - bp), 0, 1)
    c = g("contrast_adjustment")
    if c > 0:
        k = 10.0
        s = 1 / (1 + np.exp(-k * (x - 0.5)))
        s = (s - 1 / (1 + np.exp(k / 2))) / (1 / (1 + np.exp(-k / 2)) - 1 / (1 + np.exp(k / 2)))
        x = (1 - c) * x + c * s
    elif c < 0:
        x = 0.5 + (x - 0.5) * (1 + 0.45 * c)
    h = g("highlight_compression")
    if h:
        knee = 1 - 0.35 * h
        over = np.maximum(x - knee, 0)
        x = np.where(x > knee, knee + (1 - knee) * np.tanh(over / (1 - knee)) * 0.97, x)
    lift = g("shadow_lift")
    if lift:
        x = 0.12 * lift + x * (1 - 0.12 * lift)
    # colour
    sat = g("saturation_adjustment")
    if sat and not p.get("monochrome"):
        L = _luma(x)[..., None]
        x = L + (x - L) * (1 + 0.7 * sat)
    for kind in ("shadow", "highlight"):
        st = g(f"{kind}_tint_strength")
        if st:
            L = np.clip(_luma(x), 0, 1)[..., None]
            wgt = (1 - L) ** 2 if kind == "shadow" else L ** 2
            x = x + 0.06 * st * wgt * _hue_rgb(g(f"{kind}_tint_hue"))
    x = np.clip(x, 0, 1).astype(np.float32)
    # texture
    lc = g("local_contrast")
    if lc:
        x = x + 0.5 * lc * (x - _blur(x, 0.02 * short))
    so = g("softness")
    if so > 0.02:
        x = _blur(x, 0.3 + 1.2 * so)
    ha, bl = g("halation"), g("bloom")
    if ha or bl:
        m = np.clip((_luma(np.clip(x, 0, 1)) - 0.8) / 0.2, 0, 1)[..., None] * np.ones(3, np.float32)
        glow = _blur(m, 0.015 * short)
        x = x + glow * (0.6 * ha * np.array([1.0, 0.35, 0.1], np.float32) + 0.35 * bl)
    v = g("vignette")
    if v:
        H, W = x.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 / 2 + ((yy - H / 2) / (H / 2)) ** 2 / 2
        x = x * (1 - 0.35 * v * r2)[..., None]
    gs = g("grain_strength")
    if gs:
        L = np.clip(_luma(np.clip(x, 0, 1)), 0, 1)
        wmid = g("grain_midtone_weight")
        amp = 0.045 * gs * ((1 - wmid) + 4 * wmid * L * (1 - L))
        sig = 0.35 + 1.4 * g("grain_size")
        n = _blur(rng.standard_normal(L.shape).astype(np.float32), sig)
        n /= n.std() + 1e-8
        x = x + (amp * n)[..., None]
        cg = g("chroma_grain")
        if cg and not p.get("monochrome"):
            nc = _blur(rng.standard_normal(x.shape).astype(np.float32), sig)
            nc /= nc.std() + 1e-8
            x = x + 0.5 * cg * amp[..., None] * nc
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def prepare(img: Image.Image, short_side: int, crop_fraction: float, resample: str, rng) -> np.ndarray:
    """Nuisance geometry shared by every output: small random crop, then resize to ``short_side``."""
    W, H = img.size
    cf = crop_fraction
    dx, dy = int(W * cf * rng.random()), int(H * cf * rng.random())
    img = img.crop((dx, dy, W - int(W * cf) + dx, H - int(H * cf) + dy))
    s = short_side / min(img.size)
    rs = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}[resample]
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), rs)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def encode_jpeg(arr: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue()
