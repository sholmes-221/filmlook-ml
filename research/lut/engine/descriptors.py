"""Photo descriptors for film-style matching (deterministic; NumPy + OpenCV; no learned parts).

`describe(img_u8) -> np.ndarray` of length DIM with names in NAMES, grouped:
  colour  (COLOUR_IDX): CIELAB L histogram (16), chroma-weighted (a, b) histogram 10x10 (100), hue histogram x 3
          lightness bands (36), HSV saturation/value percentiles (10), RGB channel correlations and std ratios (5),
          warm/cool balance (mean b*, mean R-B, 2), shadow / midtone / highlight colour bias (a, b per band, 6),
          colourfulness (Hasler-Suesstrunk) and chroma percentiles (5)                             -> 180
  tone    (TONE_IDX): luminance percentiles 1/5/25/50/75/95/99 (7), contrast (std L), dynamic range (p99-p1),
          highlight roll-off, shadow compression, midtone share, black level, white level, per-channel black/white
          (6)                                                                                       -> 19
  texture (TEXTURE_IDX): local contrast, high-frequency energy ratio, noise/grain indicator, edge density,
          Laplacian variance                                                                        -> 5
Everything is computed on a copy whose short side is 384 px (INTER_AREA), so the texture indicators are relative to
that scale. A colour LUT (and every style in the database) cannot describe grain, halation or texture: the texture
group is reported for information and is never used for matching.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

SHORT_SIDE = 384

# sRGB (D65) -> CIELAB, D50-adapted like the forward model (Bradford), so LUT and photo descriptors agree
_M_SRGB_TO_XYZ_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                               [0.2225045, 0.7168786, 0.0606169],
                               [0.0139322, 0.0971045, 0.7141733]], np.float32)
_WHITE_D50 = np.array([0.96422, 1.0, 0.82521], np.float32)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    v = np.asarray(rgb, np.float32)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    r = (lin @ _M_SRGB_TO_XYZ_D50.T) / _WHITE_D50
    f = np.where(r > 216 / 24389, np.cbrt(np.maximum(r, 0)), (24389 / 27 * r + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def standardise(img_u8: np.ndarray, short: int = SHORT_SIDE) -> np.ndarray:
    """uint8 HxWx3 RGB -> uint8 with the short side at `short` px (area resampling; never upsampled)."""
    h, w = img_u8.shape[:2]
    s = short / min(h, w)
    if s >= 1:
        return img_u8
    size = (max(8, round(w * s)), max(8, round(h * s)))
    if cv2 is not None:
        return cv2.resize(img_u8, size, interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(Image.fromarray(img_u8).resize(size, Image.BOX))


def _pct(x, ps):
    return np.percentile(x, ps).astype(np.float32)


def _names():
    n = [f"L_hist{i}" for i in range(16)] + [f"ab_hist{i}" for i in range(100)] + [f"hue{b}_{i}" for b in ("sh", "mid", "hi") for i in range(12)]
    n += [f"sat_p{p}" for p in (5, 25, 50, 75, 95)] + [f"val_p{p}" for p in (5, 25, 50, 75, 95)]
    n += ["corr_rg", "corr_gb", "corr_rb", "std_r_over_g", "std_b_over_g", "warm_b_mean", "warm_r_minus_b"]
    n += ["shadow_a", "shadow_b", "mid_a", "mid_b", "high_a", "high_b", "colourfulness", "chroma_p25", "chroma_p50", "chroma_p75", "chroma_p95"]
    n += [f"L_p{p}" for p in (1, 5, 25, 50, 75, 95, 99)] + ["contrast_L_std", "dynamic_range", "highlight_rolloff", "shadow_compression",
                                                             "midtone_share", "black_level", "white_level"] + [f"{c}_black" for c in "rgb"] + [f"{c}_white" for c in "rgb"]
    n += ["local_contrast", "hf_energy_ratio", "noise_grain", "edge_density", "laplacian_var"]
    return n


NAMES = _names()
DIM = len(NAMES)
COLOUR_IDX = np.arange(0, 180)
TONE_IDX = np.arange(180, 200)
TEXTURE_IDX = np.arange(200, 205)
assert DIM == 205


def describe(img_u8: np.ndarray, with_texture: bool = True) -> np.ndarray:
    im = standardise(np.ascontiguousarray(img_u8))
    rgb = im.astype(np.float32) / 255
    flat = rgb.reshape(-1, 3)
    lab = srgb_to_lab(flat)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    c = np.hypot(a, b)
    out = []
    hL, _ = np.histogram(L, 16, (0, 100))
    out += list(hL / max(hL.sum(), 1))
    hab, _, _ = np.histogram2d(a, b, 10, [[-50, 50], [-50, 50]], weights=c)
    out += list(hab.ravel() / max(hab.sum(), 1e-6))
    hue = np.degrees(np.arctan2(b, a)) % 360
    band = np.digitize(L, [35, 65])
    for k in range(3):
        m = (band == k) & (c > 5)
        hh = np.histogram(hue[m], 12, (0, 360), weights=c[m])[0] if m.any() else np.zeros(12)
        out += list(hh / max(hh.sum(), 1e-6))
    mx, mn = flat.max(1), flat.min(1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    out += list(_pct(sat, (5, 25, 50, 75, 95))) + list(_pct(mx, (5, 25, 50, 75, 95)))
    sd = flat.std(0) + 1e-6
    cc = np.corrcoef(flat.T) if flat.std(0).min() > 1e-5 else np.eye(3)
    out += [float(cc[0, 1]), float(cc[1, 2]), float(cc[0, 2]), float(sd[0] / sd[1]), float(sd[2] / sd[1])]
    out += [float(b.mean() / 50), float((flat[:, 0] - flat[:, 2]).mean())]
    for lo, hi in ((0, 30), (30, 70), (70, 101)):
        m = (L >= lo) & (L < hi)
        out += [float(a[m].mean() / 20), float(b[m].mean() / 20)] if m.sum() > 20 else [0.0, 0.0]
    rg = flat[:, 0] - flat[:, 1]
    yb = 0.5 * (flat[:, 0] + flat[:, 1]) - flat[:, 2]
    colourfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    out += [colourfulness] + list(_pct(c, (25, 50, 75, 95)) / 50)
    # tone
    Lp = _pct(L, (1, 5, 25, 50, 75, 95, 99)) / 100
    out += list(Lp)
    out += [float(L.std() / 100), float(Lp[6] - Lp[0]), float((Lp[6] - Lp[5]) / max(Lp[5] - Lp[4], 1e-3)), float((Lp[1] - Lp[0]) / max(Lp[2] - Lp[1], 1e-3)),
            float(((L > 25) & (L < 75)).mean()), float(np.percentile(mn, 0.5)), float(np.percentile(mx, 99.5))]
    out += list(np.percentile(flat, 0.5, axis=0)) + list(np.percentile(flat, 99.5, axis=0))
    # texture
    if with_texture and cv2 is not None:
        g = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        blur = cv2.GaussianBlur(g, (0, 0), 1.5)
        local_sd = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 3) ** 2, 0))
        hp = g - blur
        grad = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1))
        flatmask = grad < np.percentile(grad, 30)
        noise = float(1.4826 * np.median(np.abs(hp[flatmask] - np.median(hp[flatmask])))) if flatmask.sum() > 100 else 0.0
        edges = cv2.Canny((g * 255).astype(np.uint8), 60, 140)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        out += [float(local_sd.mean()), float(hp.var() / max(g.var(), 1e-6)), noise, float((edges > 0).mean()), float(lap.var())]
    else:
        out += [0.0] * 5
    return np.asarray(out, np.float32)


def describe_pixels(pixels_u8_or_float: np.ndarray) -> np.ndarray:
    """Descriptor of a pixel cloud (N, 3) treated as an image (texture group = 0)."""
    px = np.asarray(pixels_u8_or_float)
    if px.dtype != np.uint8:
        px = (np.clip(px, 0, 1) * 255 + 0.5).astype(np.uint8)
    n = len(px)
    w = int(np.sqrt(n))
    return describe(px[: w * w].reshape(w, w, 3), with_texture=False)


def neutralise(img_u8: np.ndarray) -> np.ndarray:
    """Crude 'undo the look': grey-world white balance in linear light + 0.5/99.5 percentile levels."""
    v = img_u8.astype(np.float32) / 255
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    m = lin.reshape(-1, 3).mean(0)
    lin = np.clip(lin * (m.mean() / np.maximum(m, 1e-6)), 0, 1)
    lo, hi = np.percentile(lin, 0.5), np.percentile(lin, 99.5)
    lin = np.clip((lin - lo) / max(hi - lo, 1e-6), 0, 1)
    enc = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (np.clip(enc, 0, 1) * 255 + 0.5).astype(np.uint8)
