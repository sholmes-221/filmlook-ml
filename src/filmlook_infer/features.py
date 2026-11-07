"""Measured photographic statistics (numpy/scipy/Pillow only, CPU-cheap).

Single source of truth: training (filmlook_ml) imports this module, so train/serve features
cannot drift.

Two roles:
1. *Texture branch* for the model. DINOv2 at 224px discards most grain/acutance
   information, so we compute texture statistics on a native-resolution crop and feed
   them alongside the embedding.
2. Inputs to the transparent heuristic teacher and to explanation templates.

Every value is a plain float; ``feature_vector`` returns a fixed-order float32 array.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

FEATURE_NAMES = [
    # tone
    "clip_high", "clip_low", "p01", "p99", "contrast_iqr", "y_std", "y_mean",
    "highlight_shoulder", "hist_gap_ratio", "local_contrast_ratio",
    # colour
    "sat_mean", "sat_p95", "colorfulness", "warmth", "green_shift", "is_mono", "sat_clip",
    # acutance
    "lap_var", "hf_ratio", "overshoot",
    # grain (native-resolution crop)
    "noise_std", "noise_kurtosis", "noise_shadow_ratio", "noise_mid_ratio",
    "chroma_noise_ratio", "noise_autocorr1", "noise_autocorr2", "blockiness",
]
N_FEATURES = len(FEATURE_NAMES)


def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _resize_long(img: Image.Image, side: int) -> Image.Image:
    w, h = img.size
    s = side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def center_crop_native(img: Image.Image, size: int = 512) -> Image.Image:
    """Crop at native resolution (no resampling) so grain survives."""
    w, h = img.size
    cw, ch = min(size, w), min(size, h)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def tone_color_features(img: Image.Image) -> dict[str, float]:
    rgb = _to_float(_resize_long(img, 512))
    y = _luma(rgb)
    f: dict[str, float] = {}
    f["clip_high"] = float((y > 0.985).mean())
    f["clip_low"] = float((y < 0.015).mean())
    p01, p25, p75, p99 = np.percentile(y, [1, 25, 75, 99])
    f["p01"], f["p99"] = float(p01), float(p99)
    f["contrast_iqr"] = float(p75 - p25)
    f["y_std"], f["y_mean"] = float(y.std()), float(y.mean())
    # Highlight shoulder: of pixels above 0.8, what fraction sit in the top 3%?
    # Film shoulders compress gradually (low ratio); digital clips pile up at 1.0.
    hi = y[y > 0.8]
    f["highlight_shoulder"] = float((hi > 0.97).mean()) if hi.size > 50 else 0.0
    # Histogram combing/posterisation: per-channel, on a STRIDED (non-interpolated)
    # sample so resampling does not fill gaps.
    full = np.asarray(img.convert("RGB"))
    step = max(1, int(np.sqrt(full.shape[0] * full.shape[1] / 400_000)))
    sub = full[::step, ::step].reshape(-1, 3)
    gaps = []
    for ch in range(3):
        h = np.bincount(sub[:, ch], minlength=256)
        lo_i, hi_i = np.percentile(sub[:, ch], [1, 99]).astype(int)
        used = h[lo_i:hi_i + 1]
        gaps.append(float((used == 0).mean()) if used.size > 10 else 0.0)
    f["hist_gap_ratio"] = float(np.median(gaps))
    # HDR / local tone-mapping: strong local contrast relative to global spread,
    # measured after light smoothing so grain does not count as "local contrast".
    ys = ndimage.gaussian_filter(y, 2.0)
    local = np.abs(ys - ndimage.gaussian_filter(ys, 12)).mean()
    f["local_contrast_ratio"] = float(local / (y.std() + 1e-4))

    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    f["sat_mean"] = float(sat.mean())
    f["sat_p95"] = float(np.percentile(sat, 95))
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    f["colorfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    f["warmth"] = float((rgb[..., 0] - rgb[..., 2]).mean())
    shadows = y < np.percentile(y, 30)
    s = rgb[shadows] if shadows.any() else rgb.reshape(-1, 3)
    f["green_shift"] = float((s[:, 1] - 0.5 * (s[:, 0] + s[:, 2])).mean())
    f["is_mono"] = float(f["sat_p95"] < 0.08)
    f["sat_clip"] = float(((rgb > 0.99) | (rgb < 0.01)).any(-1).mean() - (f["clip_high"] + f["clip_low"]))
    f["sat_clip"] = max(0.0, f["sat_clip"])

    # Acutance on a 1024px version. lap_var (softness) uses the raw image; halo and
    # high-frequency measures use a 3x3 median-filtered copy, which removes grain but
    # keeps edges and the 2-4px overshoot halos produced by unsharp masking.
    yl = _luma(_to_float(_resize_long(img, 1024)))
    f["lap_var"] = float(ndimage.laplace(ndimage.median_filter(yl, 3)).var())
    ym = ndimage.median_filter(yl, 3)
    b1, b3 = ndimage.gaussian_filter(ym, 1.0), ndimage.gaussian_filter(ym, 3.0)
    f["hf_ratio"] = float(np.abs(ym - b1).mean() / (np.abs(b1 - b3).mean() + 1e-5))
    # Sharpening halos near strong edges: overshoot beyond a wider smooth envelope,
    # normalised by local edge contrast.
    gm = ndimage.gaussian_gradient_magnitude(b3, 1.0)
    edges = ndimage.binary_dilation(gm > np.percentile(gm, 95), iterations=3)
    env = ndimage.gaussian_filter(ym, 2.5)
    mxf, mnf = ndimage.maximum_filter(env, 7), ndimage.minimum_filter(env, 7)
    over = np.maximum(ym - mxf, 0) + np.maximum(mnf - ym, 0)
    # Grain raises overshoot everywhere; halos only at edges -> subtract flat-area level.
    flat = gm < np.percentile(gm, 40)
    edge_contrast = float((mxf - mnf)[edges].mean()) + 0.02 if edges.any() else 1.0
    halo = (over[edges].mean() - over[flat].mean()) if edges.any() and flat.any() else 0.0
    f["overshoot"] = float(max(halo, 0.0) / edge_contrast * 100)
    return f


def grain_features(img: Image.Image, crop: int = 512) -> dict[str, float]:
    rgb = _to_float(center_crop_native(img, crop))
    y = _luma(rgb)
    hp = y - ndimage.gaussian_filter(y, 1.5)
    # flat regions only: low structural gradient
    gm = ndimage.gaussian_gradient_magnitude(ndimage.gaussian_filter(y, 2.0), 1.0)
    flat = gm < np.percentile(gm, 40)
    n = hp[flat] if flat.sum() > 200 else hp.ravel()
    f: dict[str, float] = {}
    std = float(n.std())
    f["noise_std"] = std
    f["noise_kurtosis"] = float(((n - n.mean()) ** 4).mean() / (n.var() ** 2 + 1e-12))
    ys = ndimage.gaussian_filter(y, 2.0)

    def band(lo: float, hi: float) -> float:
        m = flat & (ys >= lo) & (ys < hi)
        return float(hp[m].std()) if m.sum() > 100 else std

    sh, mid, hl = band(0.0, 0.3), band(0.3, 0.7), band(0.7, 1.01)
    f["noise_shadow_ratio"] = sh / (std + 1e-6)
    f["noise_mid_ratio"] = mid / (0.5 * (sh + hl) + 1e-6)
    chroma = rgb - y[..., None]
    chp = chroma - ndimage.gaussian_filter(chroma, (1.5, 1.5, 0))
    f["chroma_noise_ratio"] = float(chp[flat].std() / (std + 1e-6)) if flat.any() else 0.0
    # Spatial correlation of the residual: film grain clumps (>1px), synthetic
    # per-pixel gaussian noise is ~uncorrelated.
    def ac(lag: int) -> float:
        a, b = hp[:, :-lag], hp[:, lag:]
        return float((a * b).mean() / (hp.var() + 1e-12))

    f["noise_autocorr1"], f["noise_autocorr2"] = ac(1), ac(2)
    # JPEG 8x8 blockiness: gradient energy on block boundaries vs elsewhere.
    dx = np.abs(np.diff(y, axis=1))
    if dx.shape[1] > 16:
        on = dx[:, 7::8].mean()
        f["blockiness"] = float(on / (dx.mean() + 1e-6))
    else:
        f["blockiness"] = 1.0
    return f


def compute_features(img: Image.Image, crop: int = 512) -> dict[str, float]:
    f = tone_color_features(img)
    f.update(grain_features(img, crop))
    return f


def feature_vector(f: dict[str, float]) -> np.ndarray:
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
