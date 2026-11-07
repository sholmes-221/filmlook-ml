"""Style features v1: Python/OpenCV REFERENCE implementation, mirrored exactly by js/filmlook-style/features.js.

Input contract (both runtimes): an RGB(A) uint8 image whose long side is <= WORK_SIDE (the app draws the user's
photo on a Canvas at that size and calls getImageData; this reference uses cv2.resize INTER_AREA). Every feature
below is defined so the two implementations agree to ~1e-9: double precision throughout, percentiles via 256-bin
histograms, explicit floor(x + 0.5) rounding, box means via integral images with clipped windows, interior-only
finite differences.

Groups: tone, colour, hue distribution, texture/sharpness, local contrast, spatial / composition.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_VERSION = "style-features-v1"
WORK_SIDE = 384
EPS = 1e-6

FEATURE_NAMES = [
    # tone
    "y_mean", "y_std", "p05", "p25", "p50", "p75", "p95", "clip_hi", "clip_lo", "dark_frac", "bright_frac",
    # colour
    "sat_mean", "sat_std", "sat_p90", "sat_p95", "mono", "accent_frac", "muted_frac", "warmth", "tint", "colorfulness",
    # hue distribution (saturation-weighted, 6 sectors) + entropy
    "hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta", "hue_entropy",
    # texture / sharpness
    "grad_mean", "edge_frac", "lap_logvar", "noise", "hv_balance", "diag_frac",
    # local contrast
    "local_contrast", "lc_ratio",
    # spatial / composition
    "grid_y_std", "sat_concentration", "dark_cells", "accent_cells", "centroid_x", "centroid_y",
    "dark_centroid_x", "dark_centroid_y", "center_ratio", "top_bottom", "sharp_center_ratio", "edge_center_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


def target_size(w: int, h: int, side: int = WORK_SIDE) -> tuple[int, int]:
    s = min(1.0, side / max(w, h))
    return max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))


def box2x(rgb: np.ndarray) -> np.ndarray:
    """Exact 2x2 box average with integer rounding: floor((a+b+c+d+2)/4). Bit-identical to features.js box2x."""
    h, w = rgb.shape[0] // 2, rgb.shape[1] // 2
    a = rgb[: 2 * h, : 2 * w].astype(np.int32)
    s = a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]
    return ((s + 2) // 4).astype(np.uint8)


def resize_for_style(rgb: np.ndarray, side: int = WORK_SIDE, resample=None) -> np.ndarray:
    """Two-stage downscale used by BOTH runtimes: any resampler to 2x the target size (JS: Canvas drawImage;
    reference: OpenCV INTER_AREA), then our own exact 2x2 box average. The final averaging step removes most of
    the resampler-dependent aliasing/blur (see reports/style_stability.json)."""
    h, w = rgb.shape[:2]
    if max(h, w) <= side:
        return rgb
    cw, ch = target_size(w, h, side)
    mid = (resample or (lambda a, size: cv2.resize(a, size, interpolation=cv2.INTER_AREA)))(rgb, (2 * cw, 2 * ch))
    return box2x(mid)


def _hist_pct(h: np.ndarray, n: int, q: float) -> float:
    c = np.cumsum(h)
    return float(np.searchsorted(c, q * n, side="left")) / 255.0


def _q8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x * 255.0 + 0.5), 0, 255).astype(np.int64)


def _box_mean(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over the (2r+1)^2 window clipped to the image, via an OpenCV integral image."""
    h, w = img.shape
    ii = cv2.integral(img.astype(np.float64), sdepth=cv2.CV_64F)  # (h+1, w+1)
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    cnt = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / cnt


def style_features(rgba: np.ndarray) -> dict[str, float]:
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3/4 uint8")
    rgb = a[..., :3].astype(np.float64) / 255.0
    H, W = rgb.shape[:2]
    if H < 16 or W < 16:
        raise ValueError("image too small")
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    n = H * W
    f: dict[str, float] = {}
    # ---------------- tone
    y8 = _q8(Y)
    hy = np.bincount(y8.ravel(), minlength=256)
    f["y_mean"] = float(Y.mean())
    f["y_std"] = float(Y.std())
    for k, q in (("p05", 0.05), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p95", 0.95)):
        f[k] = _hist_pct(hy, n, q)
    f["clip_hi"] = float((y8 >= 250).mean())
    f["clip_lo"] = float((y8 <= 5).mean())
    f["dark_frac"] = float((y8 < 51).mean())
    f["bright_frac"] = float((y8 > 204).mean())
    # ---------------- colour
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    sat = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    hs = np.bincount(_q8(sat).ravel(), minlength=256)
    f["sat_mean"], f["sat_std"] = float(sat.mean()), float(sat.std())
    f["sat_p90"], f["sat_p95"] = _hist_pct(hs, n, 0.90), _hist_pct(hs, n, 0.95)
    f["mono"] = 1.0 if f["sat_p95"] < 0.08 else 0.0
    accent = (sat > 0.55) & (Y > 0.15) & (Y < 0.9)
    f["accent_frac"] = float(accent.mean())
    f["muted_frac"] = float((sat < 0.25).mean())
    f["warmth"] = float((R - B).mean())
    f["tint"] = float((G - 0.5 * (R + B)).mean())
    rg, yb = R - G, 0.5 * (R + G) - B
    f["colorfulness"] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # ---------------- hue sectors (weighted by saturation), only reasonably saturated, not-black pixels
    m = (sat > 0.15) & (mx > 0.1) & (d > 0)
    dd = np.where(d > 0, d, 1.0)
    h_r = (G - B) / dd
    h_r = np.where(h_r < 0, h_r + 6.0, h_r)
    hue = np.where(mx == R, h_r, np.where(mx == G, (B - R) / dd + 2.0, (R - G) / dd + 4.0))
    sector = np.clip(np.floor(hue).astype(np.int64), 0, 5)
    wsum = np.bincount(sector[m], weights=sat[m], minlength=6)
    tot = wsum.sum()
    p = wsum / tot if tot > 0 else np.zeros(6)
    for k, name in enumerate(["hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue", "hue_magenta"]):
        f[name] = float(p[k])
    f["hue_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(6)) if tot > 0 else 0.0
    # ---------------- texture (interior pixels)
    c = Y[1:-1, 1:-1]
    gx = (Y[1:-1, 2:] - Y[1:-1, :-2]) / 2.0
    gy = (Y[2:, 1:-1] - Y[:-2, 1:-1]) / 2.0
    g = np.sqrt(gx * gx + gy * gy)
    edges = g > 0.08
    f["grad_mean"] = float(g.mean())
    f["edge_frac"] = float(edges.mean())
    lap = Y[:-2, 1:-1] + Y[2:, 1:-1] + Y[1:-1, :-2] + Y[1:-1, 2:] - 4.0 * c
    f["lap_logvar"] = float(np.log(lap.var() + 1e-8))
    ax, ay = np.abs(gx[edges]), np.abs(gy[edges])
    sx, sy = ax.sum(), ay.sum()
    f["hv_balance"] = float((sx - sy) / (sx + sy + EPS))
    f["diag_frac"] = float((np.minimum(ax, ay) > 0.5 * np.maximum(ax, ay)).mean()) if edges.any() else 0.0
    b3 = _box_mean(Y, 1)[1:-1, 1:-1]
    res = c - b3
    flat = g < 0.02
    f["noise"] = float(res[flat].std()) if flat.sum() >= 100 else float(res.std())
    # ---------------- local contrast
    b8 = _box_mean(Y, 8)
    f["local_contrast"] = float(np.abs(Y - b8).mean())
    f["lc_ratio"] = float(f["local_contrast"] / (f["y_std"] + EPS))
    # ---------------- spatial / composition (8x8 grid)
    cy = (np.arange(H) * 8) // H
    cx = (np.arange(W) * 8) // W
    cell = (cy[:, None] * 8 + cx[None, :]).ravel()
    cnt = np.bincount(cell, minlength=64)
    gy_mean = np.bincount(cell, weights=Y.ravel(), minlength=64) / cnt
    gs_sum = np.bincount(cell, weights=sat.ravel(), minlength=64)
    gs_mean = gs_sum / cnt
    f["grid_y_std"] = float(gy_mean.std())
    top8 = np.sort(gs_sum)[-8:].sum()
    f["sat_concentration"] = float(top8 / (gs_sum.sum() + EPS))
    f["dark_cells"] = float((gy_mean < 0.2).mean())
    f["accent_cells"] = float((gs_mean > 0.5).mean())
    xs = (np.arange(W) + 0.5) / W - 0.5
    ys = (np.arange(H) + 0.5) / H - 0.5
    sy_ = Y.sum()
    f["centroid_x"] = float((Y * xs[None, :]).sum() / (sy_ + EPS))
    f["centroid_y"] = float((Y * ys[:, None]).sum() / (sy_ + EPS))
    D = 1.0 - Y
    sd_ = D.sum()
    f["dark_centroid_x"] = float((D * xs[None, :]).sum() / (sd_ + EPS))
    f["dark_centroid_y"] = float((D * ys[:, None]).sum() / (sd_ + EPS))
    inner = np.zeros((H, W), bool)
    inner[H // 4: (3 * H) // 4, W // 4: (3 * W) // 4] = True
    f["center_ratio"] = float(Y[inner].mean() / (Y[~inner].mean() + 0.02))
    t3, b3_ = H // 3, H - H // 3
    f["top_bottom"] = float(Y[:t3].mean() - Y[b3_:].mean())
    inner_c = inner[1:-1, 1:-1]
    lv_in, lv_out = lap[inner_c].var(), lap[~inner_c].var()
    f["sharp_center_ratio"] = float(np.log((lv_in + 1e-8) / (lv_out + 1e-8)))
    f["edge_center_ratio"] = float((edges[inner_c].mean() + 0.01) / (edges[~inner_c].mean() + 0.01))
    return f


def feature_vector(f: dict[str, float]) -> np.ndarray:
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)


def features_from_path(path, side: int = WORK_SIDE) -> dict[str, float]:
    from filmlook_infer.imageio import load_image
    im = load_image(path, max_side=max(side * 2, 768))
    return style_features(resize_for_style(np.asarray(im), side))
