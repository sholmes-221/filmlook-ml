"""Deterministic film-likeness estimate from measured statistics (NumPy / OpenCV; weights fitted offline).

    from research.lut.engine.film_likeness import FilmLikeness, film_features
    fl = FilmLikeness("research/lut/db/film_likeness.json")
    fl.estimate(img_u8) -> {"estimate": 0-100, "band": ..., "evidence": [(statistic, value, direction)], "confidence": ...,
                            "contributions": {...}, "note": ...}

Features (`FEATURE_NAMES`): a subset of the engine descriptors on the 384 px copy (black / white level, highlight
roll-off, shadow compression, saturation and chroma distribution, colourfulness, shadow / highlight colour cast,
channel correlations and std ratios, contrast, dynamic range, midtone share, clipped-black and clipped-white shares,
local contrast, high-frequency energy, noise / grain, edge density) plus finer-scale texture statistics measured on a
1024 px short-side copy: noise sigma in flat regions, its horizontal / vertical lag-1 autocorrelation (grain is
spatially correlated after scanning; sensor noise after demosaicing is too, differently), the chroma / luminance noise
ratio, a halation proxy (red excess in a ring around clipped highlights), clipped-highlight and clipped-black shares,
and a monochrome flag. The estimator is a logistic regression on standardised features with a Platt-style
recalibration; every term is inspectable (`contributions`).
What this is not: evidence about the film stock, the camera or the process. It is the probability, under the
training distribution (Flickr film scans vs Flickr digital captures), that an image with these statistics is a scan.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.lut.engine.descriptors import NAMES, describe, srgb_to_lab, standardise

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_BASE = ["black_level", "white_level", "highlight_rolloff", "shadow_compression", "sat_p50", "sat_p95", "chroma_p50", "chroma_p95",
         "colourfulness", "shadow_a", "shadow_b", "high_a", "high_b", "corr_rg", "corr_gb", "corr_rb", "std_r_over_g", "std_b_over_g",
         "contrast_L_std", "dynamic_range", "midtone_share", "L_hist0", "L_hist15", "local_contrast", "hf_energy_ratio", "noise_grain",
         "edge_density", "laplacian_var"]
_FINE = ["fine_noise_sigma", "fine_noise_acf_h", "fine_noise_acf_v", "fine_noise_chroma_ratio", "halation_proxy", "clip_high_share",
         "clip_low_share", "monochrome_flag"]
FEATURE_NAMES = _BASE + _FINE
_IDX = [NAMES.index(n) for n in _BASE]


def fine_texture(img_u8: np.ndarray) -> np.ndarray:
    im = standardise(img_u8, short=1024)
    rgb = im.astype(np.float32) / 255
    g = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    if cv2 is None:  # pragma: no cover
        return np.zeros(len(_FINE), np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    hp = g - blur
    grad = cv2.magnitude(cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1))
    thr = np.percentile(grad, 25)
    flat = grad < thr
    if flat.sum() < 500:
        flat = np.ones_like(flat)
    r = hp[flat]
    sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))
    # lag-1 autocorrelation of the residual on flat pixels (pairs where both pixels are flat)
    def acf(shift_axis):
        a = hp
        b = np.roll(hp, 1, axis=shift_axis)
        m = flat & np.roll(flat, 1, axis=shift_axis)
        x, y = a[m], b[m]
        if len(x) < 100:
            return 0.0
        x = x - x.mean()
        y = y - y.mean()
        return float((x * y).mean() / max(x.std() * y.std(), 1e-9))
    acf_h, acf_v = acf(1), acf(0)
    lab = srgb_to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    chroma_hp = np.hypot(lab[..., 1] - cv2.GaussianBlur(lab[..., 1], (0, 0), 1.2), lab[..., 2] - cv2.GaussianBlur(lab[..., 2], (0, 0), 1.2))
    L_hp = np.abs(lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 1.2))
    chroma_ratio = float(np.median(chroma_hp[flat]) / max(np.median(L_hp[flat]), 1e-6))
    # halation proxy: red excess (R-G) in a 3..9 px ring around clipped highlights vs elsewhere
    clipped = (im.max(-1) >= 250).astype(np.uint8)
    clip_share = float(clipped.mean())
    hal = 0.0
    if 1e-4 < clip_share < 0.3:
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        ring = (cv2.dilate(clipped, k_out) > 0) & ~(cv2.dilate(clipped, k_in) > 0)
        if ring.sum() > 200:
            rg = rgb[..., 0] - rgb[..., 1]
            hal = float(rg[ring].mean() - rg[~ring & (clipped == 0)].mean())
    clip_low = float((im.min(-1) <= 2).mean())
    c = np.hypot(lab[..., 1], lab[..., 2])
    lit = lab[..., 0] > 15                      # judge monochrome on pixels that are not near black
    mono = float(np.percentile(c[lit] if lit.sum() > 100 else c, 50) < 3.0)
    return np.array([sigma, acf_h, acf_v, chroma_ratio, hal, clip_share, clip_low, mono], np.float32)


def film_features(img_u8: np.ndarray, desc: np.ndarray | None = None) -> np.ndarray:
    if desc is None:
        desc = describe(img_u8)
    return np.concatenate([desc[_IDX], fine_texture(img_u8)]).astype(np.float32)


class FilmLikeness:
    def __init__(self, path):
        self.d = json.loads(Path(path).read_text())
        assert self.d["feature_names"] == FEATURE_NAMES, "feature set changed; refit film_likeness.json"
        self.mean = np.array(self.d["mean"], np.float32)
        self.std = np.array(self.d["std"], np.float32)
        self.w = np.array(self.d["weights"], np.float32)
        self.b = float(self.d["bias"])
        self.cal_a, self.cal_c = float(self.d["calibration"]["a"]), float(self.d["calibration"]["c"])

    def logit_and_contributions(self, f: np.ndarray):
        z = (f - self.mean) / self.std
        contrib = z * self.w
        return float(contrib.sum() + self.b), contrib

    def probability(self, f: np.ndarray) -> float:
        logit, _ = self.logit_and_contributions(f)
        return float(1 / (1 + np.exp(-(self.cal_a * logit + self.cal_c))))

    def estimate(self, img_u8: np.ndarray, desc: np.ndarray | None = None) -> dict:
        f = film_features(img_u8, desc)
        logit, contrib = self.logit_and_contributions(f)
        p = float(1 / (1 + np.exp(-(self.cal_a * logit + self.cal_c))))
        order = np.argsort(-np.abs(contrib))[:6]
        evidence = [(FEATURE_NAMES[i], round(float(f[i]), 4), "towards film" if contrib[i] > 0 else "towards digital") for i in order]
        band = "film-like" if p >= self.d["bands"]["film"] else ("ambiguous" if p >= self.d["bands"]["ambiguous"] else "digital-like")
        mono = f[FEATURE_NAMES.index("monochrome_flag")] > 0.5
        strength = abs(p - 0.5)
        confidence = "low" if (mono or strength < 0.15) else ("medium" if strength < 0.3 else "high")
        return {"estimate": round(100 * p, 1), "band": band, "confidence": confidence, "evidence": evidence,
                "contributions": {FEATURE_NAMES[i]: round(float(contrib[i]), 4) for i in range(len(f))}, "logit": round(logit, 4),
                "note": self.d.get("note", "measured estimate of scan-likeness; not a stock, camera or process identification"),
                "validation": self.d.get("validation_summary")}
