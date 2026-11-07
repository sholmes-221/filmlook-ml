"""Global, explainable colour-rendition descriptor (deterministic; numpy only).

Describes how an image renders colour and tone, independent of subject and of fine texture:
  per-channel tone percentiles (R, G, B at 1/5/25/50/75/95/99 %)     21   channel response curves, crossovers
  luma percentiles + shoulder/toe/midtone slope                         10   tonal distribution, highlight roll-off, shadow toe
  chroma and colour cast (Cb, Cr) in four luminance bands               12   split toning, cast by tone
  chroma-weighted hue histogram (12 bins)                               12   palette
  saturation percentiles (25/50/75/95 %)                                 4
  channel correlations (RG, GB, RB) and std ratios (R/G, B/G)            5
  luma histogram (16 bins)                                              16   tone curve shape
Total 80 values, each on a documented scale (percentiles and means in [0, 1]; histograms sum to 1; correlations in
[-1, 1]). Computed on an RGB image resized to a 384 px short side (area), in float, so it is resolution-robust.
No texture statistics are included on purpose: this is the aesthetics axis, not the authenticity axis.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PCT = (1, 5, 25, 50, 75, 95, 99)
BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001))
NAMES = ([f"{c}_p{p}" for c in "rgb" for p in PCT] + [f"luma_p{p}" for p in PCT] + ["highlight_shoulder", "shadow_toe", "midtone_slope"]
         + [f"chroma_band{i}" for i in range(4)] + [f"cb_band{i}" for i in range(4)] + [f"cr_band{i}" for i in range(4)]
         + [f"hue_bin{i}" for i in range(12)] + [f"sat_p{p}" for p in (25, 50, 75, 95)]
         + ["corr_rg", "corr_gb", "corr_rb", "std_r_over_g", "std_b_over_g"] + [f"luma_hist{i}" for i in range(16)])
DIM = len(NAMES)


def _prep(im: Image.Image, short: int = 384) -> np.ndarray:
    s = short / min(im.size)
    im = im.convert("RGB").resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def colour_rendition(im: Image.Image) -> np.ndarray:
    x = _prep(im)
    r, g, b = x[..., 0].ravel(), x[..., 1].ravel(), x[..., 2].ravel()
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb, cr = (b - y) * 0.5, (r - y) * 0.5
    out = []
    for c in (r, g, b):
        out += list(np.percentile(c, PCT))
    lp = np.percentile(y, PCT)
    out += list(lp) + [lp[6] - lp[5], lp[1] - lp[0], lp[4] - lp[2]]
    chroma = np.sqrt(cb ** 2 + cr ** 2)
    for lo, hi in BANDS:
        m = (y >= lo) & (y < hi)
        out.append(float(chroma[m].mean()) if m.any() else 0.0)
    for comp in (cb, cr):
        for lo, hi in BANDS:
            m = (y >= lo) & (y < hi)
            out.append(float(comp[m].mean()) if m.any() else 0.0)
    hue = (np.arctan2(cr, cb) + np.pi) / (2 * np.pi)                       # [0, 1)
    w = chroma
    hist = np.histogram(hue, bins=12, range=(0, 1), weights=w)[0]
    out += list(hist / (hist.sum() + 1e-8))
    mx, mn = x.max(2).ravel(), x.min(2).ravel()
    sat = (mx - mn) / (mx + 1e-6)
    out += list(np.percentile(sat, (25, 50, 75, 95)))
    def corr(a, c):
        sa, sc = a.std(), c.std()
        return float(((a - a.mean()) * (c - c.mean())).mean() / (sa * sc)) if sa > 1e-6 and sc > 1e-6 else 0.0
    out += [corr(r, g), corr(g, b), corr(r, b), float(r.std() / (g.std() + 1e-6)), float(b.std() / (g.std() + 1e-6))]
    lh = np.histogram(y, bins=16, range=(0, 1))[0]
    out += list(lh / (lh.sum() + 1e-8))
    v = np.array(out, dtype=np.float32)
    assert v.shape == (DIM,)
    return v
