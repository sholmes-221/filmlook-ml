"""C08a: measure every LUT on standard patches and store named numerical features with provenance.

    python -m research.c08.lutfeat           # -> data/c08/lut_features.json, data/c08/lut_pairs.json

Patches (all sRGB in [0, 1]): a 9^3 RGB grid; a 33-step grey ramp; the 24 ColorChecker patches (published 8-bit sRGB
averages, BabelColor); a hue wheel of 12 hues at L* 50 / C* 30 plus chroma probes at L* 30 and 75. Features describe
tone (black lift, white level, toe, mid-contrast, shoulder), grey-axis tint in shadows / mids / highlights, colour
strength (CIEDE2000 from identity), memory colours (skin, sky, foliage, neutrals), hue rotation and chroma gain per
hue sector, HSV saturation gain, and channel coupling (off-diagonal energy of the best linear fit). Every value is a
measurement of the LUT file; nothing here says what film the LUT's author had in mind.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from research.c08.common import OUT, all_luts, ciede2000, lab, lut_kind
from research.lut.forward_model import lab_to_srgb

CHECKER = np.array([[115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
                    [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46], [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
                    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]]) / 255.0
GROUPS = {"skin": [0, 1], "sky": [2], "foliage": [3, 13], "neutral": [18, 19, 20, 21, 22, 23]}
G = np.linspace(0, 1, 9)
GRID = np.stack(np.meshgrid(G, G, G, indexing="ij"), -1).reshape(-1, 3)
RAMP = np.repeat(np.linspace(0, 1, 33)[:, None], 3, 1)
HUES = np.arange(0, 360, 30)


def _wheel(L: float, C: float) -> np.ndarray:
    h = np.radians(HUES)
    return np.clip(np.asarray(lab_to_srgb(np.stack([np.full_like(h, L), C * np.cos(h), C * np.sin(h)], -1))), 0, 1)


WHEEL50, WHEEL30, WHEEL75 = _wheel(50, 30), _wheel(30, 18), _wheel(75, 22)


def _hsv_sat(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)


def features(lut) -> dict[str, float]:
    f: dict[str, float] = {}
    gi, go = lab(GRID), lab(lut.apply(GRID))
    d = ciede2000(gi, go)
    f["strength_mean_de00"], f["strength_p90_de00"] = float(d.mean()), float(np.percentile(d, 90))
    ri, ro = lab(RAMP), lab(lut.apply(RAMP))
    Lin, Lout = ri[:, 0], ro[:, 0]
    for t in range(0, 101, 10):
        f[f"grey_L{t}"] = float(np.interp(t, Lin, Lout))
    slope = lambda a, b: float((np.interp(b, Lin, Lout) - np.interp(a, Lin, Lout)) / (b - a))  # noqa: E731
    f["tone_black_lift"], f["tone_white_level"] = float(Lout[0]), float(Lout[-1])
    f["tone_toe_slope"], f["tone_mid_contrast"], f["tone_shoulder_slope"] = slope(5, 20), slope(30, 70), slope(80, 95)
    for name, t in (("shadow", 20), ("mid", 50), ("highlight", 80)):
        f[f"tint_{name}_a"] = float(np.interp(t, Lin, ro[:, 1]))
        f[f"tint_{name}_b"] = float(np.interp(t, Lin, ro[:, 2]))
    ci, co = lab(CHECKER), lab(lut.apply(CHECKER))
    dc = ciede2000(ci, co)
    for k in range(24):
        f[f"checker_{k + 1:02d}_de00"] = float(dc[k])
    for g, idx in GROUPS.items():
        delta = (co[idx] - ci[idx]).mean(0)
        f[f"{g}_dL"], f[f"{g}_da"], f[f"{g}_db"] = (float(v) for v in delta)
    wi, wo = lab(WHEEL50), lab(lut.apply(WHEEL50))
    hi, ho = np.degrees(np.arctan2(wi[:, 2], wi[:, 1])), np.degrees(np.arctan2(wo[:, 2], wo[:, 1]))
    dh = (ho - hi + 180) % 360 - 180
    cr = np.hypot(wo[:, 1], wo[:, 2]) / np.maximum(np.hypot(wi[:, 1], wi[:, 2]), 1e-6)
    for k, h in enumerate(HUES):
        f[f"hue{h:03d}_shift_deg"], f[f"hue{h:03d}_chroma_ratio"] = float(dh[k]), float(cr[k])
    for name, w in (("shadow", WHEEL30), ("highlight", WHEEL75)):
        a, b = lab(w), lab(lut.apply(w))
        f[f"chroma_ratio_{name}"] = float((np.hypot(b[:, 1], b[:, 2]) / np.maximum(np.hypot(a[:, 1], a[:, 2]), 1e-6)).mean())
    out = lut.apply(GRID)
    f["hsv_saturation_gain"] = float(_hsv_sat(out).mean() / max(_hsv_sat(GRID).mean(), 1e-6))
    y = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731 - on encoded values, a rough luminance proxy
    f["luma_slope"] = float(np.polyfit(y(GRID), y(out), 1)[0])
    mid = (GRID.min(1) > 0.1) & (GRID.max(1) < 0.9)
    X = np.c_[GRID[mid], np.ones(mid.sum())]
    M, *_ = np.linalg.lstsq(X, out[mid], rcond=None)
    A = M[:3]
    f["coupling_offdiag_ratio"] = float((A - np.diag(np.diag(A))).__pow__(2).sum() / max((A ** 2).sum(), 1e-9))
    f["nonlinearity_rms"] = float(np.sqrt(((X @ M - out[mid]) ** 2).mean()))
    return f


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = all_luts(include_unanalysable=True)
    rows, outs = [], {}
    for it in items:
        row = {k: v for k, v in it.items() if k not in ("lut", "path")}
        row["kind"] = lut_kind(it)
        if it["lut"] is not None:
            lut = it["lut"]
            row["lattice_size_analysed"] = int(lut.size)
            row["interpolation"] = "trilinear on the published lattice (HaldCLUT level 12 = 144^3, cube 33^3/65^3; resampled to 33^3)"
            row["features"] = features(lut)
            outs[row["id"]] = lab(lut.apply(np.r_[GRID, CHECKER]))
        rows.append(row)
    (OUT / "lut_features.json").write_text(json.dumps({"generated": "2026-11-11", "patches": "9^3 grid, 33-step grey ramp, ColorChecker 24 (sRGB 8-bit averages), 12-hue wheels",
                                                         "luts": rows}, indent=1))
    # LUT-to-LUT colour distance on the same patches: how far apart the looks really are.
    ids = list(outs)
    stack = np.stack([outs[i] for i in ids])
    n = len(ids)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = ciede2000(stack[i][None], stack).mean(1)
    meta = {r["id"]: r for r in rows}
    stock = [meta[i].get("film_stock") for i in ids]
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if stock[i] and stock[j] and meta[ids[i]]["kind"] == meta[ids[j]]["kind"] == "film_emulation_colour":
                (within if stock[i] == stock[j] else between).append(D[i, j])
    nearest = []
    for i in range(n):
        j = int(np.argsort(D[i])[1])
        nearest.append({"lut": ids[i], "nearest": ids[j], "de00": round(float(D[i, j]), 2), "same_stock": bool(stock[i] and stock[i] == stock[j])})
    np.save(OUT / "lut_distance.npy", D)
    (OUT / "lut_pairs.json").write_text(json.dumps({"ids": ids, "within_stock_de00": {"n": len(within), "median": float(np.median(within)), "mean": float(np.mean(within))},
                                                     "between_stock_de00": {"n": len(between), "median": float(np.median(between)), "mean": float(np.mean(between))},
                                                     "nearest": nearest}, indent=1))
    kinds = Counter(r["kind"] for r in rows)
    print(f"{len(rows)} LUT records; analysed {len(outs)}; kinds {dict(kinds)}")
    print(f"within-stock dE00 median {np.median(within):.2f} (n={len(within)}), between-stock {np.median(between):.2f} (n={len(between)})")
    print("skipped:", Counter((r.get('skip') or '')[:50] for r in rows if r.get("skip")))


if __name__ == "__main__":
    main()
