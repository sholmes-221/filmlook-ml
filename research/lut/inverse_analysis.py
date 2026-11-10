"""Numerical analysis of the forward model: monotonicity, Jacobian conditioning, grid inverse, clipping, ambiguity.

    python -m research.lut.inverse_analysis --out reports/lut/inverse_analysis.json

Sections written to the JSON (and printed as tables):
  curves     per preset: minimum slope of the per-channel curve table, flat fraction (slope < 0.02), black lift y(0),
             white point y(1): output values below y(0) are unreachable -> any photo with such pixels was not
             produced by this preset from a non-clipped input (a plausibility cue, also a fragility: any later edit
             re-introduces blacks).
  jacobian   finite-difference Jacobian of the complete map on a 17^3 grid: fraction of negative determinants
             (orientation flips = non-injective neighbourhoods), fraction |det| < 0.05 (near collapse), median
             condition number, fraction of grid points whose output hits 0 or 1 in a channel (clipped).
  inverse    grid inverse (33^3 lattice + KD-tree nearest neighbour + 6 Gauss-Newton steps) on 20,000 natural pixels:
             round-trip CIELAB dE76 median / p95 / fraction above 2.
  reach      fraction of the sRGB cube (lattice) within dE 1 of some forward output (coverage of the output gamut).
  ambiguity  for natural pixels rendered with preset A, the fraction that also has a pre-image under preset B
             within dE 1 (pixel-level ambiguity), and the mean dE between A(x) and B(x) on natural pixels before and
             after an optimal per-channel gain (3-parameter white balance + exposure) -- how much of the
             inter-preset difference an unknown WB/exposure edit can absorb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from research.lut.forward_model import Preset, identity_preset, load_presets, srgb_to_lab


def natural_pixels(n_images: int = 120, per_image: int = 200, seed: int = 0,
                   manifest="data/manifests/raw_commoncatalog.jsonl", cache="data/lut/natural_pixels.npy") -> np.ndarray:
    p = Path(cache)
    if p.exists():
        return np.load(p)
    from PIL import Image
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r.get("medium") in ("digital_camera", "smartphone")]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out = []
    for r in rows[:n_images]:
        im = Image.open(r["path"]).convert("RGB")
        im.thumbnail((512, 512))
        a = np.asarray(im).reshape(-1, 3)
        out.append(a[rng.choice(len(a), per_image, replace=False)])
    arr = np.concatenate(out).astype(np.float64) / 255
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return arr


def de76(a, b):
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)


def curve_report(p: Preset) -> dict:
    lut = p.curve_lut(4096)
    slope = np.diff(lut, axis=0) * 4095
    return {"min_slope": float(slope.min()), "flat_fraction": float((slope < 0.02).mean()),
            "black_lift_8bit": [float(round(v * 255, 1)) for v in lut[0]],
            "white_8bit": [float(round(v * 255, 1)) for v in lut[-1]],
            "monotone": bool((slope >= -1e-9).all())}


def jacobian(p: Preset, x: np.ndarray, h: float = 1 / 512) -> np.ndarray:
    """(N, 3, 3) central-difference Jacobian d out / d in, inputs kept inside [h, 1-h]."""
    x = np.clip(x, h, 1 - h)
    J = np.zeros(x.shape + (3,))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[..., :, j] = (p.apply(x + e) - p.apply(x - e)) / (2 * h)
    return J


def jacobian_report(p: Preset, n: int = 17) -> dict:
    g = np.linspace(0.02, 0.98, n)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    y = p.apply(grid)
    J = jacobian(p, grid)
    det = np.linalg.det(J)
    sv = np.linalg.svd(J, compute_uv=False)
    cond = sv[:, 0] / np.maximum(sv[:, -1], 1e-12)
    clipped = ((y <= 0) | (y >= 1)).any(-1)
    return {"neg_det_fraction": float((det < 0).mean()), "near_singular_fraction": float((np.abs(det) < 0.05).mean()),
            "median_abs_det": float(np.median(np.abs(det))), "median_cond": float(np.median(cond)),
            "p95_cond": float(np.percentile(cond, 95)), "clipped_fraction": float(clipped.mean())}


class GridInverse:
    """Nearest-neighbour on a forward lattice followed by Gauss-Newton refinement."""

    def __init__(self, p: Preset, n: int = 33):
        self.p = p
        self.lat_in = np.stack(np.meshgrid(*[np.linspace(0, 1, n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.lat_out = p.apply(self.lat_in)
        self.tree = cKDTree(self.lat_out)

    def __call__(self, y: np.ndarray, steps: int = 6) -> tuple[np.ndarray, np.ndarray]:
        d, idx = self.tree.query(y)
        x = self.lat_in[idx].copy()
        for _ in range(steps):
            r = self.p.apply(x) - y
            J = jacobian(self.p, x)
            # damped Gauss-Newton: solve (J^T J + lam I) dx = -J^T r
            JT = np.swapaxes(J, 1, 2)
            A = JT @ J + 1e-4 * np.eye(3)
            b = -(JT @ r[..., None])
            dx = np.linalg.solve(A, b)[..., 0]
            x = np.clip(x + dx, 0, 1)
        resid = de76(self.p.apply(x), y)
        return x, resid


def inverse_report(p: Preset, nat: np.ndarray) -> dict:
    inv = GridInverse(p)
    y = p.apply(nat)
    x_hat, resid = inv(y)
    rt = de76(x_hat, nat)
    return {"roundtrip_de_median": float(np.median(rt)), "roundtrip_de_p95": float(np.percentile(rt, 95)),
            "roundtrip_de_gt2_fraction": float((rt > 2).mean()), "forward_residual_de_median": float(np.median(resid)),
            "forward_residual_de_p95": float(np.percentile(resid, 95))}


def reach_report(p: Preset, n_lattice: int = 33) -> dict:
    lat_out = p.apply(np.stack(np.meshgrid(*[np.linspace(0, 1, 65)] * 3, indexing="ij"), -1).reshape(-1, 3))
    tree = cKDTree(srgb_to_lab(lat_out))
    cube = np.stack(np.meshgrid(*[np.linspace(0, 1, n_lattice)] * 3, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(srgb_to_lab(cube))
    dark = cube.max(-1) < 0.1
    return {"cube_reachable_de1": float((d < 1.0).mean()), "cube_reachable_de3": float((d < 3.0).mean()),
            "dark_cube_reachable_de3": float((d[dark] < 3.0).mean())}


def best_gain_de(a: np.ndarray, b: np.ndarray) -> float:
    """Mean dE between a and g*b for the least-squares per-channel gain g (in linear-ish sRGB values)."""
    g = (a * b).sum(0) / np.maximum((b * b).sum(0), 1e-9)
    return float(de76(a, np.clip(b * g, 0, 1)).mean())


def ambiguity_report(presets: dict[str, Preset], nat: np.ndarray, n_sub: int = 3000) -> dict:
    """Pixel-level ambiguity across presets, measured with the refined grid inverse (no lattice-resolution floor)."""
    names = list(presets)
    sub = nat[:n_sub]
    outs = {k: presets[k].apply(sub) for k in names}
    invs = {k: GridInverse(presets[k]) for k in names}
    pre_image = np.zeros((len(names), len(names)))
    de_raw = np.zeros((len(names), len(names)))
    de_gain = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            _, resid = invs[b](outs[a])
            pre_image[i, j] = (resid < 1.0).mean()
            de_raw[i, j] = de76(outs[a], outs[b]).mean()
            de_gain[i, j] = best_gain_de(outs[a], outs[b])
    return {"names": names, "pixel_preimage_fraction_de1": pre_image.round(3).tolist(),
            "mean_de_between_presets": de_raw.round(2).tolist(), "mean_de_after_best_gain": de_gain.round(2).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", default="data/lut/presets.json")
    ap.add_argument("--out", default="reports/lut/inverse_analysis.json")
    a = ap.parse_args()
    presets = load_presets(a.presets)
    presets["none"] = identity_preset()
    nat = natural_pixels()
    rep = {"n_natural_pixels": int(len(nat)), "curves": {}, "jacobian": {}, "inverse": {}, "reach": {}}
    print(f"{'preset':22} {'minslope':>8} {'flat%':>6} {'black(8b)':>16} {'negdet%':>8} {'sing%':>6} {'cond50':>7} {'clip%':>6} "
          f"{'rtDE50':>7} {'rtDE95':>7} {'rt>2%':>6} {'reach1':>7} {'dark3':>6}")
    for name, p in presets.items():
        c = curve_report(p)
        j = jacobian_report(p)
        inv = inverse_report(p, nat)
        r = reach_report(p)
        rep["curves"][name], rep["jacobian"][name], rep["inverse"][name], rep["reach"][name] = c, j, inv, r
        print(f"{name:22} {c['min_slope']:8.3f} {100 * c['flat_fraction']:6.1f} {str([int(v) for v in c['black_lift_8bit']]):>16} "
              f"{100 * j['neg_det_fraction']:8.2f} {100 * j['near_singular_fraction']:6.1f} {j['median_cond']:7.2f} {100 * j['clipped_fraction']:6.1f} "
              f"{inv['roundtrip_de_median']:7.2f} {inv['roundtrip_de_p95']:7.2f} {100 * inv['roundtrip_de_gt2_fraction']:6.1f} "
              f"{r['cube_reachable_de1']:7.3f} {r['dark_cube_reachable_de3']:6.3f}")
    amb = ambiguity_report(presets, nat)
    rep["ambiguity"] = amb
    names = amb["names"]
    M = np.array(amb["mean_de_between_presets"])
    G = np.array(amb["mean_de_after_best_gain"])
    P = np.array(amb["pixel_preimage_fraction_de1"])
    print("\nmean dE between presets on natural pixels (upper: raw, lower: after best per-channel gain)")
    print(" " * 14 + " ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        row = [f"{M[i, j]:6.1f}" if j > i else (f"{G[i, j]:6.1f}" if j < i else "     -") for j in range(len(names))]
        print(f"{n[:13]:13} " + " ".join(row))
    off = ~np.eye(len(names), dtype=bool)
    print("\npixel pre-image fraction (row: rendered with A; col: has a pre-image under B within dE 1)")
    for i, n in enumerate(names):
        print(f"{n[:13]:13} " + " ".join(f"{P[i, j]:6.2f}" for j in range(len(names))))
    rep["ambiguity"]["summary"] = {"pixel_preimage_diag_mean": float(np.diag(P).mean()), "pixel_preimage_offdiag_mean": float(P[off].mean()), "pixel_preimage_offdiag_min": float(P[off].min()),
                                   "mean_de_offdiag_mean": float(M[off].mean()), "mean_de_offdiag_min": float(M[off].min()),
                                   "mean_de_after_gain_offdiag_mean": float(G[off].mean()), "mean_de_after_gain_offdiag_min": float(G[off].min()),
                                   "closest_pair_after_gain": [names[k] for k in np.unravel_index(np.argmin(np.where(off, G, 1e9)), G.shape)]}
    print("\nsummary:", json.dumps(rep["ambiguity"]["summary"], indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
