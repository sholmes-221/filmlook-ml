"""Matching approaches between a photo and the LUT feature database. Each returns a distance per candidate
(lower = more similar) over db.ids (the S styles followed by "none" = untouched digital look).

A  descriptor-to-descriptor: photo descriptor vs the descriptor of a natural pixel cloud rendered through the LUT.
   (A single histogram comparison against one synthetic distribution: kept as the naive reference the brief warns about.)
B  inverse-LUT plausibility: invert the photo through each style and score how plausible the result is as an
   untouched photo (unreachable pixels, residual colour cast, black-point gap). Content-free but weak.
C  reference-image matching: distance to the nearest of R licensed reference scenes rendered through the style
   (multiple scenes, so scene content is matched as well as it can be; the per-scene minimum keeps the best match).
D  weighted feature distance: Fisher-weighted (between-style / within-style variance over the reference scenes)
   distance to each style's centroid over the reference scenes.
E  scene-conditioned: neutralise the photo, find the K reference scenes whose neutralised untouched render is closest
   to the neutralised photo, and compare the photo's style delta (photo - neutralise(photo)) with those scenes' style
   deltas (render - neutralise(render)), Fisher-weighted. Content enters only through the choice of scenes.
CW reference matching with the Fisher weights (C + D's weighting), the combination evaluated in engine_eval.
"""
from __future__ import annotations

import numpy as np

from research.lut.engine.descriptors import describe, neutralise, srgb_to_lab
from research.lut.engine.lut_db import LutDB

APPROACHES = ("A", "B", "C", "CW", "D", "E")


def _z(db: LutDB, x: np.ndarray) -> np.ndarray:
    return ((x - db.mean) / db.std)[..., db.match_idx]


def _w(db: LutDB, weighted: bool) -> np.ndarray:
    if not weighted:
        return np.ones(len(db.match_idx), np.float32)
    w = db.fisher[db.match_idx]
    return np.sqrt(w / max(w.mean(), 1e-6))


def approach_A(db: LutDB, desc: np.ndarray) -> np.ndarray:
    zc = _z(db, db.cloud)
    zp = _z(db, desc)
    return np.linalg.norm(zc - zp, axis=1)


def approach_B(db: LutDB, img_u8: np.ndarray) -> np.ndarray:
    small = img_u8[:: max(1, img_u8.shape[0] // 192), :: max(1, img_u8.shape[1] // 192)]
    n = db.inverse_residual.shape[1]
    idx = np.clip(np.round(small.astype(np.float32) / 255 * (n - 1)).astype(int), 0, n - 1)
    out = np.zeros(db.n_styles + 1, np.float32)
    for i in range(db.n_styles):
        inv, resid = db.inverse(i)
        x = inv.apply(small)
        unreach = float((resid[idx[..., 0], idx[..., 1], idx[..., 2]] > 2.0).mean())
        lab = srgb_to_lab(x.reshape(-1, 3))
        cast = float(np.hypot(lab[:, 1].mean(), lab[:, 2].mean()))
        black_gap = float(np.percentile(x.min(-1), 0.5))
        out[i] = 5.0 * unreach + 0.05 * cast + 2.0 * black_gap
    lab = srgb_to_lab(small.reshape(-1, 3).astype(np.float32) / 255)
    out[db.n_styles] = 0.05 * float(np.hypot(lab[:, 1].mean(), lab[:, 2].mean())) + 2.0 * float(np.percentile(small.min(-1), 0.5) / 255)
    return out


def approach_C(db: LutDB, desc: np.ndarray, weighted: bool = False) -> np.ndarray:
    w = _w(db, weighted)
    zb = _z(db, db.bank) * w                       # (R, S+1, d)
    zp = _z(db, desc) * w
    d = np.linalg.norm(zb - zp, axis=2)            # (R, S+1)
    return d.min(0)


def approach_D(db: LutDB, desc: np.ndarray) -> np.ndarray:
    w = _w(db, True)
    cent = _z(db, db.bank).mean(0) * w             # (S+1, d)
    return np.linalg.norm(cent - _z(db, desc) * w, axis=1)


def approach_E(db: LutDB, img_u8: np.ndarray, desc: np.ndarray, k: int = 6) -> np.ndarray:
    w = _w(db, True)
    desc_n = describe(neutralise(img_u8), with_texture=False)
    zb, zbn = _z(db, db.bank), _z(db, db.bank_neutral)
    zn = _z(db, desc_n)
    scene_d = np.linalg.norm((zbn[:, -1] - zn) * w, axis=1)     # reference scenes whose neutralised look matches the neutralised photo
    near = np.argsort(scene_d)[:k]
    deltas = (zb[near] - zbn[near]).mean(0) * w                 # style delta measured exactly as for the photo: render - neutralise(render)
    photo_delta = (_z(db, desc) - zn) * w
    return np.linalg.norm(deltas - photo_delta, axis=1)


def distances(db: LutDB, img_u8: np.ndarray, desc: np.ndarray | None = None, approaches=APPROACHES) -> dict[str, np.ndarray]:
    if desc is None:
        desc = describe(img_u8)
    out = {}
    for a in approaches:
        if a == "A":
            out[a] = approach_A(db, desc)
        elif a == "B":
            out[a] = approach_B(db, img_u8)
        elif a == "C":
            out[a] = approach_C(db, desc, False)
        elif a == "CW":
            out[a] = approach_C(db, desc, True)
        elif a == "D":
            out[a] = approach_D(db, desc)
        elif a == "E":
            out[a] = approach_E(db, img_u8, desc)
    return out


def similarity_0_100(d: np.ndarray) -> np.ndarray:
    """Relative similarity: best candidate 100, a candidate at the median distance 50, exponential in between.
    This is a ranking aid, not a probability and not a calibrated match quality."""
    d = np.asarray(d, np.float64)
    dmin = d.min()
    scale = max(np.median(d) - dmin, 1e-6)
    return 100.0 * np.exp(-np.log(2.0) * (d - dmin) / scale)
