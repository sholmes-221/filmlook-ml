"""Feature extractors for preset / film-style recognition.

Colour features (numpy, CPU, deterministic; computed on a 384 px short-side copy):
  stats24      the colour / tone / grain statistics of film_physics/evaluation/family_eval.py (same formulas; that file
               calls them "26" but the vector has 24 entries), so results are comparable with the 2026-11-09 evaluation
  cr80         filmlook_infer.colour_rendition (80 values: per-channel percentiles, luma shape, chroma/cast by band,
               hue histogram, saturation, channel correlations, luma histogram)
  labhist      CIELAB histograms: 16-bin L, 12x12 (a, b) chroma-weighted histogram, 12-bin hue x 3 lightness bands
               (36), and 12 per-channel RGB percentile deltas (R-G, B-G at 6 levels) = 16 + 144 + 36 + 12 = 208
  colour       concat of the three (312)
DINOv2-small (ONNX, /home/shanmo/misc/film-evaluation/models/photo_gate/dinov2_small_cls_patch.onnx, 768 = CLS + mean
patch token, ImageNet normalisation via filmlook_infer.compact.preprocess).

    python -m research.lut.features --manifest data/lut/synth/manifest.jsonl --out data/lut/synth/features.npz --dinov2
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from filmlook_infer import colour_rendition
from filmlook_infer.compact import preprocess
from research.lut.forward_model import srgb_to_lab

DINO_ONNX = "/home/shanmo/misc/film-evaluation/models/photo_gate/dinov2_small_cls_patch.onnx"
STATS24_NAMES = ["L_mean", "L_std", "a_mean", "b_mean", "a_std", "b_std", "chroma_mean", "chroma_p90"] + [f"hue_sector{i}" for i in range(6)] + \
    ["a_shadow", "b_shadow", "a_highlight", "b_highlight", "L_p5", "L_p50", "L_p95", "L_range", "grain_proxy", "is_grey"]


def load_rgb(path, short: int = 384) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (short * 2, short * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = short / min(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BOX)
    return np.asarray(im, dtype=np.float64) / 255


def stats24(rgb: np.ndarray) -> np.ndarray:
    """Replica of film_physics/evaluation/family_eval.py::stats (thumbnail 256, D65 sRGB->Lab as there)."""
    im = Image.fromarray((rgb * 255).round().astype(np.uint8))
    im.thumbnail((256, 256))
    x = np.asarray(im, dtype=np.float64) / 255
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    l, a, b = 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    sectors = [float(((hue >= s) & (hue < s + 60) & (chroma > 8)).mean()) for s in range(0, 360, 60)]
    sh, hi = l < 30, l > 70
    g = l / 100
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((64, 64)).resize(g.shape[::-1], Image.BILINEAR), dtype=np.float64) / 255
    return np.array([l.mean(), l.std(), a.mean(), b.mean(), a.std(), b.std(), chroma.mean(), np.percentile(chroma, 90), *sectors,
                     a[sh].mean() if sh.any() else 0, b[sh].mean() if sh.any() else 0, a[hi].mean() if hi.any() else 0, b[hi].mean() if hi.any() else 0,
                     *np.percentile(l, [5, 50, 95]), np.percentile(l, 95) - np.percentile(l, 5), float((g - small).std()), float(chroma.mean() < 4)])


def labhist(rgb: np.ndarray) -> np.ndarray:
    lab = srgb_to_lab(rgb.reshape(-1, 3))
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    c = np.hypot(a, b)
    hL, _ = np.histogram(L, bins=16, range=(0, 100))
    hab, _, _ = np.histogram2d(a, b, bins=12, range=[[-60, 60], [-60, 60]], weights=c)
    hue = np.degrees(np.arctan2(b, a)) % 360
    band = np.digitize(L, [35, 65])
    hh = np.zeros((3, 12))
    for k in range(3):
        m = (band == k) & (c > 5)
        if m.any():
            hh[k], _ = np.histogram(hue[m], bins=12, range=(0, 360), weights=c[m])
    pct = np.percentile(rgb.reshape(-1, 3), [2, 10, 30, 50, 70, 90, 98], axis=0)
    deltas = np.concatenate([(pct[:, 0] - pct[:, 1])[:6], (pct[:, 2] - pct[:, 1])[:6]])
    out = np.concatenate([hL / max(hL.sum(), 1), hab.ravel() / max(hab.sum(), 1e-9), hh.ravel() / max(hh.sum(), 1e-9), deltas])
    return out.astype(np.float64)


def colour_features(path) -> np.ndarray:
    rgb = load_rgb(path)
    im = Image.fromarray((rgb * 255).round().astype(np.uint8))
    cr = colour_rendition.colour_rendition(im)
    return np.concatenate([stats24(rgb), np.asarray(cr, dtype=np.float64), labhist(rgb)])


COLOUR_SLICES = {"stats24": slice(0, 24), "cr80": slice(24, 104), "labhist": slice(104, 312), "colour": slice(0, 312)}


def colour_features_from_array(rgb_u8: np.ndarray) -> np.ndarray:
    im = Image.fromarray(rgb_u8)
    s = 384 / min(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BOX)
    rgb = np.asarray(im, dtype=np.float64) / 255
    cr = colour_rendition.colour_rendition(im)
    return np.concatenate([stats24(rgb), np.asarray(cr, dtype=np.float64), labhist(rgb)])


class Dino:
    def __init__(self, threads: int = 8, path: str = DINO_ONNX):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, srcs, batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(srcs), batch):
            x = np.concatenate([preprocess(s) for s in srcs[i:i + batch]])
            out.append(self.sess.run(None, {"pixel_values": x})[0])
        return np.concatenate(out).astype(np.float32)


def extract(paths: list[str], dinov2: bool = False, workers: int = 8, threads: int = 8) -> dict:
    with ProcessPoolExecutor(workers) as ex:
        col = np.stack(list(ex.map(colour_features, paths, chunksize=16)))
    res = {"colour": col.astype(np.float32)}
    if dinov2:
        res["dinov2"] = Dino(threads)(paths)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dinov2", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest)]
    paths = [r["path"] for r in rows]
    feats = extract(paths, a.dinov2, a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, paths=np.array(paths), **feats)
    print("wrote", a.out, {k: v.shape for k, v in feats.items()})


if __name__ == "__main__":
    main()
