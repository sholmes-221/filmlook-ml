"""Non-learned baselines for "which preset was applied?" (Phase 2).

    python -m research.lut.baselines --n 400 --out reports/lut/baselines.json

1. forward matching against an estimated neutral:  neutralise the photo (grey-world white balance in linear light +
   0.5/99.5 percentile levels), render all presets + `none` from that neutral, compare the colour features of each
   render with those of the photo (z-scored by the synthetic training set), pick the closest. Content-matched by
   construction (the same scene under every candidate), but the neutral estimate is itself biased by the preset.
2. inverse-LUT plausibility: invert the photo through each preset's grid inverse (33^3 lattice from the refined
   inverse) and score the result's plausibility as an untouched photo: fraction of pixels the preset cannot have
   produced (forward residual dE > 2), colour cast of the inverted image (|mean a|, |mean b| in CIELAB, grey-world
   prior) and black-point gap (0.5th percentile of min channel). Fixed weights, no learning.
3. (in train.py) kNN / logistic regression on CIELAB statistics -- the learned "statistics" reference.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.lut.evaluate import grouped_bootstrap, topk_hits
from research.lut.features import COLOUR_SLICES, colour_features_from_array
from research.lut.forward_model import apply_lattice, identity_preset, load_presets, srgb_decode, srgb_encode, srgb_to_lab
from research.lut.inverse_analysis import GridInverse

_STATE: dict = {}


def neutralise(img_u8: np.ndarray) -> np.ndarray:
    lin = srgb_decode(img_u8 / 255)
    m = lin.reshape(-1, 3).mean(0)
    gain = m.mean() / np.maximum(m, 1e-6)
    lin = np.clip(lin * gain, 0, 1)
    lo, hi = np.percentile(lin, 0.5), np.percentile(lin, 99.5)
    lin = np.clip((lin - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.clip(srgb_encode(lin) * 255, 0, 255).round().astype(np.uint8)


def inverse_lattice(p, n=33) -> np.ndarray:
    inv = GridInverse(p)
    cube = np.stack(np.meshgrid(*[np.linspace(0, 1, n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    x, resid = inv(cube)
    return x.reshape(n, n, n, 3), resid.reshape(n, n, n)


def _init(presets_path, scaler_path):
    presets = load_presets(presets_path)
    presets["none"] = identity_preset()
    sc = np.load(scaler_path)
    _STATE["names"] = list(presets)
    _STATE["luts"] = {k: p.lattice(33) for k, p in presets.items()}
    _STATE["inv"] = {k: inverse_lattice(p) for k, p in presets.items()}
    _STATE["mean"], _STATE["std"] = sc["mean"], sc["std"]


def interp_scalar(lat: np.ndarray, img_u8: np.ndarray) -> np.ndarray:
    """Nearest-lattice lookup of a scalar field (n, n, n) at the image's colours."""
    n = lat.shape[0]
    idx = np.clip(np.round(img_u8.astype(np.float64) / 255 * (n - 1)).astype(int), 0, n - 1)
    return lat[idx[..., 0], idx[..., 1], idx[..., 2]]


def load_small(path, short=256) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = short / min(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BOX)
    return np.asarray(im)


def score_image(path) -> dict:
    img = load_small(path)
    names, mean, std = _STATE["names"], _STATE["mean"], _STATE["std"]
    sl = COLOUR_SLICES["colour"]
    f_img = (colour_features_from_array(img)[sl] - mean) / std
    # 1. forward matching
    neutral = neutralise(img)
    d_fwd = []
    for k in names:
        r = apply_lattice(_STATE["luts"][k], neutral)
        f = (colour_features_from_array(r)[sl] - mean) / std
        d_fwd.append(float(np.linalg.norm(f - f_img)))
    # 2. inverse plausibility
    s_inv = []
    for k in names:
        lat, resid = _STATE["inv"][k]
        x = apply_lattice(lat, img)
        unreach = float((interp_scalar(resid, img) > 2.0).mean())
        lab = srgb_to_lab(x.reshape(-1, 3) / 255)
        cast = float(np.hypot(lab[:, 1].mean(), lab[:, 2].mean()))
        black_gap = float(np.percentile(x.min(-1), 0.5) / 255)
        s_inv.append(-(5.0 * unreach + 0.05 * cast + 2.0 * black_gap))
    return {"path": path, "forward_dist": d_fwd, "inverse_score": s_inv}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/lut/synth/manifest.jsonl")
    ap.add_argument("--features", default="data/lut/synth/features.npz", help="for the z-scoring statistics (train split)")
    ap.add_argument("--presets", default="data/lut/presets.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="reports/lut/baselines.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest)]
    z = np.load(a.features, allow_pickle=True)
    tr_paths = {r["path"] for r in rows if r["split"] == "train"}
    Xtr = z["colour"][[p in tr_paths for p in z["paths"]]][:, COLOUR_SLICES["colour"]]
    scaler = Path("data/lut/colour_scaler.npz")
    np.savez(scaler, mean=np.nan_to_num(Xtr.mean(0)), std=np.nan_to_num(Xtr.std(0)) + 1e-6)
    test = [r for r in rows if r["split"] == "test"]
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.choice(len(test), min(a.n, len(test)), replace=False)]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.presets, str(scaler))) as ex:
        results = list(ex.map(score_image, [r["path"] for r in test], chunksize=4))
    presets = load_presets(a.presets)
    names = list(presets) + ["none"]
    y = np.array([names.index(r["preset"]) for r in test])
    scenes = np.array([r["scene_id"] for r in test])
    out = {"n": len(test), "classes": names, "chance": 1 / len(names)}
    for key, sign in (("forward_dist", -1), ("inverse_score", 1)):
        S = sign * np.array([r[key] for r in results])
        h1, h3 = topk_hits(S, y, 1), topk_hits(S, y, 3)
        m1, m3 = grouped_bootstrap(h1, scenes), grouped_bootstrap(h3, scenes)
        pred = S.argmax(1)
        out[key] = {"top1": m1[0], "top1_ci": m1[1:], "top3": m3[0], "top3_ci": m3[1:],
                    "pred_hist": {names[i]: int((pred == i).sum()) for i in range(len(names))}}
        print(f"{key:14} top1 {m1[0]:.3f} [{m1[1]:.3f}, {m1[2]:.3f}]  top3 {m3[0]:.3f}  (chance {1 / len(names):.3f}, n={len(test)})")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
