"""Robustness of the preset classifiers to post-hoc perturbations of the test images.

    python -m research.lut.robustness --models colour_mlp,dinov2_colour_mlp --n 300

Perturbations applied to the (already rendered) synthetic test JPEGs, i.e. edits a user might make after the preset:
white-balance shifts (R x1.08 / B x0.92 "warm", the reverse "cool"), exposure -1 / +1 EV (linear light), JPEG
q50 / q80, mild edits (contrast +10 % S-curve, saturation -15 % / +15 %), and a 60 % centre crop. Reported: top-1
accuracy per perturbation against the clean accuracy on the same images (known classes only, scene-bootstrap CI).
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from research.lut.evaluate import grouped_bootstrap
from research.lut.features import Dino, colour_features_from_array
from research.lut.forward_model import srgb_decode, srgb_encode
from research.lut.train import NumpyClassifier, COLOUR_SLICES


def wb(img, gr, gb):
    lin = srgb_decode(img / 255) * np.array([gr, 1.0, gb])
    return np.clip(srgb_encode(np.clip(lin, 0, 1)) * 255, 0, 255).round().astype(np.uint8)


def exposure(img, ev):
    lin = srgb_decode(img / 255) * 2 ** ev
    return np.clip(srgb_encode(np.clip(lin, 0, 1)) * 255, 0, 255).round().astype(np.uint8)


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def contrast(img, k=0.10):
    x = img / 255
    y = np.clip(x + k * (x - 0.5) * (1 - np.abs(2 * x - 1)), 0, 1)   # gentle S-curve, fixed at 0, 0.5 and 1
    return (y * 255).round().astype(np.uint8)


def saturation(img, f):
    x = img / 255
    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    y = np.clip(lum[..., None] + (x - lum[..., None]) * f, 0, 1)
    return (y * 255).round().astype(np.uint8)


def crop(img, frac=0.6):
    h, w = img.shape[:2]
    ch, cw = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


PERTURBATIONS = {"clean": lambda im: im, "wb_warm": lambda im: wb(im, 1.08, 0.92), "wb_cool": lambda im: wb(im, 0.92, 1.08),
                 "exposure_-1ev": lambda im: exposure(im, -1), "exposure_+1ev": lambda im: exposure(im, 1),
                 "jpeg_q50": lambda im: jpeg(im, 50), "jpeg_q80": lambda im: jpeg(im, 80), "contrast_+10": contrast,
                 "saturation_-15": lambda im: saturation(im, 0.85), "saturation_+15": lambda im: saturation(im, 1.15), "crop_60": crop}


def features_for(imgs: list[np.ndarray], need_dino: bool, dino) -> dict:
    col = np.stack([colour_features_from_array(im) for im in imgs]).astype(np.float32)
    out = {"colour": col}
    if need_dino:
        out["dinov2"] = dino([Image.fromarray(im) for im in imgs])
    return out


def select_from(feats: dict, feature_set: str) -> np.ndarray:
    parts = []
    for name in feature_set.split("+"):
        parts.append(feats["dinov2"] if name == "dinov2" else feats["colour"][:, COLOUR_SLICES[name]])
    return np.nan_to_num(np.concatenate(parts, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--manifest", default="data/lut/synth/manifest.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="reports/lut/robustness.json")
    ap.add_argument("--label-field", default="preset")
    a = ap.parse_args()
    clfs = {n: NumpyClassifier.load(f"data/lut/models/{n}.npz") for n in a.models.split(",")}
    need_dino = any("dinov2" in str(c.d["feature_set"]) for c in clfs.values())
    dino = Dino(8) if need_dino else None
    rows = [json.loads(l) for l in open(a.manifest)]
    classes0 = clfs[list(clfs)[0]].classes
    test = [r for r in rows if r["split"] == "test" and str(r[a.label_field]) in classes0]
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.choice(len(test), min(a.n, len(test)), replace=False)]
    imgs = [np.asarray(Image.open(r["path"]).convert("RGB")) for r in test]
    scenes = np.array([r["scene_id"] for r in test])
    out = {"n": len(test), "models": {}}
    for pname, fn in PERTURBATIONS.items():
        pert = [fn(im) for im in imgs]
        feats = features_for(pert, need_dino, dino)
        for mname, clf in clfs.items():
            y = np.array([clf.classes.index(str(r[a.label_field])) for r in test])
            P = clf.predict_proba(select_from(feats, str(clf.d["feature_set"])))
            m = grouped_bootstrap(P.argmax(1) == y, scenes)
            conf = float(P.max(1).mean())
            out["models"].setdefault(mname, {})[pname] = {"top1": m[0], "ci": m[1:], "mean_conf": conf}
            print(f"{mname:22} {pname:16} top1 {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]  mean conf {conf:.2f}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
