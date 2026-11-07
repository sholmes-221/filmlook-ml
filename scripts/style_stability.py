#!/usr/bin/env python
"""Label-free checks for the style engine on held-out photos (val + test CommonCatalog originals):

* stability under perturbations a phone / Mini Tool pipeline can introduce:
  resampler (browser drawImage differs: area vs bilinear vs bicubic), JPEG re-encode (q75), 4% centre crop,
  +8% exposure, horizontal flip;
* reliable-match rate (how often the engine commits to a match) per medium / subject.
Writes reports/style_stability.json.
"""
import io
import json
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image

from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.style.engine import analyze_features
from filmlook_ml.style.features import WORK_SIDE, resize_for_style, style_features

JS = P.ROOT / "js" / "filmlook-style" / "model-data.js"


def _mid(how):
    """Resampler used for the 2x intermediate (what differs between browsers); box2x always follows."""
    def f(a, size):
        if how == "area":
            return cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        if how == "nearest":   # worst-case aliasing (no smoothing at all)
            return cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)
        if how == "cv_linear":  # plain bilinear WITHOUT antialiasing (aliasing like a naive drawImage)
            return cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)
        if how == "steps_linear":  # index.js progressive halving with non-antialiased bilinear steps
            im = a
            while im.shape[1] / 2 >= size[0] and im.shape[0] / 2 >= size[1]:
                im = cv2.resize(im, (max(size[0], im.shape[1] // 2), max(size[1], im.shape[0] // 2)),
                                interpolation=cv2.INTER_LINEAR)
            return cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        m = {"pil_bilinear": Image.BILINEAR, "pil_bicubic": Image.BICUBIC}[how]
        return np.asarray(Image.fromarray(a).resize(size, m))
    return f


def _resize(a, how):
    return resize_for_style(a, WORK_SIDE, _mid(how))


def perturb(a, kind):
    if kind == "jpeg75":
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=75)
        return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "crop4":
        h, w = a.shape[:2]
        dy, dx = int(h * 0.02), int(w * 0.02)
        return a[dy:h - dy, dx:w - dx]
    if kind == "exposure+8":
        return np.clip(a.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    if kind == "flip":
        return a[:, ::-1].copy()
    raise ValueError(kind)


def main():
    q = json.loads(JS.read_text().split("module.exports = ", 1)[1].rstrip().rstrip(";"))["quantiles"]
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    raw = {r["source_id"]: r["path"] for r in read_jsonl(P.MANIFESTS / "raw_commoncatalog.jsonl")}
    rows = [r for r in man if r["edit_type"] == "original" and r["dataset_id"] == "commoncatalog_ccby"
            and r["split"] in ("val", "test") and r.get("source_id") in raw]
    RES = ["pil_bilinear", "pil_bicubic", "cv_linear", "steps_linear", "nearest"]
    variants = [*RES, "jpeg75", "crop4", "exposure+8", "flip"]
    agree = defaultdict(list)
    status_agree = defaultdict(list)
    dsim = defaultdict(list)
    rate = defaultdict(list)
    tops = Counter()
    for r in rows:
        full = np.asarray(Image.open(raw[r["source_id"]]).convert("RGB"))
        base_img = _resize(full, "area")
        base = analyze_features(style_features(base_img), q)
        m0 = base["matches"][0]
        rate[r.get("medium")].append(base["status"] == "match")
        for s in (r.get("subjects") or ["other"]):
            rate["subject:" + s].append(base["status"] == "match")
        if base["status"] == "match":
            tops[m0["id"]] += 1
        for v in variants:
            img = _resize(full, v) if v in RES else perturb(base_img, v)
            o = analyze_features(style_features(img), q)
            agree[v].append(o["matches"][0]["id"] == m0["id"])
            status_agree[v].append(o["status"] == base["status"])
            dsim[v].append(abs(o["matches"][0]["similarity"] - m0["similarity"]) if o["matches"][0]["id"] == m0["id"]
                           else np.nan)
    res = {"n_images": len(rows), "note": "held-out CommonCatalog originals; no labels involved",
           "stability": {v: {"top1_same": float(np.mean(agree[v])), "status_same": float(np.mean(status_agree[v])),
                             "mean_abs_similarity_change": float(np.nanmean(dsim[v]))} for v in variants},
           "reliable_match_rate": {k: {"n": len(v), "rate": float(np.mean(v))} for k, v in sorted(rate.items(), key=str)},
           "reliable_top1_distribution": dict(tops.most_common())}
    (P.REPORTS / "style_stability.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
