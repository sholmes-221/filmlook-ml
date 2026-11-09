"""Film-style family recognition on independent real scans (Commons, uploader-declared stocks).

Families (overlapping looks collapsed to what the labelled data can support):
  warm_negative  Kodak Gold 200                         cool_negative  Fuji Superia 200 / X-TRA 400 / X-TRA 800 / Reala
  vivid_negative Kodak Ektar 100                        bw             Ilford HP5 Plus, Kodak Tri-X
Portra (2 scans) is too small to evaluate and is left out of the supervised test.

Methods, all with leave-one-uploader-out splits (no photographer in both train and test):
  stats    26 colour / tone / grain statistics, logistic regression
  dinov2   DINOv2-small CLS embedding (frozen), logistic regression
  both     concatenation
  physics  zero-shot: nearest family prototype built only from spektrafilm renders of *digital* photos (no real scans
           used for training), colour families only; plus a top-k stock recommendation check.
Run with the filmlook-ml venv: python film_physics/evaluation/family_eval.py <manifest> <img dir> <spk out dir> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

FAMILY = {
    "Photographs taken on Kodak Gold 200 film": "warm_negative",
    "Taken on Fuji Superia 200": "cool_negative", "Taken on Fuji Superia X-TRA 400": "cool_negative",
    "Taken on Fuji Superia X-TRA 800": "cool_negative", "Taken on Fuji Superia Reala": "cool_negative",
    "Photographs taken on Kodak Ektar 100 film": "vivid_negative",
    "Taken on Ilford HP5 plus 400": "bw", "Photographs taken on Kodak TRI-X film": "bw",
}
STOCK_FAMILY = {"kodak_gold_200": "warm_negative", "kodak_ultramax_400": "warm_negative", "fujifilm_xtra_400": "cool_negative",
                "fujifilm_c200": "cool_negative", "kodak_ektar_100": "vivid_negative", "kodak_portra_400": "soft_negative"}


def lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def stats(img: Image.Image) -> np.ndarray:
    im = img.convert("RGB"); im.thumbnail((256, 256))
    rgb = np.asarray(im, dtype=np.float64) / 255
    L = lab(rgb); l, a, b = L[..., 0], L[..., 1], L[..., 2]
    chroma = np.hypot(a, b); hue = np.degrees(np.arctan2(b, a)) % 360
    sectors = [float(((hue >= s) & (hue < s + 60) & (chroma > 8)).mean()) for s in range(0, 360, 60)]
    sh, hi = l < 30, l > 70
    g = l / 100
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((64, 64)).resize(g.shape[::-1], Image.BILINEAR), dtype=np.float64) / 255
    return np.array([l.mean(), l.std(), a.mean(), b.mean(), a.std(), b.std(), chroma.mean(), np.percentile(chroma, 90), *sectors,
                     a[sh].mean() if sh.any() else 0, b[sh].mean() if sh.any() else 0, a[hi].mean() if hi.any() else 0, b[hi].mean() if hi.any() else 0,
                     *np.percentile(l, [5, 50, 95]), np.percentile(l, 95) - np.percentile(l, 5), float((g - small).std()), float(chroma.mean() < 4)])


def dinov2(paths: list[Path]) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
    model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
            x = proc(images=batch, return_tensors="pt").to(dev)
            cls = model(**x).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out)


def logo_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        if len(np.unique(y[~test])) < 2:
            pred[test] = Counter(y[~test]).most_common(1)[0][0]; continue
        sc = StandardScaler().fit(X[~test])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(X[~test]), y[~test])
        pred[test] = clf.predict(sc.transform(X[test]))
    labels = sorted(set(y))
    cm = [[int(((y == a) & (pred == b)).sum()) for b in labels] for a in labels]
    return {"accuracy": round(accuracy_score(y, pred), 3), "macro_f1": round(f1_score(y, pred, average="macro"), 3), "labels": labels, "confusion": cm}


def main(manifest: Path, imgdir: Path, spk: Path, out: Path) -> None:
    items = json.loads(manifest.read_text())
    for it in items:
        it["file"] = it.get("file") or hashlib.sha1(it["title"].encode()).hexdigest()[:16] + ".jpg"
    film = [it for it in items if it["label"] == "film" and it["category"] in FAMILY and (imgdir / it["file"]).exists()]
    y = np.array([FAMILY[it["category"]] for it in film]); groups = np.array([it["user"] for it in film])
    S = np.stack([stats(Image.open(imgdir / it["file"])) for it in film])
    D = dinov2([imgdir / it["file"] for it in film])
    res = {"n": len(film), "uploaders": int(len(set(groups))), "class_counts": dict(Counter(y)),
           "majority_baseline": round(Counter(y).most_common(1)[0][1] / len(y), 3)}
    for name, X in (("stats", S), ("dinov2", D), ("both", np.concatenate([S, D * 10], 1))):
        res[name] = logo_eval(X, y, groups)
        print(name, res[name]["accuracy"], res[name]["macro_f1"], flush=True)
    # Physics zero-shot on colour families: prototypes only from spektrafilm renders of digital photos.
    proto, proto_stock = {}, {}
    for stock, fam in STOCK_FAMILY.items():
        files = sorted((spk / stock).glob("*.png"))
        if files:
            feats = np.stack([stats(Image.open(f)) for f in files])
            proto_stock[stock] = feats.mean(0); proto.setdefault(fam, []).append(feats)
    if proto:
        allsim = np.concatenate([f for fs in proto.values() for f in fs])
        mu, sd = allsim.mean(0), allsim.std(0) + 1e-6
        fam_centroid = {f: (np.concatenate(v).mean(0) - mu) / sd for f, v in proto.items()}
        colour = y != "bw"
        Xc = (S[colour] - mu) / sd; yc = y[colour]
        fams = [f for f in fam_centroid if f in set(yc)]
        pred = np.array([min(fams, key=lambda f: np.linalg.norm(x - fam_centroid[f])) for x in Xc])
        res["physics_zero_shot"] = {"n": int(colour.sum()), "families": fams, "accuracy": round(float((pred == yc).mean()), 3),
                                    "chance": round(1 / len(fams), 3), "majority": round(Counter(yc).most_common(1)[0][1] / len(yc), 3),
                                    "pred_counts": dict(Counter(pred))}
        st = {s: (v - mu) / sd for s, v in proto_stock.items()}
        hits = {1: 0, 3: 0}
        for x, fam in zip(Xc, yc):
            order = sorted(st, key=lambda s: np.linalg.norm(x - st[s]))
            for k in hits:
                hits[k] += int(any(STOCK_FAMILY[s] == fam for s in order[:k]))
        res["physics_topk_family_hit"] = {k: round(v / len(yc), 3) for k, v in hits.items()}
        print("physics", res["physics_zero_shot"], res["physics_topk_family_hit"], flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
