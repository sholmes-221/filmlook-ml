#!/usr/bin/env python
"""Evaluate the existing EXIF-trained film-authenticity student (checkpoints/student_sup) the hard way.

  python scripts/eval_film_authenticity.py     # -> reports/film_authenticity_eval.json

Labels are EXIF capture medium (CommonCatalog) and the owner's records (BlueNeg): weak, not proof of provenance.
Reports, with uploader-grouped bootstrap CIs:
  within-source  CommonCatalog TEST originals, film_scan vs digital (camera + phone), AUROC, Brier, ECE;
  per-scanner    AUROC of each scanner family vs digital (cross-camera generalisation);
  cross-source   BlueNeg (all / intact) vs CommonCatalog digital; BlueNeg vs CommonCatalog FILM (both film: an AUROC
                 far from 0.5 means the model separates SOURCES, not film from digital);
  source probe   a logistic regression on the 28 measured statistics, BlueNeg film vs CommonCatalog film: how
                 separable the two film sources are regardless of the model;
  robustness     JPEG q55, 0.5x rescale, 80 % centre crop on the within-source test: AUROC and flip rate.
"""
import io
import json
import re
import sys
import time
from collections import Counter

import numpy as np
import onnxruntime as ort
from PIL import Image

from filmlook_infer.compact import preprocess
from filmlook_infer.features import compute_features, feature_vector
from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import filter_manifest

CK = P.CHECKPOINTS / "student_sup"
N_BOOT = 500


def scanner_family(camera: str | None) -> str:
    c = (camera or "").lower()
    for k in ("frontier", "noritsu", "pakon", "coolscan", "epson", "plustek", "imacon", "minolta", "canon", "hasselblad"):
        if k in c:
            return k
    return "other" if c else "unknown"


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot_ci(y, s, groups, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y, s, groups = map(np.asarray, (y, s, groups))
    ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(auroc(y[idx], s[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(e), 4)


def main():
    t0 = time.time()
    meta = json.loads((CK / "meta.json").read_text())
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(CK / "model.onnx"), so, providers=["CPUExecutionProvider"])
    ci = meta["outputs"].index("capture_type"); fi = meta["heads"]["capture_type"].index("film_scan")

    def pfilm(ims):
        out = []
        for i in range(0, len(ims), 64):
            xs = np.concatenate([preprocess(im) for im in ims[i:i + 64]])
            lg = sess.run(None, {"pixel_values": xs})[ci]
            e = np.exp(lg - lg.max(1, keepdims=True)); out.append((e / e.sum(1, keepdims=True))[:, fi])
        return np.concatenate(out)

    man = filter_manifest(read_jsonl(P.MANIFESTS / "manifest.jsonl"))
    orig = [r for r in man if r["edit_type"] == "original" and r["dataset_id"] == "commoncatalog_ccby"]
    test = [r for r in orig if r["split"] == "test" and r.get("medium") in ("film_scan", "digital_camera", "smartphone")]
    ims = [load_image(r["path"], max_side=1280) for r in test]
    y = np.array([r["medium"] == "film_scan" for r in test]).astype(int)
    g = np.array([r["group_id"] for r in test])
    p = pfilm(ims)
    rep = {"model": str(CK.relative_to(P.ROOT)), "labels": "EXIF capture medium (CommonCatalog) / owner record (BlueNeg); weak labels",
           "within_source_test": {"n": len(test), "n_film": int(y.sum()), "auroc": round(auroc(y, p), 4), "ci95": boot_ci(y, p, g),
                                  "brier": round(float(np.mean((p - y) ** 2)), 4), "ece_10bin": ece(y, p),
                                  "mean_p_film": {"film": round(float(p[y == 1].mean()), 3), "digital": round(float(p[y == 0].mean()), 3)}}}
    # per scanner family vs all digital
    fam = np.array([scanner_family(r.get("camera")) if r["medium"] == "film_scan" else "digital" for r in test])
    per = {}
    for f in sorted(set(fam) - {"digital"}):
        m = (fam == f) | (fam == "digital")
        per[f] = {"n_film": int((fam == f).sum()), "auroc_vs_digital": round(auroc(y[m], p[m]), 4), "mean_p_film": round(float(p[fam == f].mean()), 3)}
    rep["per_scanner_family"] = per
    # cross-source
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    bim = [load_image(r["path"], max_side=1280) for r in bn]
    pb = pfilm(bim)
    intact = np.array([r["partition"] == "blue-intact" for r in bn])
    dig = [r for r in orig if r["split"] in ("val", "test") and r.get("medium") in ("digital_camera", "smartphone")]
    ccf = [r for r in orig if r["split"] in ("val", "test") and r.get("medium") == "film_scan"]
    pd = pfilm([load_image(r["path"], max_side=1280) for r in dig]); pf = pfilm([load_image(r["path"], max_side=1280) for r in ccf])
    rep["cross_source"] = {"blueneg_all_vs_cc_digital": {"auroc": round(auroc(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd]), 4), "n": [len(bn), len(dig)]},
                           "blueneg_intact_vs_cc_digital": {"auroc": round(auroc(np.r_[np.ones(intact.sum()), np.zeros(len(dig))], np.r_[pb[intact], pd]), 4)},
                           "blueneg_vs_cc_film_both_film": {"auroc": round(auroc(np.r_[np.ones(len(bn)), np.zeros(len(ccf))], np.r_[pb, pf]), 4), "n": [len(bn), len(ccf)],
                                                            "reading": "0.5 = no source preference; far from 0.5 = the model separates sources"},
                           "mean_p_film": {"blueneg_intact": round(float(pb[intact].mean()), 3), "blueneg_corrupted": round(float(pb[~intact].mean()), 3),
                                           "cc_film": round(float(pf.mean()), 3), "cc_digital": round(float(pd.mean()), 3)}}
    # source-identity probe on measured statistics (film vs film)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Fb = np.stack([feature_vector(compute_features(im)) for im in bim]); Ff = np.stack([feature_vector(compute_features(load_image(r["path"], max_side=1280))) for r in ccf])
    Xs = np.vstack([Fb, Ff]); ys = np.r_[np.ones(len(Fb)), np.zeros(len(Ff))]
    grp = np.r_[[r["roll_id"] for r in bn], [r["group_id"] for r in ccf]]
    rng = np.random.default_rng(0); ug = np.unique(grp); test_g = set(rng.choice(ug, len(ug) // 3, replace=False))
    te = np.array([x in test_g for x in grp]); tr = ~te
    sc = StandardScaler().fit(Xs[tr]); m = LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced").fit(sc.transform(Xs[tr]), ys[tr])
    pred = m.predict(sc.transform(Xs[te]))
    rep["source_identity_probe_measured_stats"] = {"blueneg_vs_cc_film_accuracy": round(float((pred == ys[te]).mean()), 4), "chance": round(float(max(ys[te].mean(), 1 - ys[te].mean())), 4),
                                                   "n_test": int(te.sum()), "reading": "high = the two film sources differ systematically in measured statistics (scanner, era, resolution), so cross-source film-vs-digital numbers partly measure source identity"}
    # robustness on the within-source test
    def perturb(im, kind):
        if kind == "jpeg_q55":
            b = io.BytesIO(); im.save(b, "JPEG", quality=55); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        if kind == "rescale_0.5":
            return im.resize((max(64, im.width // 2), max(64, im.height // 2)), Image.BILINEAR)
        w, h = im.size; return im.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    rob = {}
    base = p > 0.5
    for kind in ("jpeg_q55", "rescale_0.5", "crop_80pct"):
        pp = pfilm([perturb(im, kind) for im in ims])
        rob[kind] = {"auroc": round(auroc(y, pp), 4), "flip_rate": round(float(((pp > 0.5) != base).mean()), 4)}
    rep["robustness_within_source"] = rob
    rep["seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / "film_authenticity_eval.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
