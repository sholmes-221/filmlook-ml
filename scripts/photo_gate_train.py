#!/usr/bin/env python
"""Photograph vs non-photograph gate: DINOv2-small CLS features + a linear classifier, three outcomes.

  python scripts/photo_gate_train.py [--export]

Data (all legally usable; labels by construction or from catalogue records, never from a model's opinion):
  photographs, training sources: commoncatalog (CC-BY originals), commoncatalog_film, commons_film, refdb (public-domain
      FSA/OWI scans), aic_photographs, met_photographs, artefact_film
  photographs, held-out test sources: commoncatalog_film_hd (uploaders excluded from every other set), PD12M iNaturalist
  non-photographs, training: locally rendered documents / charts / posters / UI (family a), AIC paintings and prints,
      Z-Image-Turbo illustrations (prompt family a)
  non-photographs, held-out test: the same kinds from *unseen* families (b), AIC drawings/watercolours and posters
  composites for training (family a, colour CommonCatalog photos): posters over a photo and UIs containing a photo
      (not photo), photos with a small caption (photo); real UI test: screenshots of FilmAnalysis's own pages
  hard (reported per kind): photos with burned-in captions (photo), UI screenshots containing a photo (not photo),
      posters over a photo (not photo), transformed test photos (blur, motion blur, heavy grain, dark, monochrome)
  ambiguous (reported, never scored as right or wrong): photorealistic generated images
Training rows are split 80/20 (train/val) by a stable hash; every choice (C, thresholds) is made on val; test is read
once. Thresholds: REJECT below t_lo, ACCEPT at or above t_hi, UNCERTAIN between, chosen on val so that at most 0.5 %
of photographs are rejected and at most 1 % of non-photographs are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

RAW = P.DATA / "raw"
GATE = RAW / "photo_gate"
FEATS = P.DATA / "photo_gate" / "features_cls_patch.npz"  # 768 = CLS (384) + mean of patch tokens (384), both L2-normalised
OUT = P.REPORTS / "photo_gate"
FRR_MAX, FAR_MAX = 0.002, 0.02  # A false reject cannot be undone; a false accept still meets the provider's own check.
CAPS = {"commoncatalog": 500, "commoncatalog_film": 500, "commons_film": 266, "refdb": 400, "aic_photographs": 400, "met_photographs": 400, "artefact_film": 88}


def files(d: Path, cap: int | None = None) -> list[Path]:
    fs = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))
    fs = sorted(fs, key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
    return fs[:cap] if cap else fs


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def collect() -> list[dict]:
    rows = []
    for src, cap in CAPS.items():
        for f in files(RAW / src, cap):
            rows.append({"path": f, "label": 1, "kind": "photo_" + src, "split": "train"})
    for f in files(RAW / "commoncatalog_film_hd", 474):
        rows.append({"path": f, "label": 1, "kind": "photo_commoncatalog_film_hd", "split": "test"})
    for m in jsonl(GATE / "pd12m_inaturalist" / "manifest_pd12m.jsonl") + jsonl(GATE / "manifest_synthetic.jsonl") + jsonl(GATE / "aic" / "manifest_aic.jsonl") + jsonl(GATE / "generated" / "manifest_generated.jsonl") + jsonl(GATE / "composites" / "manifest_composites.jsonl"):
        p = P.DATA / m["path"]
        if p.exists():
            rows.append({"path": p, "label": m["label"], "kind": m["kind"] + ("" if m["split"] in ("hard", "ambiguous") else f"_{m['family']}"), "split": m["split"]})
    for r in rows:
        if r["split"] == "train":
            r["split"] = "val" if int(hashlib.sha256(str(r["path"]).encode()).hexdigest(), 16) % 5 == 0 else "train"
    return rows


TRANSFORMS = {
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(6)),
    "motion": lambda im: im.filter(ImageFilter.BoxBlur(1)).transform(im.size, Image.AFFINE, (1, 0.04, 0, 0, 1, 0)).filter(ImageFilter.BLUR),
    "grain": lambda im: Image.fromarray(np.clip(np.asarray(im.convert("RGB"), np.float32) + np.random.default_rng(0).normal(0, 28, (im.height, im.width, 3)), 0, 255).astype("uint8")),
    "dark": lambda im: ImageEnhance.Brightness(im).enhance(0.18),
    "mono": lambda im: ImageOps.grayscale(im).convert("RGB"),
}


def load_image(r: dict) -> Image.Image:
    im = Image.open(r["path"]); im.load(); im = im.convert("RGB")
    im.thumbnail((1024, 1024))
    if "transform" in r:
        im = TRANSFORMS[r["transform"]](im)
    return im


@torch.no_grad()
def features(rows: list[dict]) -> np.ndarray:
    from transformers import AutoModel
    cache = dict(np.load(FEATS, allow_pickle=True)) if FEATS.exists() else {}
    keys = [f"{r['path']}|{r.get('transform', '')}" for r in rows]
    known = dict(zip(cache.get("keys", []), cache.get("X", []))) if cache else {}
    todo = [i for i, k in enumerate(keys) if k not in known]
    if todo:
        m = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        for b in range(0, len(todo), 64):
            idx = todo[b:b + 64]
            px = np.stack([dinov2_pixels(load_image(rows[i]), 224) for i in idx])
            h = m(pixel_values=torch.from_numpy(px).cuda()).last_hidden_state.float()
            v = torch.cat([torch.nn.functional.normalize(h[:, 0], dim=1), torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)], dim=1).cpu().numpy()
            for i, vec in zip(idx, v):
                known[keys[i]] = vec
            print(f"features {b + len(idx)}/{len(todo)}", flush=True)
        FEATS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FEATS, keys=np.array(list(known)), X=np.stack(list(known.values())))
    return np.stack([known[k] for k in keys]).astype(np.float32)


def thresholds(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    photos, others = np.sort(p[y == 1]), np.sort(p[y == 0])
    t_lo = float(photos[int(np.floor(FRR_MAX * len(photos)))]) if len(photos) else 0.5  # at most FRR_MAX of photos below
    t_hi = float(others[min(len(others) - 1, int(np.ceil((1 - FAR_MAX) * len(others))))]) + 1e-6  # at most FAR_MAX of others at or above
    if t_lo > t_hi:  # Classes separate cleanly on val: no uncertain band is needed.
        t_lo = t_hi = (t_lo + t_hi) / 2
    return t_lo, t_hi


def outcome(p: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    return np.where(p >= t_hi, "accept", np.where(p < t_lo, "reject", "uncertain"))


def report(p, y, kinds, t_lo, t_hi) -> dict:
    o = outcome(p, t_lo, t_hi)
    res = {"n": int(len(y)), "photos": int((y == 1).sum()), "non_photos": int((y == 0).sum()),
           "false_rejection_rate": float(((o == "reject") & (y == 1)).sum() / max((y == 1).sum(), 1)),
           "false_acceptance_rate": float(((o == "accept") & (y == 0)).sum() / max((y == 0).sum(), 1)),
           "uncertain_rate_photos": float(((o == "uncertain") & (y == 1)).sum() / max((y == 1).sum(), 1)),
           "uncertain_rate_non_photos": float(((o == "uncertain") & (y == 0)).sum() / max((y == 0).sum(), 1)),
           "accuracy_at_0.5": float(((p >= 0.5) == (y == 1)).mean()),
           "decided_accuracy": float(((o == "accept") == (y == 1))[o != "uncertain"].mean()) if (o != "uncertain").any() else None}
    res["per_kind"] = {k: {"n": int((kinds == k).sum()), "label": int(y[kinds == k][0]), "accept": float((o[kinds == k] == "accept").mean()),
                           "uncertain": float((o[kinds == k] == "uncertain").mean()), "reject": float((o[kinds == k] == "reject").mean())} for k in sorted(set(kinds))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--export", action="store_true"); ap.add_argument("--features", choices=("cls", "cat"), default="cls")
    a = ap.parse_args()
    rows = collect()
    test_photos = [r for r in rows if r["split"] == "test" and r["label"] == 1]
    for t in TRANSFORMS:
        rows += [{"path": r["path"], "label": 1, "kind": f"hard_photo_{t}", "split": "hard", "transform": t} for r in test_photos[:120]]
    t0 = time.time(); X_all = features(rows); print(f"features {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    X = X_all if a.features == "cat" else X_all[:, :384]
    y = np.array([r["label"] if r["label"] is not None else -1 for r in rows]); split = np.array([r["split"] for r in rows]); kinds = np.array([r["kind"] for r in rows])
    tr, va = split == "train", split == "val"
    results = {"counts": {s: {"photos": int(((split == s) & (y == 1)).sum()), "non_photos": int(((split == s) & (y == 0)).sum()), "other": int(((split == s) & (y == -1)).sum())} for s in ("train", "val", "test", "hard", "ambiguous")}, "models": {}}
    candidates = {}
    for C in (0.1, 0.3, 1, 3, 10):
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
        candidates[f"logreg_C{C}"] = m
    for C in (0.01, 0.03, 0.1, 0.3, 1):
        m = CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", max_iter=20000), method="sigmoid", cv=5).fit(X[tr], y[tr])
        candidates[f"linsvm_C{C}"] = m
    for name, m in candidates.items():
        pv = m.predict_proba(X[va])[:, 1]
        t_lo, t_hi = thresholds(pv, y[va])
        val = report(pv, y[va], kinds[va], t_lo, t_hi)
        results["models"][name] = {"val": {k: v for k, v in val.items() if k != "per_kind"}, "t_lo": t_lo, "t_hi": t_hi}
    # Choose on val: fewest uncertain decisions subject to the FRR/FAR limits (already enforced by the thresholds), then accuracy.
    best = min(results["models"], key=lambda n: (results["models"][n]["val"]["uncertain_rate_photos"] + results["models"][n]["val"]["uncertain_rate_non_photos"], -results["models"][n]["val"]["accuracy_at_0.5"]))
    m = candidates[best]; t_lo, t_hi = results["models"][best]["t_lo"], results["models"][best]["t_hi"]
    for part in ("test", "hard"):
        mask = (split == part) & (y >= 0)
        results[part] = report(m.predict_proba(X[mask])[:, 1], y[mask], kinds[mask], t_lo, t_hi)
    amb = split == "ambiguous"
    if amb.any():
        pa = m.predict_proba(X[amb])[:, 1]; oa = outcome(pa, t_lo, t_hi)
        results["ambiguous_ai_photoreal"] = {"n": int(amb.sum()), "accept": float((oa == "accept").mean()), "uncertain": float((oa == "uncertain").mean()), "reject": float((oa == "reject").mean()),
                                             "note": "generated photorealistic images; reported only, the gate does not claim to detect AI generation"}
    results["chosen"] = {"model": best, "t_lo": t_lo, "t_hi": t_hi, "features": a.features}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{a.features}.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({"chosen": results["chosen"], "counts": results["counts"], "test": {k: v for k, v in results["test"].items() if k != "per_kind"}}, indent=1))
    for part in ("test", "hard"):
        for k, v in results[part]["per_kind"].items():
            print(f"{part:5s} {k:38s} n={v['n']:4d} label={v['label']} accept={v['accept']:.3f} uncertain={v['uncertain']:.3f} reject={v['reject']:.3f}")
    if "ambiguous_ai_photoreal" in results:
        print("ambiguous", results["ambiguous_ai_photoreal"])
    if a.export:
        export(m, best, t_lo, t_hi, results, a.features)


def export(m, name: str, t_lo: float, t_hi: float, results: dict, features: str = "cls") -> None:
    """Fold the classifier into one linear map over the CLS vector: p = sigmoid(a * (w.x + b) + c)."""
    if hasattr(m, "coef_"):
        w, b, a_, c = m.coef_[0], float(m.intercept_[0]), 1.0, 0.0
    else:  # CalibratedClassifierCV over LinearSVC: average the folds' linear maps (sigmoid of an affine map of w.x + b).
        ws = [cc.estimator.coef_[0] for cc in m.calibrated_classifiers_]; bs = [float(cc.estimator.intercept_[0]) for cc in m.calibrated_classifiers_]
        As = [float(cc.calibrators[0].a_) for cc in m.calibrated_classifiers_]; Bs = [float(cc.calibrators[0].b_) for cc in m.calibrated_classifiers_]
        w = np.mean([-A * wi for A, wi in zip(As, ws)], axis=0); b = float(np.mean([-A * bi - B for A, bi, B in zip(As, bs, Bs)])); a_, c = 1.0, 0.0
    version = time.strftime("photo-gate-%Y%m%d-") + hashlib.sha256(np.asarray(w, np.float32).tobytes()).hexdigest()[:8]
    art = {"version": version, "features": features, "encoder": "facebook/dinov2-small 224 (dinov2_pixels): " + ("CLS, L2-normalised" if features == "cls" else "[CLS, mean patch], each L2-normalised"), "classifier": name,
           "weights": [float(v) for v in w], "bias": b, "scale": a_, "offset": c, "t_lo": t_lo, "t_hi": t_hi,
           "outcomes": {"accept": "p >= t_hi", "uncertain": "t_lo <= p < t_hi", "reject": "p < t_lo"},
           "metrics": {"test": {k: v for k, v in results["test"].items() if k != "per_kind"}, "hard": {k: v for k, v in results["hard"].items() if k != "per_kind"}},
           "note": "p is a validation-calibrated photo score, not a guarantee; the gate does not detect AI-generated images"}
    d = P.CHECKPOINTS / "photo_gate"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{version}.json").write_text(json.dumps(art, indent=1))
    print("exported", d / f"{version}.json")


if __name__ == "__main__":
    main()
