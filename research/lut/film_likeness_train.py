"""Fit and evaluate the deterministic film-likeness estimator on an uploader-disjoint split of real scans vs digital photos.

    python -m research.lut.film_likeness_train build     # dataset manifest + features -> data/lut/film_likeness/
    python -m research.lut.film_likeness_train fit       # logistic regression, calibration, held-out evaluation, robustness
                                                         # -> research/lut/db/film_likeness.json, reports/lut/film_likeness_eval.json

Data (all already on disk, licences per configs/datasets.yaml; evaluation-only sets are used only to fit a 36-weight
logistic regression, which the registry allows as evaluation/analysis, and nothing is distilled from a model):
  film     commoncatalog_film + commoncatalog_film_hd (uploader-tagged scans, class B, all film types), artefact_film
           originals (class A), and the film-scan rows of the CommonCatalog training manifest (scanner EXIF, no stock)
  digital  CommonCatalog digital_camera / smartphone / *_filtered rows
Split by uploader hash (sha1 % 100 < 50 -> train, else test); no uploader on both sides. Metrics on held-out
uploaders: AUROC with uploader bootstrap (all images and colour-only), ECE and a 10-bin reliability table, baselines
(monochrome flag, fine-noise sigma alone), robustness to JPEG q50 / q80 and 0.5x resizing (AUROC on a perturbed test
subsample).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.lut.engine.film_likeness import FEATURE_NAMES, film_features
from research.lut.evaluate import auroc, grouped_bootstrap

OUT = Path("data/lut/film_likeness")
DB_JSON = Path("research/lut/db/film_likeness.json")


def build_manifest() -> list[dict]:
    rows = []
    for ds in ("commoncatalog_film", "commoncatalog_film_hd", "artefact_film"):
        for l in open(f"data/manifests/{ds}_manifest.jsonl"):
            r = json.loads(l)
            if r.get("kind") == "film_restored":
                continue
            rows.append({"path": r["path"], "label": 1, "uploader": str(r.get("uploader_id") or r.get("uploader") or ds), "dataset": ds,
                         "film_type": r.get("film_type"), "colour": r.get("film_type") not in ("bw_negative", "bw_negative_c41") and not r.get("monochrome_tag")})
    for l in open("data/manifests/raw_commoncatalog.jsonl"):
        r = json.loads(l)
        med = r.get("medium")
        if med == "film_scan":
            rows.append({"path": r["path"], "label": 1, "uploader": r["group_id"], "dataset": "commoncatalog_train_filmscan", "film_type": "unknown",
                         "colour": not r.get("monochrome_tag")})
        elif med in ("digital_camera", "smartphone", "smartphone_filtered", "digital_filtered"):
            rows.append({"path": r["path"], "label": 0, "uploader": r["group_id"], "dataset": "commoncatalog_" + med, "film_type": None,
                         "colour": not r.get("monochrome_tag")})
    for r in rows:
        r["split"] = "train" if int(hashlib.sha1(r["uploader"].encode()).hexdigest(), 16) % 100 < 50 else "test"
    return [r for r in rows if Path(r["path"]).exists()]


def _feat(path):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im.draft("RGB", (2048, 2048))
    img = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return film_features(img)


def _feat_perturbed(args):
    path, kind = args
    from PIL import Image, ImageOps
    im = Image.open(path)
    im.draft("RGB", (2048, 2048))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if kind == "resize_0.5":
        im = im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.BOX)
    elif kind.startswith("jpeg_q"):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(kind[6:]))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return film_features(np.asarray(im))


def cmd_build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    with ProcessPoolExecutor(a.workers) as ex:
        F = np.stack(list(ex.map(_feat, [r["path"] for r in rows], chunksize=8)))
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.save(OUT / "features.npy", F)
    y = np.array([r["label"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    print(f"{len(rows)} rows; film {int(y.sum())} digital {int((1 - y).sum())}; train {int((sp == 'train').sum())} test {int((sp == 'test').sum())}; "
          f"uploaders {len({r['uploader'] for r in rows})}")


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    table, e = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            table.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_film": float(y[m].mean())})
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e), table


def cmd_fit(a):
    from sklearn.linear_model import LogisticRegression
    rows = [json.loads(l) for l in open(OUT / "manifest.jsonl")]
    F = np.nan_to_num(np.load(OUT / "features.npy")).astype(np.float64)
    y = np.array([r["label"] for r in rows])
    up = np.array([r["uploader"] for r in rows])
    col = np.array([bool(r["colour"]) for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    te = ~tr
    mean, std = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = (F - mean) / std
    clf = LogisticRegression(C=a.C, max_iter=5000, class_weight="balanced").fit(Z[tr], y[tr])
    w, b = clf.coef_[0], float(clf.intercept_[0])
    # Platt recalibration from grouped 5-fold out-of-fold logits on the train uploaders
    ug = np.unique(up[tr])
    fold_of = {g: int(hashlib.sha1(g.encode()).hexdigest(), 16) % 5 for g in ug}
    folds = np.array([fold_of[g] for g in up[tr]])
    oof = np.zeros(tr.sum())
    Ztr, ytr = Z[tr], y[tr]
    for k in range(5):
        m = folds == k
        c = LogisticRegression(C=a.C, max_iter=5000, class_weight="balanced").fit(Ztr[~m], ytr[~m])
        oof[m] = c.decision_function(Ztr[m])
    cal = LogisticRegression(C=1e6, max_iter=5000).fit(oof[:, None], ytr)
    cal_a, cal_c = float(cal.coef_[0, 0]), float(cal.intercept_[0])

    def prob(Zx):
        return 1 / (1 + np.exp(-(cal_a * (Zx @ w + b) + cal_c)))

    p = prob(Z)
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_film": int(y[te].sum()), "n_test_digital": int((1 - y[te]).sum()),
           "n_test_uploaders": int(len(np.unique(up[te]))), "C": a.C, "feature_names": FEATURE_NAMES,
           "weights": {n: round(float(v), 4) for n, v in sorted(zip(FEATURE_NAMES, w), key=lambda t: -abs(t[1]))}}

    def auc_ci(mask):
        pos, neg = p[mask & (y == 1)], p[mask & (y == 0)]
        a0 = auroc(pos, neg)
        # uploader bootstrap
        ug = np.unique(up[mask])
        rng = np.random.default_rng(0)
        vals = []
        idx = {g: np.where(mask & (up == g))[0] for g in ug}
        for _ in range(500):
            sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
            vals.append(auroc(p[sel][y[sel] == 1], p[sel][y[sel] == 0]))
        return [a0, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    res["auroc_test_all"] = auc_ci(te)
    res["auroc_test_colour_only"] = auc_ci(te & col)
    res["auroc_train_all"] = auc_ci(tr)
    ece, table = reliability(p[te], y[te])
    res["ece_test"], res["reliability_test"] = ece, table
    res["accuracy_test_at_0.5"] = float(((p[te] >= 0.5) == (y[te] == 1)).mean())
    res["grouped_accuracy_test"] = grouped_bootstrap(((p[te] >= 0.5) == (y[te] == 1)), up[te])
    # baselines on the same held-out uploaders
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    res["baseline_auroc_monochrome_flag"] = auroc(F[te & (y == 1), fi["monochrome_flag"]], F[te & (y == 0), fi["monochrome_flag"]])
    res["baseline_auroc_fine_noise_sigma"] = auroc(F[te & (y == 1), fi["fine_noise_sigma"]], F[te & (y == 0), fi["fine_noise_sigma"]])
    res["baseline_auroc_noise_grain_384"] = auroc(F[te & (y == 1), fi["noise_grain"]], F[te & (y == 0), fi["noise_grain"]])
    res["single_feature_auroc_test"] = {n: round(auroc(F[te & (y == 1), i], F[te & (y == 0), i]), 3) for i, n in enumerate(FEATURE_NAMES)}
    # robustness on a test subsample
    rng = np.random.default_rng(0)
    sub = rng.choice(np.where(te)[0], min(a.n_robust, int(te.sum())), replace=False)
    rob = {"clean_subsample": auroc(p[sub][y[sub] == 1], p[sub][y[sub] == 0]), "n": int(len(sub))}
    for kind in ("jpeg_q80", "jpeg_q50", "resize_0.5"):
        with ProcessPoolExecutor(a.workers) as ex:
            Fp = np.nan_to_num(np.stack(list(ex.map(_feat_perturbed, [(rows[i]["path"], kind) for i in sub], chunksize=8)))).astype(np.float64)
        pp = prob((Fp - mean) / std)
        rob[kind] = {"auroc": auroc(pp[y[sub] == 1], pp[y[sub] == 0]), "mean_abs_shift": float(np.abs(pp - p[sub]).mean())}
    res["robustness"] = rob
    print(json.dumps({k: v for k, v in res.items() if k not in ("weights", "reliability_test", "single_feature_auroc_test", "feature_names")}, indent=1))
    print("reliability:", table)
    print("top weights:", list(res["weights"].items())[:10])
    Path("reports/lut").mkdir(exist_ok=True, parents=True)
    Path("reports/lut/film_likeness_eval.json").write_text(json.dumps(res, indent=1, default=float))
    model = {"feature_names": FEATURE_NAMES, "mean": mean.round(6).tolist(), "std": std.round(6).tolist(), "weights": w.round(6).tolist(), "bias": round(b, 6),
             "calibration": {"a": cal_a, "c": cal_c, "method": "Platt on grouped 5-fold out-of-fold logits (train uploaders)"},
             "bands": {"ambiguous": 0.35, "film": 0.65},
             "training": {"n_train": int(tr.sum()), "film_sources": ["commoncatalog_film", "commoncatalog_film_hd", "artefact_film", "commoncatalog_train_filmscan"],
                          "digital_sources": ["commoncatalog digital_camera / smartphone / filtered"], "split": "uploader sha1 % 100 < 50", "fitted": "2026-11-10"},
             "validation_summary": {"auroc_heldout_uploaders_all": res["auroc_test_all"], "auroc_heldout_colour_only": res["auroc_test_colour_only"],
                                    "ece_heldout": ece, "robustness": rob},
             "note": "measured estimate of scan-likeness (Flickr film scans vs Flickr digital captures); not a stock, camera or process identification"}
    DB_JSON.write_text(json.dumps(model, indent=1))
    print("wrote", DB_JSON, "and reports/lut/film_likeness_eval.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=8)
    f = sub.add_parser("fit")
    f.add_argument("--C", type=float, default=0.3)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--n-robust", type=int, default=400)
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_fit(a)


if __name__ == "__main__":
    main()
