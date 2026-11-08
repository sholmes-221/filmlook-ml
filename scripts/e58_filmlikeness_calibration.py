"""E58: film-likeness calibration on independent data, and rendering-family prototypes for the retrieval MVP.

Film-likeness. Two existing signals are scored on images none of the models saw: the v1 texture model's
`film_score` (checkpoints/best; trained on heuristic + synthetic edit orderings) and the supervised student's
P(film_scan) (checkpoints/student_sup; trained on EXIF capture labels of the TRAIN split). Positives: the TEST-split
film-scan originals of the main manifest plus the fresh `commoncatalog_film_hd` scans (never used for training or
selection). Negatives: the TEST-split digital-camera and smartphone originals. Groups (uploader / group_id) are
split into a FIT half and a READ half by seeded choice; Platt scaling (a 2-parameter logistic map) is fitted on the
FIT half for each signal and for their combination; AUROC, Brier and ECE are read on the READ half only, with CIs
by bootstrap over groups. Robustness: JPEG q50 and 0.5x rescale on the READ half (AUROC and decision flips).

Prototypes. Process-family centroids on the standardised 28 statistics are fitted on the SPENT AIC analysis half
and evaluated (balanced accuracy, nearest centroid) on the Met analysis half: a fit set disjoint from the read set.

Artifacts: checkpoints/retrieval_mvp/{calibration.json, prototypes.npz, meta.json}; report
reports/e58_filmlikeness_calibration.json. No training, no labels made by hand, no model output used as truth.

  .venv/bin/python scripts/e58_filmlikeness_calibration.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from eval_commons_film import boot_groups  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from filmlook_infer import FilmLookAnalyzer  # noqa: E402
from filmlook_infer.compact import CompactAnalyzer, preprocess  # noqa: E402
from filmlook_infer.features import compute_features, feature_vector  # noqa: E402
from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import filter_manifest  # noqa: E402

OUT_DIR = ROOT / "checkpoints" / "retrieval_mvp"
V1 = ROOT / "checkpoints" / "best"
SUP = ROOT / "checkpoints" / "student_sup"
MIN_FAMILY = 15


def auroc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def boot_auroc(y, s, groups, n=1000, seed=0):
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups); ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
        v = auroc(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float); edges = np.linspace(0, 1, bins + 1); total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return round(float(total), 4)


def platt(x, y, iters=500, lr=0.1):
    """Fit p = sigmoid(a*x + b) by gradient descent on log loss; x may be (n, d)."""
    X = np.asarray(x, float); X = X[:, None] if X.ndim == 1 else X; y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= lr * (Z.T @ g / len(y) + 1e-3 * w); b -= lr * g.mean()
    return {"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist()}


def apply_platt(cal, x):
    X = np.asarray(x, float); X = X[:, None] if X.ndim == 1 else X
    Z = (X - np.array(cal["mu"])) / np.array(cal["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(cal["w"]) + cal["b"])))


def perturb(im: Image.Image, kind: str) -> Image.Image:
    if kind == "jpeg50":
        b = io.BytesIO(); im.save(b, "JPEG", quality=50); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    w, h = im.size; return im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def main():
    t0 = time.time()
    v1 = FilmLookAnalyzer(V1, prefer_int8=False); sup = CompactAnalyzer(SUP, int8=False)
    ci = sup.meta["outputs"].index("capture_type"); fi = sup.meta["heads"]["capture_type"].index("film_scan")

    def p_film(im: Image.Image) -> float:
        out = sup.session.run(None, {"pixel_values": preprocess(im, sup.size)})[ci][0]; z = out - out.max(); p = np.exp(z) / np.exp(z).sum(); return float(p[fi])

    def signals(src) -> tuple[float, float, np.ndarray]:
        """src: a path or a seekable binary buffer (the runtime loaders take paths and file objects, not PIL images)."""
        r = v1.predict_raw(src)
        if hasattr(src, "seek"):
            src.seek(0)
        im = load_image(src, max_side=1280)
        return float(r["score"]), p_film(im), feature_vector(r["features"])

    def as_buffer(im: Image.Image) -> io.BytesIO:
        b = io.BytesIO(); im.save(b, "PNG", compress_level=1); b.seek(0); return b

    # ---- film-likeness rows
    man = filter_manifest(read_jsonl(P.MANIFESTS / "manifest.jsonl"))
    test = [r for r in man if r.get("edit_type") == "original" and r["split"] == "test" and r.get("medium") in ("film_scan", "digital_camera", "smartphone")]
    fresh = [r for r in read_jsonl(P.MANIFESTS / "commoncatalog_film_hd_manifest.jsonl") if r.get("film_stock")]
    rows = [(r["path"], 1 if r["medium"] == "film_scan" else 0, f"cc:{r['group_id']}", "test") for r in test] + [(r["path"], 1, f"hd:{r['uploader']}", "fresh") for r in fresh]
    y = np.array([r[1] for r in rows]); groups = np.array([r[2] for r in rows]); source = np.array([r[3] for r in rows])
    s_v1, s_sup, ims = [], [], []
    for path, *_ in rows:
        a, b, _ = signals(path); s_v1.append(a); s_sup.append(b); ims.append(path)
    s_v1, s_sup = np.array(s_v1), np.array(s_sup)
    # FIT / READ halves by group, seeded, stratified by label of the group's majority
    rng = np.random.default_rng(58); ug = np.unique(groups)
    pos_groups = [g for g in ug if y[groups == g].mean() >= 0.5]; neg_groups = [g for g in ug if y[groups == g].mean() < 0.5]
    fit_groups = set(rng.choice(pos_groups, len(pos_groups) // 2, replace=False)) | set(rng.choice(neg_groups, len(neg_groups) // 2, replace=False))
    fit = np.array([g in fit_groups for g in groups]); read = ~fit
    rep = {"experiment": "E58 film-likeness calibration + rendering prototypes", "n_rows": len(rows), "n_fit": int(fit.sum()), "n_read": int(read.sum()),
           "n_read_film": int(y[read].sum()), "n_read_digital": int((1 - y[read]).sum()), "n_read_groups": int(len(set(groups[read]))),
           "sources": {"test_originals": int((source == "test").sum()), "fresh_hd_scans": int((source == "fresh").sum())}, "signals": {}}
    cals = {}
    for name, s in (("v1_film_score", s_v1), ("sup_p_film", s_sup), ("combined", np.stack([s_v1, s_sup], 1))):
        cal = platt(s[fit], y[fit]); p = apply_platt(cal, s[read]); raw = s[read] if s.ndim == 1 else None
        entry = {"auroc_read": round(auroc(y[read], p), 4), "auroc_ci95_over_groups": boot_auroc(y[read], p, groups[read]), "brier_read": round(float(np.mean((p - y[read]) ** 2)), 4),
                 "ece_read_after_platt": ece(y[read], p), "mean_p_film_read": {"film": round(float(p[y[read] == 1].mean()), 3), "digital": round(float(p[y[read] == 0].mean()), 3)},
                 "auroc_read_test_originals_only": round(auroc(y[read & (source == "test")], p[(source == "test")[read]]), 4) if (read & (source == "test")).sum() else None}
        if raw is not None:
            entry["ece_read_raw"] = ece(y[read], np.clip(raw, 0, 1)); entry["auroc_read_raw"] = round(auroc(y[read], raw), 4)
        rep["signals"][name] = entry; cals[name] = cal
        print(name, entry, flush=True)
    # robustness on the READ half, combined signal
    cal = cals["sup_p_film"]; base = apply_platt(cal, s_sup[read]); rob = {}
    read_idx = np.where(read)[0]
    for kind in ("jpeg50", "rescale0.5"):
        pert = np.array([signals(as_buffer(perturb(load_image(ims[i], max_side=1280), kind)))[1] for i in read_idx]); p2 = apply_platt(cal, pert)
        rob[kind] = {"auroc": round(auroc(y[read], p2), 4), "flip_rate_at_0.5": round(float(((p2 >= 0.5) != (base >= 0.5)).mean()), 4), "mean_abs_shift_pct_points": round(float(np.abs(p2 - base).mean() * 100), 2),
                     "p95_abs_shift_pct_points": round(float(np.percentile(np.abs(p2 - base), 95) * 100), 2)}
        print("robustness", kind, rob[kind], flush=True)
    rep["robustness_sup_read"] = rob
    # ---- rendering-family prototypes: fit on AIC analysis (spent), evaluate on Met analysis
    aic = [r for r in read_jsonl(P.MANIFESTS / "aic_photographs_manifest.jsonl") if r["split"] == "analysis"]
    met = [r for r in read_jsonl(P.MANIFESTS / "met_photographs_manifest.jsonl") if r["split"] == "analysis" and min(r["image_width"], r["image_height"]) >= 256]

    def feats(rows_):
        F, L = [], []
        for r in rows_:
            try:
                F.append(feature_vector(compute_features(load_image(r["path"], max_side=1280), 512))); L.append(process_family(r.get("process")))
            except Exception:  # noqa: BLE001 - unreadable rows are skipped and counted
                continue
        return np.array(F), np.array(L)
    Fa, La = feats(aic); Fm, Lm = feats(met)
    mu, sd = Fa.mean(0), Fa.std(0) + 1e-6
    families = sorted(c for c in set(La) if (La == c).sum() >= MIN_FAMILY)
    cents = np.stack([((Fa[La == c] - mu) / sd).mean(0) for c in families]); cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    Zm = (Fm - mu) / sd; Zm /= np.linalg.norm(Zm, axis=1, keepdims=True) + 1e-9
    sims = Zm @ cents.T; pred = np.array(families)[sims.argmax(1)]; margin = np.sort(sims, 1)[:, -1] - np.sort(sims, 1)[:, -2]
    keep = np.isin(Lm, families); acc = {c: float((pred[keep & (Lm == c)] == c).mean()) for c in families if (keep & (Lm == c)).sum()}
    proto = {"families": families, "n_fit_per_family": {c: int((La == c).sum()) for c in families}, "n_eval": int(keep.sum()), "per_family_accuracy_met": {c: round(v, 4) for c, v in acc.items()},
             "balanced_accuracy_met": round(float(np.mean(list(acc.values()))), 4), "chance_balanced": round(1 / len(families), 4), "accuracy_met": round(float((pred[keep] == Lm[keep]).mean()), 4),
             "margin_quantiles": {q: round(float(np.percentile(margin[keep], q)), 4) for q in (10, 50, 90)},
             "accuracy_when_margin_above_median": round(float((pred[keep & (margin > np.median(margin[keep]))] == Lm[keep & (margin > np.median(margin[keep]))]).mean()), 4)}
    rep["prototypes"] = proto; print("prototypes", proto, flush=True)
    # ---- artifacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "prototypes.npz", centroids=cents.astype(np.float32), mean=mu.astype(np.float32), std=sd.astype(np.float32), families=np.array(families))
    # The shipped film-likeness calibration uses the single best signal on the READ half (the student's capture head
    # alone on 2026-11-08: the v1 rendering score reads at chance against real film and only adds noise when combined).
    best_name = max(("v1_film_score", "sup_p_film", "combined"), key=lambda n: rep["signals"][n]["auroc_read"])
    inputs = {"v1_film_score": ["v1_film_score"], "sup_p_film": ["sup_p_film"], "combined": ["v1_film_score", "sup_p_film"]}[best_name]
    (OUT_DIR / "calibration.json").write_text(json.dumps({"signal": best_name, "inputs": inputs, "platt": cals[best_name], "fit_set": "E58 FIT half (test originals + fresh hd scans, grouped by uploader)",
                                                          "read_auroc": rep["signals"][best_name]["auroc_read"], "read_ci95": rep["signals"][best_name]["auroc_ci95_over_groups"], "read_ece": rep["signals"][best_name]["ece_read_after_platt"],
                                                          "read_n": {"film": rep["n_read_film"], "digital": rep["n_read_digital"], "groups": rep["n_read_groups"]}, "all_signals": {n: {k: rep["signals"][n][k] for k in ("auroc_read", "auroc_ci95_over_groups", "ece_read_after_platt")} for n in rep["signals"]}}, indent=1))
    np.savez(OUT_DIR / "e58_scores.npz", y=y, groups=groups, source=source, fit=fit, s_v1=s_v1, s_sup=s_sup)
    rep["shipped_signal"] = best_name
    meta = {"artifact_version": "retrieval-mvp-2026-11-08", "v1_model": hashlib.sha256((V1 / "model.onnx").read_bytes()).hexdigest()[:16], "sup_model": hashlib.sha256((SUP / "model.onnx").read_bytes()).hexdigest()[:16],
            "prototype_fit_set": "aic_photographs analysis half (spent)", "prototype_eval": {"set": "met_photographs analysis half", "balanced_accuracy": proto["balanced_accuracy_met"], "chance": proto["chance_balanced"]},
            "film_likeness": {"definition": "calibrated probability that the rendering resembles real film scans more than digital captures (weak EXIF/owner provenance labels); a resemblance estimate, not proof of film",
                              "signal": best_name, "auroc_read": rep["signals"][best_name]["auroc_read"], "ci95": rep["signals"][best_name]["auroc_ci95_over_groups"], "ece": rep["signals"][best_name]["ece_read_after_platt"]},
            "rendering_family_prototypes": {"status": "experimental, off by default", "reason": f"fit on AIC, read on Met: balanced accuracy {proto['balanced_accuracy_met']} against chance {proto['chance_balanced']} (museum digitisation pipelines dominate across collections)"}}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    rep["seconds"] = round(time.time() - t0, 1); (P.REPORTS / "e58_filmlikeness_calibration.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"wrote artifacts to {OUT_DIR} and the report in {rep['seconds']} s")


if __name__ == "__main__":
    main()
