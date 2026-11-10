"""Real-film validation: do preset classifiers trained on synthetic renders recognise the real stock?

    python -m research.lut.real_eval build                       # manifest + features (colour, DINOv2) for the real set
    python -m research.lut.real_eval eval --models colour_mlp,dinov2_colour_mlp,stats24_lr

Real images (all evaluation-only, licences in configs/datasets.yaml): commoncatalog_film and commoncatalog_film_hd
colour-negative scans whose uploader stock tag maps to a preset, artefact_film (owner-labelled emulsions), and as
unknown styles: colour-negative stocks with no preset (Superia, Natura, Vista, ProImage, Vision3, Reala, Pro 400H,
Pro 160, Konica, CineStill 800T), colour-reversal scans, and the unlabelled film scans of the training manifest.
Labels are uploader claims (class B) except ARTeFACT (class A). A row counts as "exact" when its stock is one of the
13 preset stocks; "line" when only the line is known (Portra -> {Portra 160, Portra 400}; Gold -> Gold 200; ...).
Family labels (for comparison with film_physics/evaluation): warm consumer = Gold 200 / Ultramax 400 / ColorPlus 200,
soft = Portra 160 / 400, vivid = Ektar 100, cool = Fuji 200, lomo = Lomography 400, cine = CineStill 50D.
Every metric is bootstrapped over uploaders; chance = mean acceptable-set size / number of classes; a majority-label
baseline is reported next to each number. Comparator: logistic regression trained on the real scans themselves with
leave-one-uploader-out (the 2026-11-09 recipe), on the same label set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.lut.evaluate import auroc, confusion, format_confusion, grouped_bootstrap, topk_hits_sets
from research.lut.features import COLOUR_SLICES, extract
from research.lut.train import NumpyClassifier, select

EXACT = {"kodak_portra_160": "Kodak_Portra_160", "kodak_portra_400": "Kodak_Portra_400", "kodak_ektar_100": "Kodak_Ektar_100",
         "kodak_gold_200": "Kodak_Gold_200", "kodak_ultramax_400": "Kodak_Ultramax_400", "kodak_colorplus_200": "Kodak_Color_Plus_200",
         "fuji_c200": "Fujifilm_200", "lomography_colour_negative_400": "Lomography_Color_400", "cinestill_50d": "Cinestill_50D"}
LINE = {"kodak_portra": {"Kodak_Portra_160", "Kodak_Portra_400"}, "kodak_ektar": {"Kodak_Ektar_100"}, "kodak_gold": {"Kodak_Gold_200"},
        "kodak_ultramax": {"Kodak_Ultramax_400"}, "kodak_colorplus": {"Kodak_Color_Plus_200"}, "fuji_c200": {"Fujifilm_200"},
        "lomography_colour_negative": {"Lomography_Color_400"}, "cinestill_50d": {"Cinestill_50D"}}
LINE_NAME = {"Kodak_Portra_160": "portra", "Kodak_Portra_400": "portra", "Kodak_Ektar_100": "ektar", "Kodak_Gold_200": "gold",
             "Kodak_Ultramax_400": "ultramax", "Kodak_Color_Plus_200": "colorplus", "Fujifilm_200": "fuji200",
             "Lomography_Color_400": "lomo400", "Cinestill_50D": "cinestill50d"}
FAMILY = {"Kodak_Gold_200": "warm", "Kodak_Ultramax_400": "warm", "Kodak_Color_Plus_200": "warm", "Kodak_Portra_160": "soft",
          "Kodak_Portra_400": "soft", "Kodak_Ektar_100": "vivid", "Fujifilm_200": "cool", "Lomography_Color_400": "lomo", "Cinestill_50D": "cine"}
REAL_DIR = Path("data/lut/real")


def build_manifest() -> list[dict]:
    rows = []
    for ds in ("commoncatalog_film", "commoncatalog_film_hd", "artefact_film"):
        for l in open(f"data/manifests/{ds}_manifest.jsonl"):
            r = json.loads(l)
            if r.get("kind") == "film_restored":
                continue
            ftype = r.get("film_type")
            stock = (r.get("film_stock") or "").lower()
            line = (r.get("film_stock_line") or "").lower()
            if not line and stock:
                line = next((k for k in LINE if stock.startswith(k)), "")
            accept = set()
            exact = None
            if stock in EXACT:
                exact = EXACT[stock]
                accept = {exact}
            elif line in LINE:
                accept = set(LINE[line])
            if ftype in ("colour_negative", "colour_negative_cine", "colour_reversal"):
                kind = "labelled" if accept else "unknown_real_stock"
            else:
                continue
            rows.append({"path": r["path"], "dataset": ds, "uploader": r.get("uploader_id") or r.get("uploader") or ds, "stock": stock,
                         "line": line, "film_type": ftype, "exact": exact, "accept": sorted(accept), "kind": kind,
                         "authenticity_class": r.get("authenticity_class"), "width": r.get("image_width"), "height": r.get("image_height")})
    for l in open("data/manifests/raw_commoncatalog.jsonl"):
        r = json.loads(l)
        if r.get("medium") == "film_scan" and not r.get("monochrome_tag"):
            rows.append({"path": r["path"], "dataset": "commoncatalog_train_filmscan", "uploader": r["group_id"], "stock": "", "line": "",
                         "film_type": "unknown", "exact": None, "accept": [], "kind": "unknown_real_unlabelled", "authenticity_class": "B"})
    return rows


def cmd_build(a) -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    rows = [r for r in rows if Path(r["path"]).exists()]
    with open(REAL_DIR / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(len(rows), "rows", kinds)
    lab = [r for r in rows if r["kind"] == "labelled"]
    acc = {}
    for r in lab:
        k = "/".join(LINE_NAME[x] for x in sorted(set(r["accept"])))
        acc[k] = acc.get(k, 0) + 1
    print("labelled by line:", acc, "exact:", sum(1 for r in lab if r["exact"]), "uploaders:", len({r["uploader"] for r in lab}))
    feats = extract([r["path"] for r in rows], dinov2=True, workers=a.workers)
    np.savez(REAL_DIR / "features.npz", paths=np.array([r["path"] for r in rows]), **feats)
    print("wrote", REAL_DIR / "features.npz", {k: v.shape for k, v in feats.items()})


def line_probs(P: np.ndarray, classes: list[str]) -> tuple[np.ndarray, list[str]]:
    lines = sorted(set(LINE_NAME.values()))
    Q = np.zeros((len(P), len(lines)))
    for i, c in enumerate(classes):
        if c in LINE_NAME:
            Q[:, lines.index(LINE_NAME[c])] += P[:, i]
    return Q, lines


def family_probs(P: np.ndarray, classes: list[str]) -> tuple[np.ndarray, list[str]]:
    fams = sorted(set(FAMILY.values()))
    Q = np.zeros((len(P), len(fams)))
    for i, c in enumerate(classes):
        if c in FAMILY:
            Q[:, fams.index(FAMILY[c])] += P[:, i]
    return Q, fams


def evaluate_model(name: str, P: np.ndarray, classes: list[str], rows: list[dict], P_unknown: np.ndarray, rows_unknown: list[dict]) -> dict:
    cidx = {c: i for i, c in enumerate(classes)}
    accept = [{cidx[c] for c in r["accept"] if c in cidx} for r in rows]
    keep = np.array([len(s) > 0 for s in accept])
    rows = [r for r, k in zip(rows, keep) if k]
    accept = [s for s, k in zip(accept, keep) if k]
    P = P[keep]
    up = np.array([r["uploader"] for r in rows])
    res = {"name": name, "n": len(rows), "n_uploaders": len(np.unique(up)), "classes": classes}
    # line-level hits over the 14-way prediction (restricted to the 9 stock classes that exist in the real set)
    h1 = topk_hits_sets(P, accept, 1)
    h3 = topk_hits_sets(P, accept, 3)
    res["line_top1"] = grouped_bootstrap(h1, up)
    res["line_top3"] = grouped_bootstrap(h3, up)
    res["chance_line_top1"] = float(np.mean([len(s) for s in accept]) / len(classes))
    # majority baseline: always predict the most common acceptable class
    counts = np.zeros(len(classes))
    for s in accept:
        for c in s:
            counts[c] += 1 / len(s)
    maj = int(counts.argmax())
    res["majority_class"] = classes[maj]
    res["majority_line_top1"] = float(np.mean([maj in s for s in accept]))
    # exact stock, where known
    ex = [(i, cidx[r["exact"]]) for i, r in enumerate(rows) if r["exact"] in cidx]
    if ex:
        idx, ye = zip(*ex)
        idx, ye = np.array(idx), np.array(ye)
        hits = P[idx].argmax(1) == ye
        res["exact_n"] = int(len(idx))
        res["exact_top1"] = grouped_bootstrap(hits, up[idx])
        res["exact_chance"] = 1 / len(classes)
    # line-level with merged probabilities (Portra 160+400 etc.) and confusion by line
    Q, lines = line_probs(P, classes)
    yl = np.array([lines.index(LINE_NAME[sorted(r["accept"])[0]]) for r in rows])
    pred_l = Q.argmax(1)
    res["line_merged_top1"] = grouped_bootstrap(pred_l == yl, up)
    res["line_merged_chance"] = 1 / len(lines)
    res["line_merged_majority"] = float(np.bincount(yl, minlength=len(lines)).max() / len(yl))
    res["line_confusion_text"] = format_confusion(confusion(yl, pred_l, len(lines)), lines, width=8)
    res["line_support"] = {l: int((yl == i).sum()) for i, l in enumerate(lines)}
    res["line_per_class_recall"] = {l: float((pred_l[yl == i] == i).mean()) if (yl == i).any() else None for i, l in enumerate(lines)}
    # family
    F, fams = family_probs(P, classes)
    yf = np.array([fams.index(FAMILY[sorted(r["accept"])[0]]) for r in rows])
    pred_f = F.argmax(1)
    res["family_top1"] = grouped_bootstrap(pred_f == yf, up)
    res["family_chance"] = 1 / len(fams)
    res["family_majority"] = float(np.bincount(yf, minlength=len(fams)).max() / len(yf))
    res["family_confusion_text"] = format_confusion(confusion(yf, pred_f, len(fams)), fams, width=6)
    # predicted-class histogram and 'none' rate on real film
    pred = P.argmax(1)
    res["pred_hist"] = {c: int((pred == i).sum()) for i, c in enumerate(classes)}
    if "none" in cidx:
        res["predicted_none_fraction"] = float((pred == cidx["none"]).mean())
    # rejection: known-stock scans vs unknown-stock real scans by max-softmax
    if len(rows_unknown):
        res["reject_auroc_known_vs_unknown_real"] = auroc(P.max(1), P_unknown.max(1))
        res["unknown_real_predicted_none_fraction"] = float((P_unknown.argmax(1) == cidx["none"]).mean()) if "none" in cidx else None
        res["unknown_real_pred_hist"] = {c: int((P_unknown.argmax(1) == i).sum()) for i, c in enumerate(classes)}
        c800 = np.array([r["stock"] == "cinestill_800t" for r in rows_unknown])
        if c800.any():
            res["cinestill_800t_pred_hist"] = {c: int((P_unknown[c800].argmax(1) == i).sum()) for i, c in enumerate(classes)}
    return res


def logo_real_comparator(X: np.ndarray, rows: list[dict], label: str) -> dict:
    """Logistic regression trained on the real scans with leave-one-uploader-out (the film_physics recipe)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y_names = [LINE_NAME[sorted(r["accept"])[0]] for r in rows]
    lines = sorted(set(y_names))
    y = np.array([lines.index(v) for v in y_names])
    up = np.array([r["uploader"] for r in rows])
    pred = np.zeros_like(y)
    for u in np.unique(up):
        tr, te = up != u, up == u
        if len(np.unique(y[tr])) < 2:
            pred[te] = np.bincount(y[tr]).argmax()
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    acc = grouped_bootstrap(pred == y, up)
    fams = sorted(set(FAMILY.values()))
    fam_of_line = {LINE_NAME[k]: v for k, v in FAMILY.items()}
    yf = np.array([fams.index(fam_of_line[lines[v]]) for v in y])
    pf = np.array([fams.index(fam_of_line[lines[v]]) for v in pred])
    return {"features": label, "n": int(len(y)), "line_acc": acc, "line_majority": float(np.bincount(y).max() / len(y)),
            "family_acc": grouped_bootstrap(pf == yf, up), "family_majority": float(np.bincount(yf).max() / len(yf)),
            "line_confusion_text": format_confusion(confusion(y, pred, len(lines)), lines, width=8)}


def cmd_eval(a) -> None:
    rows = [json.loads(l) for l in open(REAL_DIR / "manifest.jsonl")]
    z = np.load(REAL_DIR / "features.npz", allow_pickle=True)
    assert list(z["paths"]) == [r["path"] for r in rows]
    lab = np.array([r["kind"] == "labelled" for r in rows])
    unk = np.array([r["kind"] != "labelled" for r in rows])
    out = {"n_labelled": int(lab.sum()), "n_unknown": int(unk.sum()), "models": {}, "comparators": {}}
    for name in a.models.split(","):
        clf = NumpyClassifier.load(f"data/lut/models/{name}.npz")
        X = select(z, str(clf.d["feature_set"]))
        P = clf.predict_proba(X)
        r = evaluate_model(name, P[lab], clf.classes, [x for x, k in zip(rows, lab) if k], P[unk], [x for x, k in zip(rows, unk) if k])
        out["models"][name] = r
        print(f"[{name}] n={r['n']} ({r['n_uploaders']} uploaders) line top1 {r['line_top1'][0]:.3f} [{r['line_top1'][1]:.3f}, {r['line_top1'][2]:.3f}] "
              f"(chance {r['chance_line_top1']:.3f}, majority {r['majority_line_top1']:.3f}) top3 {r['line_top3'][0]:.3f} | merged-line top1 "
              f"{r['line_merged_top1'][0]:.3f} (maj {r['line_merged_majority']:.3f}) | family {r['family_top1'][0]:.3f} [{r['family_top1'][1]:.3f}, {r['family_top1'][2]:.3f}] "
              f"(maj {r['family_majority']:.3f}) | exact top1 {r.get('exact_top1', [float('nan')])[0]:.3f} (n={r.get('exact_n', 0)}) | none-rate {r.get('predicted_none_fraction', float('nan')):.2f} "
              f"| reject AUROC {r.get('reject_auroc_known_vs_unknown_real', float('nan')):.3f}")
        print(r["line_confusion_text"])
    lab_rows = [x for x, k in zip(rows, lab) if k]
    for label, X in (("stats24", z["colour"][:, COLOUR_SLICES["stats24"]]), ("colour312", z["colour"]), ("dinov2", z["dinov2"]),
                     ("dinov2+colour", np.concatenate([z["dinov2"], z["colour"]], 1))):
        c = logo_real_comparator(np.nan_to_num(X[lab]), lab_rows, label)
        out["comparators"][label] = c
        print(f"[LOGO real {label}] line acc {c['line_acc'][0]:.3f} [{c['line_acc'][1]:.3f}, {c['line_acc'][2]:.3f}] (majority {c['line_majority']:.3f}) | "
              f"family acc {c['family_acc'][0]:.3f} (majority {c['family_majority']:.3f})")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--models", required=True)
    e.add_argument("--out", default="reports/lut/real_eval.json")
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_eval(a)


if __name__ == "__main__":
    main()
