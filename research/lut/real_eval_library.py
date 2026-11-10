"""Real-film evaluation of stock-level classifiers trained on the LUT-library synthetic set (phase 7).

    python -m research.lut.real_eval_library --models lib_stock_colour_mlp,lib_stock_colour_lr [--features data/lut/real/features.npz]

Classes of a stock-level model are library stock ids (kodak_portra_400, fuji_superia_400, ...) plus `none`; a LUT-level
model (classes = lut ids) is mapped to stocks by summing probabilities. Real scans (data/lut/real/manifest.jsonl, the
same 1,146 scans as real_eval.py) are labelled at three levels when the library covers the line:
  line    portra (Portra 160/400/800), ektar, gold, ultramax, colorplus, superia (Superia 100-1600), pro400h, vista
          (Agfa Vista), cinestill800t, cinestill50d -- 10 lines; fuji_c200, natura, proimage, reala, pro160, konica,
          lomography, vision3 and the reversal films have no library LUT and are the unknown set
  family  warm consumer (gold, ultramax, colorplus, vista), soft (portra, pro400h), vivid (ektar), cool (superia),
          cine (cinestill)
  exact   where the uploader tag names the specific stock and the library has it (kodak_portra_400, fuji_superia_400, ...)
Metrics: top-1 / top-3 with uploader bootstrap, chance, majority baseline, confusion by line, rejection AUROC of
known-line scans vs unknown-stock scans (max-softmax), and the same leave-one-uploader-out logistic regression trained on
the real scans as comparator. Nothing here claims identification: read the numbers against the majority column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.lut.evaluate import auroc, confusion, format_confusion, grouped_bootstrap, topk_hits
from research.lut.real_eval import REAL_DIR
from research.lut.train import NumpyClassifier, select

STOCK_TO_LINE = {"kodak_portra_160": "portra", "kodak_portra_400": "portra", "kodak_portra_800": "portra", "kodak_ektar_100": "ektar",
                 "kodak_gold_200": "gold", "kodak_ultramax_400": "ultramax", "kodak_colorplus_200": "colorplus",
                 "fuji_superia_100": "superia", "fuji_superia_200": "superia", "fuji_superia_400": "superia", "fuji_superia_800": "superia",
                 "fuji_superia_1600": "superia", "fuji_pro400h": "pro400h", "agfa_vista_200": "vista", "cinestill_800t": "cinestill800t",
                 "cinestill_50d": "cinestill50d"}
REAL_LINE_TO_LINE = {"kodak_portra": "portra", "kodak_ektar": "ektar", "kodak_gold": "gold", "kodak_ultramax": "ultramax", "kodak_colorplus": "colorplus",
                     "fuji_superia": "superia", "fuji_pro400h": "pro400h", "agfa_vista": "vista", "cinestill_800t": "cinestill800t", "cinestill_50d": "cinestill50d"}
FAMILY = {"portra": "soft", "pro400h": "soft", "ektar": "vivid", "gold": "warm", "ultramax": "warm", "colorplus": "warm", "vista": "warm",
          "superia": "cool", "cinestill800t": "cine", "cinestill50d": "cine"}


def real_rows():
    rows = [json.loads(l) for l in open(REAL_DIR / "manifest.jsonl")]
    for r in rows:
        line = r.get("line") or ""
        stock = r.get("stock") or ""
        if not line and stock:
            line = next((k for k in REAL_LINE_TO_LINE if stock.startswith(k)), "")
        if stock.startswith("cinestill_800t"):
            line = "cinestill_800t"
        if stock.startswith("cinestill_50d"):
            line = "cinestill_50d"
        r["eval_line"] = REAL_LINE_TO_LINE.get(line)
        r["eval_exact"] = stock if stock in STOCK_TO_LINE else None
    return rows


def stock_probs(P, classes):
    """Collapse model classes to (stocks, none): LUT-level classes are mapped by their stock prefix in the class name."""
    stocks = sorted({c for c in classes if c in STOCK_TO_LINE} | {"none"})
    if len(stocks) <= 1:   # LUT-level model: classes are library ids like 'rawtherapee_film_simulation__color_kodak_kodak_portra_400_1_-'
        raise ValueError("LUT-level models must be evaluated through their stock mapping; pass a stock-level model")
    Q = np.zeros((len(P), len(stocks)))
    for i, c in enumerate(classes):
        if c in stocks:
            Q[:, stocks.index(c)] += P[:, i]
    return Q, stocks


def collapse(Q, stocks, mapping, keys):
    R = np.zeros((len(Q), len(keys)))
    for i, s in enumerate(stocks):
        k = mapping.get(s)
        if k in keys:
            R[:, keys.index(k)] += Q[:, i]
    return R


def evaluate(name, P, classes, rows, lab_mask, unk_mask):
    Q, stocks = stock_probs(P, classes)
    lines = sorted(set(STOCK_TO_LINE.values()))
    fams = sorted(set(FAMILY.values()))
    L = collapse(Q, stocks, STOCK_TO_LINE, lines)
    F = collapse(L, lines, FAMILY, fams)
    lab = [r for r, k in zip(rows, lab_mask) if k]
    up = np.array([r["uploader"] for r in lab])
    yl = np.array([lines.index(r["eval_line"]) for r in lab])
    yf = np.array([fams.index(FAMILY[r["eval_line"]]) for r in lab])
    Ll, Fl = L[lab_mask], F[lab_mask]
    res = {"name": name, "n": len(lab), "n_uploaders": int(len(np.unique(up))), "lines": lines, "line_support": {l: int((yl == i).sum()) for i, l in enumerate(lines)}}
    h1, h3 = topk_hits(Ll, yl, 1), topk_hits(Ll, yl, 3)
    res["line_top1"], res["line_top3"] = grouped_bootstrap(h1, up), grouped_bootstrap(h3, up)
    res["line_chance"] = 1 / len(lines)
    res["line_majority"] = float(np.bincount(yl).max() / len(yl))
    res["line_confusion_text"] = format_confusion(confusion(yl, Ll.argmax(1), len(lines)), lines, width=9)
    res["line_per_class_recall"] = {l: float((Ll.argmax(1)[yl == i] == i).mean()) if (yl == i).any() else None for i, l in enumerate(lines)}
    res["family_top1"] = grouped_bootstrap(Fl.argmax(1) == yf, up)
    res["family_majority"] = float(np.bincount(yf).max() / len(yf))
    res["family_confusion_text"] = format_confusion(confusion(yf, Fl.argmax(1), len(fams)), fams, width=6)
    ex = [(i, stocks.index(r["eval_exact"])) for i, r in enumerate(lab) if r["eval_exact"] in stocks]
    if ex:
        idx, ye = map(np.array, zip(*ex))
        res["exact_n"] = int(len(idx))
        res["exact_top1"] = grouped_bootstrap(Q[lab_mask][idx].argmax(1) == ye, up[idx])
        res["exact_chance"] = 1 / len(stocks)
        res["exact_majority"] = float(np.bincount(ye).max() / len(ye))
    pred_stock = Q[lab_mask].argmax(1)
    res["pred_stock_hist"] = {stocks[i]: int((pred_stock == i).sum()) for i in range(len(stocks)) if (pred_stock == i).any()}
    res["predicted_none_fraction"] = float((pred_stock == stocks.index("none")).mean())
    if unk_mask.any():
        res["n_unknown"] = int(unk_mask.sum())
        res["reject_auroc_known_vs_unknown"] = auroc(P[lab_mask].max(1), P[unk_mask].max(1))
        res["unknown_predicted_none_fraction"] = float((Q[unk_mask].argmax(1) == stocks.index("none")).mean())
    return res


def logo_comparator(X, rows, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    lines = sorted(set(STOCK_TO_LINE.values()))
    y = np.array([lines.index(r["eval_line"]) for r in rows])
    up = np.array([r["uploader"] for r in rows])
    pred = np.zeros_like(y)
    for u in np.unique(up):
        tr, te = up != u, up == u
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    fams = sorted(set(FAMILY.values()))
    yf = np.array([fams.index(FAMILY[lines[v]]) for v in y])
    pf = np.array([fams.index(FAMILY[lines[v]]) for v in pred])
    return {"features": label, "n": int(len(y)), "line_acc": grouped_bootstrap(pred == y, up), "line_majority": float(np.bincount(y).max() / len(y)),
            "family_acc": grouped_bootstrap(pf == yf, up), "family_majority": float(np.bincount(yf).max() / len(yf))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--features", default=str(REAL_DIR / "features.npz"))
    ap.add_argument("--out", default="reports/lut/real_eval_library.json")
    a = ap.parse_args()
    rows = real_rows()
    z = np.load(a.features, allow_pickle=True)
    assert list(z["paths"]) == [r["path"] for r in rows]
    lab = np.array([r["eval_line"] is not None for r in rows])
    unk = np.array([r["eval_line"] is None for r in rows])
    out = {"n_labelled": int(lab.sum()), "n_unknown": int(unk.sum()), "models": {}, "comparators": {}}
    print(f"real scans with a library-covered line: {lab.sum()} ({len({r['uploader'] for r, k in zip(rows, lab) if k})} uploaders); unknown {unk.sum()}")
    for name in a.models.split(","):
        clf = NumpyClassifier.load(f"data/lut/models/{name}.npz")
        P = clf.predict_proba(select(z, str(clf.d["feature_set"])))
        r = evaluate(name, P, clf.classes, rows, lab, unk)
        out["models"][name] = r
        print(f"[{name}] n={r['n']} ({r['n_uploaders']} uploaders) line top1 {r['line_top1'][0]:.3f} [{r['line_top1'][1]:.3f}, {r['line_top1'][2]:.3f}] "
              f"(chance {r['line_chance']:.3f}, majority {r['line_majority']:.3f}) top3 {r['line_top3'][0]:.3f} | family {r['family_top1'][0]:.3f} "
              f"[{r['family_top1'][1]:.3f}, {r['family_top1'][2]:.3f}] (maj {r['family_majority']:.3f}) | exact {r.get('exact_top1', [float('nan')])[0]:.3f} "
              f"(n={r.get('exact_n', 0)}, maj {r.get('exact_majority', float('nan')):.3f}) | none-rate {r['predicted_none_fraction']:.2f} | reject AUROC {r.get('reject_auroc_known_vs_unknown', float('nan')):.3f}")
        print(r["line_confusion_text"])
    lab_rows = [r for r, k in zip(rows, lab) if k]
    from research.lut.features import COLOUR_SLICES
    for label, X in (("stats24", z["colour"][:, COLOUR_SLICES["stats24"]]), ("colour312", z["colour"]), ("dinov2", z["dinov2"])):
        c = logo_comparator(np.nan_to_num(X[lab]), lab_rows, label)
        out["comparators"][label] = c
        print(f"[LOGO real {label}] line acc {c['line_acc'][0]:.3f} [{c['line_acc'][1]:.3f}, {c['line_acc'][2]:.3f}] (majority {c['line_majority']:.3f}) | "
              f"family {c['family_acc'][0]:.3f} (majority {c['family_majority']:.3f})")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
