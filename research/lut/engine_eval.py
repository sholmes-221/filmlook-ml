"""Compare the engine's matching approaches (A, B, C, CW, D, E) on held-out data and pick the default.

    python -m research.lut.engine_eval --n 700 --out reports/lut/engine_eval.json [--write-config]

Synthetic: held-out test scenes (never used as reference scenes), 14 candidates (13 styles + none); top-1 / top-3 with
scene bootstrap, plus the untouched filtered uploads as unknown-style negatives and the margin / distance statistics
used for the "uncertain" rule. Real: the labelled real-film set of real_eval.py (line-level acceptance, uploader
bootstrap, majority baseline) and the unknown-stock scans (rejection AUROC with the negative best distance).
--write-config stores the chosen method and the uncertainty thresholds in research/lut/db/engine_config.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.lut.evaluate import auroc, grouped_bootstrap, topk_hits, topk_hits_sets
from research.lut.real_eval import LINE_NAME, REAL_DIR

_E = None


def _init(db_dir):
    global _E
    from research.lut.engine import FilmStyleEngine
    _E = FilmStyleEngine(db_dir)


def _dist(path):
    from research.lut.engine.descriptors import describe
    from research.lut.engine.engine import load_image
    from research.lut.engine.matching import distances
    img = load_image(path)
    d = distances(_E.db, img, describe(img))
    return {k: v.tolist() for k, v in d.items()}


def run_set(paths, db_dir, workers):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(db_dir,)) as ex:
        return list(ex.map(_dist, paths, chunksize=4))


def margin_stats(D):
    s = np.sort(D, axis=1)
    scale = np.maximum(np.median(D, axis=1) - s[:, 0], 1e-6)
    return (s[:, 1] - s[:, 0]) / scale, s[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="research/lut/db")
    ap.add_argument("--manifest", default="data/lut/synth/manifest.jsonl")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="reports/lut/engine_eval.json")
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args()
    from research.lut.engine.lut_db import LutDB
    db = LutDB(a.db)
    ids = db.ids
    id_of = {s["id"]: s["id"] for s in db.styles}
    id_of["none"] = "none"
    for k, v in list(LINE_NAME.items()):          # preset names used by real_eval -> style ids
        id_of[k] = k.replace("'", "").lower()
    rows = [json.loads(l) for l in open(a.manifest)]
    test = [r for r in rows if r["split"] == "test"]
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.choice(len(test), min(a.n, len(test)), replace=False)]
    neg = [r for r in rows if r["split"] == "negatives"]
    print(f"synthetic test {len(test)} images, negatives {len(neg)}")
    res_t = run_set([r["path"] for r in test], a.db, a.workers)
    res_n = run_set([r["path"] for r in neg], a.db, a.workers)
    y = np.array([ids.index(r["preset"].replace("'", "").lower()) for r in test])
    scenes = np.array([r["scene_id"] for r in test])
    out = {"n_synth_test": len(test), "n_negatives": len(neg), "candidates": ids, "chance": 1 / len(ids), "synthetic": {}, "real": {}}
    methods = list(res_t[0].keys())
    for m in methods:
        D = np.array([r[m] for r in res_t])
        S = -D
        h1, h3 = topk_hits(S, y, 1), topk_hits(S, y, 3)
        b1, b3 = grouped_bootstrap(h1, scenes), grouped_bootstrap(h3, scenes)
        marg, best = margin_stats(D)
        Dn = np.array([r[m] for r in res_n])
        margn, bestn = margin_stats(Dn)
        pred = D.argmin(1)
        out["synthetic"][m] = {"top1": b1[0], "top1_ci": b1[1:], "top3": b3[0], "top3_ci": b3[1:],
                               "top1_excluding_none_class": float(h1[y != ids.index("none")].mean()),
                               "pred_hist": {ids[i]: int((pred == i).sum()) for i in range(len(ids))},
                               "per_class_top1": {ids[i]: float(h1[y == i].mean()) for i in range(len(ids)) if (y == i).any()},
                               "margin_median_correct": float(np.median(marg[h1])) if h1.any() else None,
                               "margin_median_wrong": float(np.median(marg[~h1])) if (~h1).any() else None,
                               "margin_auroc_correct_vs_wrong": auroc(marg[h1], marg[~h1]),
                               "best_distance_p95": float(np.percentile(best, 95)),
                               "negatives_predicted_none": float((Dn.argmin(1) == ids.index("none")).mean()),
                               "negatives_reject_auroc_by_best_distance": auroc(bestn, best),
                               "negatives_reject_auroc_by_margin": auroc(marg, margn)}
        print(f"[synthetic {m:2}] top1 {b1[0]:.3f} [{b1[1]:.3f}, {b1[2]:.3f}] top3 {b3[0]:.3f} | margin AUROC correct-vs-wrong {out['synthetic'][m]['margin_auroc_correct_vs_wrong']:.3f} "
              f"| negatives->none {out['synthetic'][m]['negatives_predicted_none']:.2f} reject AUROC dist {out['synthetic'][m]['negatives_reject_auroc_by_best_distance']:.3f}")
    # real set
    rrows = [json.loads(l) for l in open(REAL_DIR / "manifest.jsonl")]
    lab = [r for r in rrows if r["kind"] == "labelled"]
    unk = [r for r in rrows if r["kind"] != "labelled"]
    res_l = run_set([r["path"] for r in lab], a.db, a.workers)
    res_u = run_set([r["path"] for r in unk], a.db, a.workers)
    accept = [{ids.index(id_of[c]) for c in r["accept"]} for r in lab]
    up = np.array([r["uploader"] for r in lab])
    counts = np.zeros(len(ids))
    for s in accept:
        for c in s:
            counts[c] += 1 / len(s)
    maj = int(counts.argmax())
    lines = sorted(set(LINE_NAME.values()))
    line_of_id = {id_of[k]: v for k, v in LINE_NAME.items()}
    yl = np.array([lines.index(LINE_NAME[sorted(r["accept"])[0]]) for r in lab])
    for m in methods:
        D = np.array([r[m] for r in res_l])
        Du = np.array([r[m] for r in res_u])
        S = -D
        h1, h3 = topk_hits_sets(S, accept, 1), topk_hits_sets(S, accept, 3)
        b1, b3 = grouped_bootstrap(h1, up), grouped_bootstrap(h3, up)
        # merged line prediction: best style among those that map to a line
        line_scores = np.full((len(D), len(lines)), np.inf)
        for i, sid in enumerate(ids):
            if sid in line_of_id:
                j = lines.index(line_of_id[sid])
                line_scores[:, j] = np.minimum(line_scores[:, j], D[:, i])
        pl = line_scores.argmin(1)
        bl = grouped_bootstrap(pl == yl, up)
        _, best = margin_stats(D)
        _, bestu = margin_stats(Du)
        out["real"][m] = {"n": len(lab), "n_uploaders": int(len(np.unique(up))), "line_top1": b1[0], "line_top1_ci": b1[1:], "line_top3": b3[0],
                          "chance_line_top1": float(np.mean([len(s) for s in accept]) / len(ids)), "majority_line_top1": float(np.mean([maj in s for s in accept])),
                          "line_merged_top1": bl[0], "line_merged_ci": bl[1:], "line_merged_majority": float(np.bincount(yl).max() / len(yl)),
                          "pred_hist": {ids[i]: int((D.argmin(1) == i).sum()) for i in range(len(ids))},
                          "unknown_reject_auroc_by_best_distance": auroc(bestu, best)}
        print(f"[real {m:2}] line top1 {b1[0]:.3f} [{b1[1]:.3f}, {b1[2]:.3f}] (chance {out['real'][m]['chance_line_top1']:.3f}, majority {out['real'][m]['majority_line_top1']:.3f}) "
              f"top3 {b3[0]:.3f} | merged-line {bl[0]:.3f} (maj {out['real'][m]['line_merged_majority']:.3f}) | unknown reject AUROC {out['real'][m]['unknown_reject_auroc_by_best_distance']:.3f}")
    best_m = max(methods, key=lambda m: out["synthetic"][m]["top1"])
    out["chosen_method"] = best_m
    cfg = {"method": best_m, "margin_min": 0.08, "abs_distance_max": out["synthetic"][best_m]["best_distance_p95"] * 1.25, "min_chroma_p50": 0.04, "none_id": "none",
           "chosen_by": "highest synthetic held-out top-1 (engine_eval.py)", "synthetic_top1": out["synthetic"][best_m]["top1"], "real_line_top1": out["real"][best_m]["line_top1"]}
    out["engine_config"] = cfg
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    if a.write_config:
        (Path(a.db) / "engine_config.json").write_text(json.dumps(cfg, indent=1))
        print("wrote", Path(a.db) / "engine_config.json")
    print("chosen method:", best_m, "| wrote", a.out)


if __name__ == "__main__":
    main()
