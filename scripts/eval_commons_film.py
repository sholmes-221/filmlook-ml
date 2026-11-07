#!/usr/bin/env python
"""Real-film evaluation on the Commons film set (second real source; evaluation only).

  python scripts/eval_commons_film.py --features            # one GPU pass -> data/embeddings/commons_film_features_v1.{npz,json}
  python scripts/eval_commons_film.py --split analysis      # metrics on the analysis half  -> reports/commons_film_analysis.json
  python scripts/eval_commons_film.py --split test          # the held-out half: run once, for the final table only

Split: uploaders are hashed into two halves (deterministic); every file follows its uploader, so no photographer
crosses. Film-stock labels are uploader categories (asserted). All retrieval metrics exclude the query's own
uploader from the bank and bootstrap over uploaders, because most stocks are dominated by one uploader.
Candidates: raw DINOv2 global (content reference), raw hybrid, hybrid + family SupCon (projection trained on
filmsynth v2 train sources, as in benchmark v1), student_fs_v2 64-d, colour/tone 16, texture 12.
Metrics
  same_stock      p@5 of the query's stock among other uploaders' files (chance = bank prior), uploader CI
  same_type       p@5 of the query's film type (colour negative / reversal / b&w)
  uploader_share  share of top-5 from the query's uploader when it is NOT excluded (photographer bias, vs prior)
  scanner_probe   uploader-grouped CV accuracy of a linear probe for the scanner category (files with one), vs majority
  authenticity    real film scans vs DxO FilmPack digital emulations: EXIF student p(film) AUROC, and grouped-CV
                  linear probes on texture 12 / colour-tone 16 / hybrid (which axis separates them)
"""
import argparse
import os
import hashlib
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

DATASET = "commons_film"; OUT = P.EMBEDDINGS / "commons_film_features_v1"; MANIFEST = P.MANIFESTS / "commons_film_manifest.jsonl"


def configure(dataset: str) -> None:
    """Point the script at another real-film dataset (same manifest fields: path, image_id, uploader, kind, film_stock_claim, film_type, scanner_category)."""
    global DATASET, OUT, MANIFEST
    DATASET, OUT, MANIFEST = dataset, P.EMBEDDINGS / f"{dataset}_features_v1", P.MANIFESTS / f"{dataset}_manifest.jsonl"
N_BOOT = 500


def half_of(uploader):
    return "analysis" if int.from_bytes(hashlib.sha256(("commons:" + str(uploader)).encode()).digest()[:2], "little") % 2 == 0 else "test"


def do_features():
    import onnxruntime as ort
    import torch
    from PIL import Image
    from filmlook_infer.compact import preprocess
    t0 = time.time(); dev = torch.device("cuda")
    rows = read_jsonl(MANIFEST)
    with Pool(16) as pool:
        arrs = pool.map(V3._load, [(r["path"], r["image_id"], True, True) for r in rows], chunksize=8)
    ims = [Image.fromarray(a) for a in arrs]
    explicit = np.stack([X.explicit_of(im) for im in ims]).astype(np.float32)
    dg, ds = X.Dino(dev)(ims)
    _, ds3 = X.Dino(dev, layers=(3,))(ims)                                   # E27: layer-3 mean++std teacher candidate
    feats = {"explicit": explicit, "dino_global": dg.astype(np.float32), "dino_style": ds.astype(np.float32), "dino_style_l3": ds3.astype(np.float32)}
    extra = [("student_" + t, "student_" + t, ["style_embedding"]) for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]
    for name, ck, outs in (("student64", "student_style_fs_v2", ["style_embedding"]), ("student_sup", "student_sup", ["capture_type"]), *extra):
        sess = ort.InferenceSession(str(P.CHECKPOINTS / ck / "model.onnx"), providers=["CPUExecutionProvider"])
        meta = json.loads((P.CHECKPOINTS / ck / "meta.json").read_text()); oi = meta["outputs"].index(outs[0]); acc = []
        for i in range(0, len(ims), 64):
            acc.append(sess.run(None, {"pixel_values": np.concatenate([preprocess(im) for im in ims[i:i + 64]])})[oi])
        a = np.concatenate(acc).astype(np.float32)
        if name == "student_sup":
            e = np.exp(a - a.max(1, keepdims=True)); a = (e / e.sum(1, keepdims=True))[:, meta["heads"]["capture_type"].index("film_scan")]
        feats[name] = a
    np.savez(OUT.with_suffix(".npz"), **feats)
    OUT.with_suffix(".json").write_text(json.dumps({"version": f"{DATASET}-features-v1", "n": len(rows), "rows": rows, "seconds": round(time.time() - t0, 1),
                                                    "student_tags": [t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]}))
    print(json.dumps({"n": len(rows), "seconds": round(time.time() - t0, 1)}))


def boot_groups(values, groups, seed=0):
    values, groups = np.asarray(values, float), np.asarray(groups); ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    b = [values[np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(N_BOOT)]
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def retrieval(E, label, group, exclude_group=True):
    S = E @ E.T; prec, prior, qg = [], [], []
    for i in range(len(E)):
        ok = (group != group[i]) if exclude_group else (np.arange(len(E)) != i)
        rel = label[ok] == label[i]
        if rel.sum() == 0 or (~rel).sum() == 0:
            continue
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:5]
        prec.append((label[order] == label[i]).mean()); prior.append(rel.mean()); qg.append(group[i])
    if not prec:
        return {"precision@5": None, "n": 0}
    return {"precision@5": round(float(np.mean(prec)), 4), "ci95": boot_groups(prec, qg), "chance": round(float(np.mean(prior)), 4), "n": len(prec), "n_groups": int(len(set(qg)))}


def grouped_probe(Z, y, g, min_class=8):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    keep = np.isin(y, [c for c, n in Counter(y).items() if n >= min_class]); Z, y, g = Z[keep], y[keep], g[keep]
    if len(set(y)) < 2 or len(set(g)) < 3:
        return {"accuracy": None, "n": int(len(y))}
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(Z, y, g):
        sc = StandardScaler().fit(Z[tr]); m = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Z[tr]), y[tr]); pred[te] = m.predict(sc.transform(Z[te]))
    hit = (pred == y).astype(float)
    return {"accuracy": round(float(hit.mean()), 4), "ci95": boot_groups(hit, g), "majority": round(float(Counter(y).most_common(1)[0][1] / len(y)), 4), "n": int(len(y)), "classes": len(set(y))}


def grouped_auroc_probe(Z, y, g):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Z, y, g):
        sc = StandardScaler().fit(Z[tr]); m = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Z[tr]), y[tr]); s[te] = m.predict_proba(sc.transform(Z[te]))[:, 1]
    return auroc_ci(y, s, g)


def auroc_ci(y, s, g):
    from sklearn.metrics import roc_auc_score
    ug = np.unique(g); gi = {k: np.where(g == k)[0] for k in ug}; rng = np.random.default_rng(0); b = []
    for _ in range(N_BOOT):
        idx = np.concatenate([gi[k] for k in rng.choice(ug, len(ug))])
        if len(set(y[idx])) == 2:
            b.append(roc_auc_score(y[idx], s[idx]))
    return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], "n": int(len(y))}


def do_eval(split):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = dict(np.load(OUT.with_suffix(".npz"))); rows = json.loads(OUT.with_suffix(".json").read_text())["rows"]
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam_c = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam_c[tr], dev)
    hyb = np.hstack([F["explicit"], F["dino_style"]])
    cands = {"raw_dino_global": F["dino_global"], "raw_hybrid": hyb, "hybrid_supcon_family": proj(hyb), "student_fs_v2_64": F["student64"],
             "colour_tone16": F["explicit"][:, IDX_C], "texture12": F["explicit"][:, IDX_T]}
    layers = P.EMBEDDINGS / "bench_layers_v1.npz"                              # E27 teacher candidate: explicit28 ++ L3 mean/std -> family SupCon
    if "dino_style_l3" in F and layers.exists():
        z = np.load(layers)
        if (z["ids"] == np.array([r["id"] for r in rows_c])).all():
            H3 = np.hstack([Fc["explicit"], z["ms_L3"].astype(np.float32)]); proj3 = X.supcon(H3[tr], fam_c[tr], dev)
            cands["hybrid_l3_supcon_family"] = proj3(np.hstack([F["explicit"], F["dino_style_l3"]]))
    for t in [t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]:
        if "student_" + t in F:
            cands[f"student_{t}_64"] = F["student_" + t]
    up = np.array([r["uploader"] for r in rows]); half = np.array([r["split"] if r.get("split") in ("analysis", "test") else half_of(u) for r, u in zip(rows, up)]); kind = np.array([r["kind"] for r in rows])
    line = np.array([r.get("film_stock_line") or r["film_stock_claim"] or "?" for r in rows])
    stock = np.array([r["film_stock_claim"] or "?" for r in rows]); ftype = np.array([r["film_type"] for r in rows]); scanner = np.array([r["scanner_category"] or "" for r in rows])
    m = half == split; film = m & (kind == "film"); emu = m & (kind == "digital_emulation")
    out = {"dataset": DATASET, "split": split, "n_film": int(film.sum()), "n_emulation": int(emu.sum()), "n_uploaders_film": int(len(set(up[film]))),
           "stocks_top": Counter(stock[film]).most_common(15), "film_types": dict(Counter(ftype[film])),
           "stock_uploader_concentration": {s: round(max(Counter(up[film & (stock == s)]).values()) / (film & (stock == s)).sum(), 2) for s, _ in Counter(stock[film]).most_common(10)},
           "candidates": {}}
    # authenticity: real film vs digital emulation
    y = np.r_[np.ones(film.sum()), np.zeros(emu.sum())].astype(int); idx = np.r_[np.where(film)[0], np.where(emu)[0]]; g = up[idx]
    out["authenticity_film_vs_emulation"] = {"exif_student_p_film": auroc_ci(y, F["student_sup"][idx], g) if emu.sum() else None,
                                             "mean_p_film": {"film": round(float(F["student_sup"][film].mean()), 3), "emulation": round(float(F["student_sup"][emu].mean()), 3) if emu.sum() else None}}
    if emu.sum() >= 20 and len(set(g)) >= 5:
        for nm, Z in (("texture12", F["explicit"][:, IDX_T]), ("colour_tone16", F["explicit"][:, IDX_C]), ("raw_hybrid", hyb), ("dino_global", F["dino_global"])):
            out["authenticity_film_vs_emulation"][f"probe_{nm}"] = grouped_auroc_probe(Z[idx], y, g)
    for name, Z in cands.items():
        E = unit(Z[film].astype(np.float32)); r = {}
        r["same_stock"] = retrieval(E, stock[film], up[film]); r["same_type"] = retrieval(E, ftype[film], up[film])
        r["same_stock_line"] = retrieval(E, line[film], up[film])                 # secondary reading (brand + line, speed dropped); never the decision metric
        r["same_stock_uploader_not_excluded"] = retrieval(E, stock[film], up[film], exclude_group=False)
        own = []; S = E @ E.T
        for i in range(len(E)):
            order = np.argsort(-np.where(np.arange(len(E)) != i, S[i], -np.inf))[:5]; own.append((up[film][order] == up[film][i]).mean())
        r["uploader_share_top5"] = {"share": round(float(np.mean(own)), 4), "prior": round(float(np.mean([(up[film] == u).mean() for u in up[film]])), 4)}
        has = film & (scanner != "")
        r["scanner_probe"] = grouped_probe(Z[has], scanner[has], up[has]) if has.sum() >= 30 else {"accuracy": None, "n": int(has.sum())}
        out["candidates"][name] = r
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / f"{DATASET}_{split}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("split", "n_film", "n_emulation", "n_uploaders_film", "film_types", "stock_uploader_concentration", "authenticity_film_vs_emulation")}, indent=1, default=str))
    print(f"{'candidate':22s} {'stock p@5':>10s} {'CI':>16s} {'chance':>7s} {'type p@5':>9s} {'chance':>7s} {'upl share':>10s} {'prior':>6s} {'scanner acc':>12s} {'maj':>6s}")
    for name, r in out["candidates"].items():
        s, t, u, sc = r["same_stock"], r["same_type"], r["uploader_share_top5"], r["scanner_probe"]
        print(f"{name:22s} {s['precision@5'] or 0:10.3f} {str(s.get('ci95')):>16s} {s.get('chance', 0):7.3f} {t['precision@5'] or 0:9.3f} {t.get('chance', 0):7.3f} {u['share']:10.3f} {u['prior']:6.3f} {sc['accuracy'] or 0:12.3f} {sc.get('majority', 0):6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--features", action="store_true"); ap.add_argument("--split", choices=["analysis", "test"], default=None)
    ap.add_argument("--dataset", default="commons_film")
    a = ap.parse_args()
    configure(a.dataset)
    if a.features:
        do_features()
    if a.split:
        do_eval(a.split)
