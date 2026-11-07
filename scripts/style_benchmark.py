#!/usr/bin/env python
"""Content-invariance benchmark v1 (CPU; see docs/BENCHMARK.md).

  python scripts/style_benchmark.py --split val [--candidates all]     # selection
  python scripts/style_benchmark.py --split test                        # final, once per retained candidate

Candidates are functions from the cached feature arrays to an embedding. Learned projections train on the TRAIN
split only. Extra candidates can be registered with --extra module:function (returns {name: callable(feats)->array}).
"""
import argparse
import importlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

CACHE = P.EMBEDDINGS / "bench_features_v1"
N_BOOT = 300


def load_cache():
    z = np.load(CACHE.with_suffix(".npz")); meta = json.loads(CACHE.with_suffix(".json").read_text())
    F = {k: z[k] for k in z.files}; F["hybrid"] = np.hstack([F["explicit"], F["dino_style"]])
    return F, meta["rows"]


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def boot(values, groups, n=N_BOOT, seed=0):
    values, groups = np.asarray(values, float), np.asarray(groups)
    rng = np.random.default_rng(seed); ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    bs = [values[np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def candidates(F, rows, dev):
    split = np.array([r["split"] for r in rows]); st = np.array([r["set"] for r in rows]); fam = np.array([r["family"] for r in rows])
    tr_fs = (st == "filmsynth") & (split == "train")
    proj = X.supcon(F["hybrid"][tr_fs], fam[tr_fs], dev)
    return {"raw_dino_global": lambda F: F["dino_global"], "raw_hybrid": lambda F: F["hybrid"], "hybrid_supcon_family": lambda F: proj(F["hybrid"]),
            "student_fs_v2_64": lambda F: F["student64"], "colour_tone16": lambda F: F["explicit"][:, IDX_C], "texture12": lambda F: F["explicit"][:, IDX_T],
            "raw_dino_style_L4L8": lambda F: F["dino_style"]}


def evaluate(E, F, rows, split):
    E = unit(E.astype(np.float32)); Z_content = unit(F["dino_global"])
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); src = np.array([r["source"] for r in rows]); fam = np.array([r["family"] for r in rows])
    ids = np.array([r["id"] for r in rows]); idx = {i: k for k, i in enumerate(ids)}
    out = {}
    # ---- A: treatment vs nuisance (filmsynth variants of split sources)
    fsi = np.where((st == "filmsynth") & (sp == split))[0]
    wins, ratios, gsrc = [], [], []
    by_src = {}
    for i in fsi:
        by_src.setdefault(src[i], []).append(i)
    for s, members in by_src.items():
        for i in members:
            n_i = idx[ids[i] + "__nuis"]; d_n = 1 - E[i] @ E[n_i]
            for j in members:
                if j != i:
                    d_t = 1 - E[i] @ E[j]; wins.append(d_t > d_n); ratios.append(d_t / (d_n + 1e-9)); gsrc.append(s)
    out["A_treatment_gt_nuisance"] = {"p": round(float(np.mean(wins)), 4), "ci95": boot(wins, gsrc), "median_ratio": round(float(np.median(ratios)), 3), "n_triples": len(wins)}
    # ---- A2: style gap ratio
    same_src_diff_fam, diff_src_same_fam, g1, g2 = [], [], [], []
    for i in fsi:
        for j in fsi:
            if i < j:
                d = 1 - E[i] @ E[j]
                if src[i] == src[j] and fam[i] != fam[j]:
                    same_src_diff_fam.append(d); g1.append(src[i])
                elif src[i] != src[j] and fam[i] == fam[j]:
                    diff_src_same_fam.append(d); g2.append(src[i])
    out["A2_style_gap_ratio"] = {"ratio": round(float(np.median(same_src_diff_fam) / (np.median(diff_src_same_fam) + 1e-9)), 3),
                                 "d_same_source_diff_family": round(float(np.median(same_src_diff_fam)), 4), "d_diff_source_same_family": round(float(np.median(diff_src_same_fam)), 4)}
    # ---- B: same family across scenes (precision@5, mAP), per generator set
    B = {}
    for setname in ("filmsynth", "sdxl", "zimage"):
        qi = np.where((st == setname) & (sp == split))[0]
        if setname != "filmsynth":
            qi = qi[np.array([rows[i]["treated"] for i in qi])]
        S = E[qi] @ E[qi].T
        prec, aps, prior, gq = [], [], [], []
        for a, i in enumerate(qi):
            other = np.array([b for b, j in enumerate(qi) if src[j] != src[i]])
            if len(other) == 0:
                continue
            order = other[np.argsort(-S[a, other])]
            rel = (fam[qi[order]] == fam[i]).astype(float)
            if rel.sum() == 0:
                continue
            prec.append(rel[:5].mean()); prior.append(rel.mean()); gq.append(src[i])
            hits = np.cumsum(rel); aps.append(float(np.mean([hits[k] / (k + 1) for k in range(len(rel)) if rel[k]])))
        B[setname] = {"precision@5": round(float(np.mean(prec)), 4), "ci95": boot(prec, gq), "mAP": round(float(np.mean(aps)), 4), "chance": round(float(np.mean(prior)), 4), "n_queries": len(prec)}
    out["B_same_treatment_across_scenes"] = B
    # ---- C: generator fingerprint
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    gset = np.isin(st, ["filmsynth", "sdxl", "zimage"])
    glab = np.where(np.array([r.get("treated", False) for r in rows]), st, "control")
    tr, te = gset & (sp == "train"), gset & (sp == split)
    sc = StandardScaler().fit(E[tr]); m = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(E[tr]), glab[tr])
    pred = m.predict(sc.transform(E[te])); from collections import Counter
    acc = (pred == glab[te]).mean(); chance = max(Counter(glab[te]).values()) / te.sum()
    ti = np.where(te & (glab != "control"))[0]
    S = E[ti] @ E[ti].T; share, gq = [], []
    for a, i in enumerate(ti):
        other = np.array([b for b, j in enumerate(ti) if src[j] != src[i]])
        order = other[np.argsort(-S[a, other])][:5]
        share.append(float(np.mean(st[ti[order]] == st[i]))); gq.append(src[i])
    prior = float(np.mean([np.mean(st[ti] == st[i]) for i in ti]))
    out["C_generator_fingerprint"] = {"probe_accuracy": round(float(acc), 4), "chance": round(float(chance), 4), "same_generator_share_top5": round(float(np.mean(share)), 4), "ci95": boot(share, gq), "share_prior": round(prior, 4)}
    # ---- E: semantic leakage
    ti = np.where(np.isin(st, ["filmsynth", "sdxl", "zimage"]) & (sp == split) & np.array([r.get("treated", False) for r in rows]))[0]
    so = np.where(st == "source")[0]
    S = E[ti] @ E[so].T; r1 = [src[so[S[a].argmax()]] == src[i] for a, i in enumerate(ti)]
    cc = np.where(st == "cc_original")[0]
    prim = np.array([(rows[i].get("subjects") or ["none"])[0] for i in cc]); cc_sp = sp[cc]
    tr_c, te_c = cc[cc_sp == "train"], cc[cc_sp == split]
    def probe_acc(Z):
        sc = StandardScaler().fit(Z[tr_c]); m = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Z[tr_c]), prim[np.isin(cc, tr_c)])
        return round(float((m.predict(sc.transform(Z[te_c])) == prim[np.isin(cc, te_c)]).mean()), 4)
    maj = round(float(max(Counter(prim[np.isin(cc, te_c)]).values()) / len(te_c)), 4)
    Se, Sc = E[te_c] @ E[te_c].T, Z_content[te_c] @ Z_content[te_c].T
    np.fill_diagonal(Se, -np.inf); np.fill_diagonal(Sc, -np.inf)
    jac = []
    for a in range(len(te_c)):
        ne, nc = set(np.argsort(-Se[a])[:10]), set(np.argsort(-Sc[a])[:10]); jac.append(len(ne & nc) / len(ne | nc))
    out["E_semantic_leakage"] = {"own_source_rank1": round(float(np.mean(r1)), 4), "subject_probe_accuracy": probe_acc(E), "subject_probe_content_ceiling_dino_global": probe_acc(Z_content),
                                 "subject_probe_majority": maj, "n_subject_test": int(len(te_c)), "content_neighbour_jaccard_top10": round(float(np.mean(jac)), 4)}
    # ---- D: BlueNeg (test only)
    if split == "test":
        bi = np.where((st == "blueneg") & np.array([r.get("partition") == "blue-intact" for r in rows]))[0]
        stock = fam[bi]; roll = np.array([rows[i]["roll"] for i in bi])
        import synth_experiment_v3 as V3
        out["D_blueneg_same_stock"] = V3.same_stock(E[bi], stock, roll)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--candidates", default="all")
    ap.add_argument("--extra", default=None, help="module:function returning {name: fn(F)->array}")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    F, rows = load_cache()
    cands = candidates(F, rows, dev)
    if a.extra:
        mod, fn = a.extra.split(":"); cands.update(getattr(importlib.import_module(mod), fn)(F, rows, dev))
    if a.candidates != "all":
        cands = {k: v for k, v in cands.items() if k in a.candidates.split(",")}
    res = {name: evaluate(fn(F), F, rows, a.split) for name, fn in cands.items()}
    rep = {"benchmark": "content-invariance-v1", "split": a.split, "results": res, "seconds": round(time.time() - t0, 1)}
    out = P.REPORTS / f"style_benchmark_{a.split}{('_' + a.tag) if a.tag else ''}.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    hdr = f"{'candidate':24s} {'A p':>6s} {'A2':>6s} {'B fs':>6s} {'B sdxl':>7s} {'B zim':>6s} {'C acc':>6s} {'C shr':>6s} {'E r1':>6s} {'E subj':>7s} {'E jac':>6s}" + (f" {'D p@5':>7s}" if a.split == "test" else "")
    print(hdr)
    for n, r in res.items():
        line = (f"{n:24s} {r['A_treatment_gt_nuisance']['p']:6.3f} {r['A2_style_gap_ratio']['ratio']:6.2f} {r['B_same_treatment_across_scenes']['filmsynth']['precision@5']:6.3f} "
                f"{r['B_same_treatment_across_scenes']['sdxl']['precision@5']:7.3f} {r['B_same_treatment_across_scenes']['zimage']['precision@5']:6.3f} {r['C_generator_fingerprint']['probe_accuracy']:6.3f} "
                f"{r['C_generator_fingerprint']['same_generator_share_top5']:6.3f} {r['E_semantic_leakage']['own_source_rank1']:6.3f} {r['E_semantic_leakage']['subject_probe_accuracy']:7.3f} {r['E_semantic_leakage']['content_neighbour_jaccard_top10']:6.3f}")
        if a.split == "test":
            line += f" {r['D_blueneg_same_stock']['precision@5']:7.3f}"
        print(line)
    r0 = next(iter(res.values()))
    print("chance: B fs", r0["B_same_treatment_across_scenes"]["filmsynth"]["chance"], "B sdxl", r0["B_same_treatment_across_scenes"]["sdxl"]["chance"], "C acc", r0["C_generator_fingerprint"]["chance"],
          "C share prior", r0["C_generator_fingerprint"]["share_prior"], "E subj majority", r0["E_semantic_leakage"]["subject_probe_majority"], "E content ceiling", r0["E_semantic_leakage"]["subject_probe_content_ceiling_dino_global"])


if __name__ == "__main__":
    main()
