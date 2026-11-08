#!/usr/bin/env python
"""E53: the single read of the session-7 fresh holdouts with the frozen candidate and its baselines.

  python scripts/e53_fresh_holdout.py --self-test     # same code on the spent analysis halves (aic / commoncatalog_film); reports/e53_self_test.json
  python scripts/e53_fresh_holdout.py --final         # reads met_photographs (analysis half) and commoncatalog_film_hd once; refuses to run twice

Candidate (frozen before any holdout image was read: checkpoints/frozen_session7/freeze.json): PCA-32 int8 of the pooled
576-d features of the three degrade-augmented seeds (E52). Comparators, all frozen or deterministic: the same seeds'
PCA-64 and float PCA-32, the E23b seeds (held-out reading E44), the untrained ImageNet backbone with the same PCA, frozen
DINOv2-Small (CLS ++ mean, and L4+L8 statistics) with a PCA-64 fitted on the same CC-BY originals, and the 28 measured
statistics standardised on that set. Readings:
  museum (met_photographs): process p@5, artist excluded, stereo views removed, decade matched; artist share of top-5;
  film scans (commoncatalog_film_hd): stock-line p@5 uploader excluded; uploader share; scanner probe; within-uploader
  different-day AUC (all / colour-only); photographer p@5 (subjects disjoint, aspect matched);
  controlled transformations on the museum half: own-image rank-1 under centre crop 80 %, flip, JPEG q50, 0.5x rescale,
  70 % corner crop;
  effect sizes: paired bootstrap over artists / uploaders of the candidate minus explicit-28 and minus ImageNet-init;
  failure cases: the queries with no relevant neighbour in the top 5 (ids and catalogue labels only, no pixels);
  per-query records (top-10 neighbour labels and cosines) are stored for the abstention analysis so that it never re-reads.
Pre-registered rule (research/backlog.yaml E53; thresholds (b), (c) and the domain of (d) calibrated on the self-test
over the spent halves before the holdout read, see the backlog entry): for every candidate seed's PCA-32 int8, (a) museum
process p@5 lower CI above chance, (b) museum p@5 >= explicit-28 p@5 - 0.15 (the spent-half gap is 0.10-0.13), (c) artist
share <= ImageNet-init's artist share (spent half: 0.14-0.15 vs 0.19-0.21), (d) own-image rank-1 >= 0.9 under the
four invariant transformations on the film scans (museum digitisations recorded only: mounts and borders), (f) uploader share <= ImageNet-init uploader share + 0.02; and (e) the within-uploader AUC
paired difference against ImageNet-init PCA-64 has a lower CI above 0 on the seed-mean. supported = 1 iff all hold.
If the within-uploader reading has fewer than 30 queries, (e) is not evaluable: inconclusive = 1 and supported = 0.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import FIT  # noqa: E402
from eval_commons_film import boot_groups, grouped_probe  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from freeze_candidate import CANDIDATE, COMPARATORS, FROZEN, Frozen  # noqa: E402
from style_benchmark import unit  # noqa: E402
from within_photographer_stock import within_auc  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

PERTURB = ["crop80", "flip", "jpeg50", "rescale0.5", "crop70_corner"]; INVARIANT = PERTURB[:4]
N_BOOT = 1000


# ------------------------------------------------------------------------------------------------- data
def perturb(a, kind):
    from PIL import Image
    h, w = a.shape[:2]
    if kind == "crop80":
        y, x = int(0.1 * h), int(0.1 * w); return a[y:h - y, x:w - x]
    if kind == "crop70_corner":
        return a[: int(0.7 * h), : int(0.7 * w)]
    if kind == "flip":
        return np.ascontiguousarray(a[:, ::-1])
    if kind == "jpeg50":
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=50); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    im = Image.fromarray(a); return np.asarray(im.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR))


def load_arrays(rows):
    with Pool(16) as pool:
        return pool.map(V3._load, [(r["path"], r["image_id"], True, True) for r in rows], chunksize=8)


def museum_labels(rows):
    artist = np.array([r["uploader"] for r in rows]); proc = np.array([process_family(r.get("process")) for r in rows])
    stereo = np.array(["stereo" in " ".join(str(r.get(k) or "") for k in ("process", "title", "classification", "object_name")).lower() for r in rows])
    decade = np.array([str((r["date_start"] // 10) * 10) if isinstance(r.get("date_start"), int) else "nd" for r in rows])
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    return {"artist": artist, "proc": proc, "allowed": allowed, "stereo": stereo, "decade": decade}


def scan_labels(rows):
    up = np.array([r["uploader"] for r in rows]); line = np.array([r.get("film_stock_line") or r["film_stock_claim"] for r in rows]); scanner = np.array([r.get("scanner_category") or "" for r in rows])
    day = np.array([str(r.get("datetaken") or "?")[:10] for r in rows]); colour = np.array([r.get("film_type") in ("colour_negative", "colour_reversal") for r in rows])
    subj = [set(r.get("subjects") or []) for r in rows]; lar = np.array([np.log(r["image_width"] / r["image_height"]) for r in rows])
    photog_ok = np.array([[not (a & b) for b in subj] for a in subj]) & (np.abs(lar[:, None] - lar[None]) < 0.1)
    per = {u: Counter(line[up == u]) for u in set(up)}; kmulti = np.isin(up, [u for u, c in per.items() if len(c) >= 2 and sum(c.values()) >= 4])
    return {"up": up, "line": line, "scanner": scanner, "has": scanner != "", "day": day, "colour": colour, "photog_ok": photog_ok, "kmulti": kmulti}


# ------------------------------------------------------------------------------------------------- metrics (per query, for pairing)
def hits(E, label, group, allowed=None, k=5):
    """Per-query top-k precision with the query's group excluded; returns (hits, groups, prior, query_index, top_index)."""
    S = E @ E.T; h, g, pr, qi, top = [], [], [], [], []
    for i in range(len(E)):
        ok = group != group[i]
        if allowed is not None:
            ok &= allowed[i]
        rel = label[ok] == label[i]
        if rel.sum() == 0 or (~rel).sum() == 0 or ok.sum() < k:
            continue
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:10]
        h.append((label[order[:k]] == label[i]).mean()); pr.append(rel.mean()); g.append(group[i]); qi.append(i); top.append(order)
    return np.array(h), np.array(g), np.array(pr), np.array(qi, int), np.array(top, int).reshape(-1, 10)


def summarise(h, g, pr):
    if len(h) == 0:
        return {"precision@5": None, "n": 0}
    return {"precision@5": round(float(h.mean()), 4), "ci95": boot_groups(h, g), "chance": round(float(pr.mean()), 4), "n": int(len(h)), "n_groups": int(len(set(g)))}


def paired_diff(a, b, groups, seed=0):
    """Mean of (a - b) with a bootstrap CI over groups (both arrays are per-query on the same query set)."""
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups); d = a - b; ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed); bs = [d[np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(N_BOOT)]
    sd = d.std() + 1e-9
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "cohen_d": round(float(d.mean() / sd), 3), "n": int(len(d))}


def share(E, group):
    S = E @ E.T; np.fill_diagonal(S, -np.inf)
    return round(float(np.mean([(group[np.argsort(-S[i])[:5]] == group[i]).mean() for i in range(len(E))])), 4)


def rank1(Eq, Eb):
    return round(float(((unit(Eq) @ unit(Eb).T).argmax(1) == np.arange(len(Eq))).mean()), 4)


# ------------------------------------------------------------------------------------------------- representations
def representations(arrays_by_set, dev):
    """{name: {set: embedding}} for every frozen tag (pca64 / pca32 / pca32_int8), DINOv2 (pca64) and explicit-28 (std)."""
    from PIL import Image
    reps = {}; frozen = {t: Frozen(t) for t in CANDIDATE + COMPARATORS}
    feats = {t: {s: frozen[t].features(a) for s, a in arrays_by_set.items()} for t in frozen}
    for t in frozen:
        reps[f"{t}:pca64"] = {s: frozen[t].reduce(F, 64, False) for s, F in feats[t].items()}
        reps[f"{t}:pca32"] = {s: frozen[t].reduce(F, 32, False) for s, F in feats[t].items()}
        reps[f"{t}:pca32_int8"] = {s: frozen[t].reduce(F, 32, True) for s, F in feats[t].items()}
    Fit = dict(np.load(FIT.with_suffix(".npz"))); mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6
    dino = X.Dino(dev); ims = {s: [Image.fromarray(x) for x in a] for s, a in arrays_by_set.items()}
    ex = {s: np.stack([X.explicit_of(im) for im in ims[s]]).astype(np.float32) for s in ims}
    dg, dsty = {}, {}
    for s in ims:
        dg[s], dsty[s] = dino(ims[s])
    reps["explicit28_std"] = {s: (ex[s] - mu) / sd for s in ims}
    for key, D in (("dino_global", dg), ("dino_style", dsty)):
        pca = PCA64(Fit[key], 64); reps[f"{key}:pca64"] = {s: pca(np.asarray(D[s], np.float32)) for s in ims}
    return reps, frozen, feats


def evaluate(reps, frozen, arrays, rows_m, rows_s, Lm, Ls, label):
    out = {"representations": {}, "queries": {}}
    for name, e in reps.items():
        r = {}
        if rows_m:
            Em = unit(np.nan_to_num(e["museum"].astype(np.float32))); h, g, pr, qi, top = hits(Em, Lm["proc"], Lm["artist"], Lm["allowed"])
            r["museum_p5"] = summarise(h, g, pr); r["artist_share"] = share(Em, Lm["artist"]); out["queries"][name + "|museum"] = {"hits": h, "groups": g, "qi": qi, "top": top, "S_top": (Em @ Em.T)[qi[:, None], top]}
        if rows_s:
            Es = unit(np.nan_to_num(e["scans"].astype(np.float32))); h, g, pr, qi, top = hits(Es, Ls["line"], Ls["up"])
            r["stock_line_p5"] = summarise(h, g, pr); r["uploader_share"] = share(Es, Ls["up"]); r["scanner_probe"] = grouped_probe(Es[Ls["has"]], Ls["scanner"][Ls["has"]], Ls["up"][Ls["has"]])
            hp, gp, prp, _, _ = hits(Es, Ls["up"], np.arange(len(Es)), Ls["photog_ok"]); r["photographer_p5"] = summarise(hp, gp, prp)
            a_all, g_all = within_auc(Es[Ls["kmulti"]], Ls["line"][Ls["kmulti"]], Ls["up"][Ls["kmulti"]], Ls["day"][Ls["kmulti"]], True)
            a_col, g_col = within_auc(Es[Ls["kmulti"]], Ls["line"][Ls["kmulti"]], Ls["up"][Ls["kmulti"]], Ls["day"][Ls["kmulti"]], True, Ls["colour"][Ls["kmulti"]])
            r["within_uploader_auc"] = {"auc": round(float(a_all.mean()), 4), "ci95": boot_groups(a_all, g_all), "n_queries": int(len(a_all)), "n_uploaders": int(len(set(g_all)))} if len(a_all) else {"auc": None, "n_queries": 0}
            r["colour_only_auc"] = {"auc": round(float(a_col.mean()), 4), "ci95": boot_groups(a_col, g_col), "n_queries": int(len(a_col))} if len(a_col) else {"auc": None, "n_queries": 0}
            out["queries"][name + "|scans"] = {"hits": h, "groups": g, "qi": qi, "top": top, "S_top": (Es @ Es.T)[qi[:, None], top], "within_auc": a_all, "within_groups": g_all}
        r["eff_rank_museum"] = eff_rank(e["museum"]) if rows_m else None
        out["representations"][name] = r
        print(f"{name:40s} " + (f"museum {r['museum_p5']['precision@5']} {r['museum_p5']['ci95']} ch {r['museum_p5']['chance']} art {r['artist_share']} " if rows_m else "") +
              (f"| stock {r['stock_line_p5']['precision@5']} {r['stock_line_p5'].get('ci95')} ch {r['stock_line_p5'].get('chance')} up {r['uploader_share']} within {r['within_uploader_auc']['auc']} {r['within_uploader_auc'].get('ci95')} n {r['within_uploader_auc']['n_queries']}" if rows_s else ""), flush=True)
    # controlled transformations (candidate and comparator seeds, PCA-32 int8): the film scans carry the rule (user-like
    # photographs); the museum digitisations are recorded (the self-test on the spent AIC half showed that a 10 % crop
    # loses the nearest neighbour a third of the time there, consistent with mounts and borders being part of the signal)
    out["robustness"] = {}
    for s in [k for k in ("scans", "museum") if k in arrays]:
        for t in CANDIDATE + COMPARATORS:
            base = frozen[t].reduce(frozen[t].features(arrays[s]), 32, True); rr = {}
            for kind in PERTURB:
                Z = frozen[t].reduce(frozen[t].features([perturb(a, kind) for a in arrays[s]]), 32, True); rr[kind] = rank1(Z, base)
            rr["min_invariant"] = min(rr[k] for k in INVARIANT); out["robustness"][f"{t}|{s}"] = rr; print(f"robustness {t} on {s}: {rr}", flush=True)
    return out


def eff_rank(Z):
    Zc = unit(np.nan_to_num(Z.astype(np.float32))); Zc = Zc - Zc.mean(0); lam = np.linalg.svd(Zc, compute_uv=False) ** 2
    return round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2)


def failure_cases(q, rows, label, group, n=10):
    """Queries with no relevant neighbour in the top 5: ids and catalogue labels only."""
    bad = np.where(q["hits"] == 0)[0][:n]; out = []
    for j in bad:
        i = int(q["qi"][j]); r = rows[i]; nb = q["top"][j][:5]
        out.append({"image_id": r["image_id"], "label": str(label[i]), "group": str(group[i]), "top5_labels": [str(label[k]) for k in nb], "top5_cos": [round(float(c), 3) for c in q["S_top"][j][:5]]})
    return {"n_zero_hit": int((q["hits"] == 0).sum()), "n_queries": int(len(q["hits"])), "examples": out}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if not (a.final or a.self_test):
        raise SystemExit("pass --self-test (spent analysis halves) or --final (the single holdout read)")
    OUT = P.REPORTS / ("e53_fresh_holdout.json" if a.final else "e53_self_test.json"); QOUT = OUT.with_name(OUT.stem + "_queries.npz")
    if a.final and OUT.exists():
        raise SystemExit(f"{OUT} exists: the fresh holdouts have been read; the evaluation is not repeated")
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze = json.loads((FROZEN / "freeze.json").read_text())
    for t, rec in freeze["tags"].items():                                     # the frozen bytes are what is evaluated
        h = hashlib.sha256((FROZEN / t / "model.onnx").read_bytes()).hexdigest(); assert h == rec["model_sha256"], f"frozen model changed: {t}"
    if a.final:
        rows_m = [r for r in read_jsonl(P.MANIFESTS / "met_photographs_manifest.jsonl") if r["split"] == "analysis"]
        rows_s = [r for r in read_jsonl(P.MANIFESTS / "commoncatalog_film_hd_manifest.jsonl") if r.get("film_stock")]   # brand-only rows carry no stock label and are left out
        label = {"museum": "met_photographs analysis half (test half untouched)", "scans": "commoncatalog_film_hd, all rows with a stock claim"}
    else:
        rows_m = [r for r in read_jsonl(P.MANIFESTS / "aic_photographs_manifest.jsonl") if r["split"] == "analysis"]
        rows_s = [r for r in read_jsonl(P.MANIFESTS / "commoncatalog_film_manifest.jsonl") if r["split"] == "analysis" and r.get("film_stock")]
        label = {"museum": "aic_photographs analysis half (spent; self-test only)", "scans": "commoncatalog_film analysis half (spent; self-test only)"}
    arrays = {}
    if rows_m:
        arrays["museum"] = load_arrays(rows_m)
    if rows_s:
        arrays["scans"] = load_arrays(rows_s)
    Lm = museum_labels(rows_m) if rows_m else None; Ls = scan_labels(rows_s) if rows_s else None
    reps, frozen, _ = representations(arrays, dev)
    res = evaluate(reps, frozen, arrays, rows_m, rows_s, Lm, Ls, label)
    R = res["representations"]; Q = res["queries"]
    # effect sizes: candidate seeds (PCA-32 int8) minus explicit-28 and minus ImageNet-init PCA-64, paired over queries
    eff = {}
    for t in CANDIDATE:
        c = f"{t}:pca32_int8"; e = {}
        for base in ("explicit28_std", "imagenet_init:pca64", "dino_global:pca64", "style_fs_v2_real_f576:pca32_int8"):
            if rows_m:
                e[f"museum_p5_vs_{base}"] = paired_diff(Q[c + "|museum"]["hits"], Q[base + "|museum"]["hits"], Q[c + "|museum"]["groups"])
            if rows_s:
                e[f"stock_line_p5_vs_{base}"] = paired_diff(Q[c + "|scans"]["hits"], Q[base + "|scans"]["hits"], Q[c + "|scans"]["groups"])
                if len(Q[c + "|scans"]["within_auc"]):
                    e[f"within_auc_vs_{base}"] = paired_diff(Q[c + "|scans"]["within_auc"], Q[base + "|scans"]["within_auc"], Q[c + "|scans"]["within_groups"])
        eff[t] = e
    # seed-mean of the within-uploader AUC difference against ImageNet-init (criterion e)
    within_mean = None
    if rows_s and len(Q[f"{CANDIDATE[0]}:pca32_int8|scans"]["within_auc"]):
        m = np.mean([Q[f"{t}:pca32_int8|scans"]["within_auc"] for t in CANDIDATE], 0)
        within_mean = paired_diff(m, Q["imagenet_init:pca64|scans"]["within_auc"], Q[f"{CANDIDATE[0]}:pca32_int8|scans"]["within_groups"])
    n_within = R[f"{CANDIDATE[0]}:pca32_int8"]["within_uploader_auc"]["n_queries"] if rows_s else 0
    checks = {}
    for t in CANDIDATE:
        c = R[f"{t}:pca32_int8"]; ch = {}
        if rows_m:
            ch["a_museum_ci_low_above_chance"] = c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"]
            ch["b_museum_not_below_explicit_minus_0.15"] = c["museum_p5"]["precision@5"] >= R["explicit28_std"]["museum_p5"]["precision@5"] - 0.15
            ch["c_artist_share_not_above_init"] = c["artist_share"] <= R["imagenet_init:pca64"]["artist_share"]
        if rows_s:
            ch["d_invariant_rank1_ge_0.9_scans"] = res["robustness"][f"{t}|scans"]["min_invariant"] >= 0.9
            ch["f_uploader_share_not_above_init"] = c["uploader_share"] <= R["imagenet_init:pca64"]["uploader_share"] + 0.02
        checks[t] = ch
    e_ok = (within_mean is not None and n_within >= 30 and within_mean["ci95"][0] > 0)
    inconclusive = int(rows_s is not None and n_within < 30)
    supported = int(all(all(ch.values()) for ch in checks.values()) and e_ok)
    cand = [R[f"{t}:pca32_int8"] for t in CANDIDATE]
    summary = {"supported": supported, "inconclusive": inconclusive, "e_within_diff_vs_init_seed_mean": within_mean, "n_within_queries": n_within, "checks": checks,
               "candidate_museum_p5_seed_mean": round(float(np.mean([c["museum_p5"]["precision@5"] for c in cand])), 4) if rows_m else None,
               "candidate_museum_ci_low_min": min(c["museum_p5"]["ci95"][0] for c in cand) if rows_m else None, "museum_chance": cand[0]["museum_p5"]["chance"] if rows_m else None,
               "explicit_museum_p5": R["explicit28_std"]["museum_p5"]["precision@5"] if rows_m else None, "init_museum_p5": R["imagenet_init:pca64"]["museum_p5"]["precision@5"] if rows_m else None,
               "dino_global_museum_p5": R["dino_global:pca64"]["museum_p5"]["precision@5"] if rows_m else None, "dino_style_museum_p5": R["dino_style:pca64"]["museum_p5"]["precision@5"] if rows_m else None,
               "candidate_artist_share_max": max(c["artist_share"] for c in cand) if rows_m else None,
               "candidate_stock_line_p5_seed_mean": round(float(np.mean([c["stock_line_p5"]["precision@5"] for c in cand])), 4) if rows_s and cand[0]["stock_line_p5"]["precision@5"] is not None else None,
               "stock_line_chance": cand[0]["stock_line_p5"].get("chance") if rows_s else None, "candidate_within_auc_seed_mean": round(float(np.mean([c["within_uploader_auc"]["auc"] for c in cand])), 4) if rows_s and n_within else None,
               "init_within_auc": R["imagenet_init:pca64"]["within_uploader_auc"]["auc"] if rows_s and n_within else None, "explicit_within_auc": R["explicit28_std"]["within_uploader_auc"]["auc"] if rows_s and n_within else None,
               "candidate_uploader_share_max": max(c["uploader_share"] for c in cand) if rows_s else None, "init_uploader_share": R["imagenet_init:pca64"]["uploader_share"] if rows_s else None,
               "robustness_min_invariant_min": min(res["robustness"][f"{t}|scans"]["min_invariant"] for t in CANDIDATE) if rows_s else None,
               "corner_crop_min": min(res["robustness"][f"{t}|scans"]["crop70_corner"] for t in CANDIDATE) if rows_s else None,
               "museum_robustness_min_invariant_min": min(res["robustness"][f"{t}|museum"]["min_invariant"] for t in CANDIDATE) if rows_m else None, "seconds": round(time.time() - t0, 1)}
    fails = {}
    if rows_m:
        fails["museum"] = failure_cases(Q[f"{CANDIDATE[0]}:pca32_int8|museum"], rows_m, Lm["proc"], Lm["artist"])
    if rows_s:
        fails["scans"] = failure_cases(Q[f"{CANDIDATE[0]}:pca32_int8|scans"], rows_s, Ls["line"], Ls["up"])
    out = {"experiment": "E53_fresh_holdout" if a.final else "E53_self_test", "held_out_data": bool(a.final), "data": label, "n_museum": len(rows_m), "n_scans": len(rows_s), "n_museum_artists": int(len(set(Lm["artist"]))) if rows_m else 0,
           "n_scan_uploaders": int(len(set(Ls["up"]))) if rows_s else 0, "frozen": {t: freeze["tags"][t]["model_sha256"] for t in freeze["tags"]}, "representations": R, "robustness": res.get("robustness"), "effect_sizes": eff,
           "failure_cases": fails, "summary": summary}
    P.REPORTS.mkdir(exist_ok=True); OUT.write_text(json.dumps(out, indent=1, default=str))
    np.savez(QOUT, **{f"{k}|{f}": v for k, q in Q.items() for f, v in q.items()}, museum_proc=Lm["proc"] if rows_m else np.array([]), museum_artist=Lm["artist"] if rows_m else np.array([]),
             scans_line=Ls["line"] if rows_s else np.array([]), scans_up=Ls["up"] if rows_s else np.array([]), museum_ids=np.array([r["image_id"] for r in rows_m]), scans_ids=np.array([r["image_id"] for r in rows_s]))
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
