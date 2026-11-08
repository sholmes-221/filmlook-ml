"""E57: retrieval methods on the fresh sets. Hybrid descriptors, class-balanced weighted kNN, prototypes, calibration.

Exploratory (post-hoc) analysis on the same fresh sets E53 read once (Met analysis half, commoncatalog_film_hd scans,
ARTeFACT). Nothing here is a pre-registered rule; it compares *methods* on a fixed representation set and reports
numbers with CIs over artists / uploaders. Representations come from `e53_fresh_holdout.representations`
(frozen student PCA-32 int8, untrained ImageNet backbone PCA-64, frozen DINOv2-Small global PCA-64, the 28
standardised statistics); the new ones are concatenations (hybrids) of a learned embedding with the statistics.

Methods compared per representation and task:
  A  cosine top-5 precision with the query's artist / uploader excluded (E53 metric, for reference)
  E1 kNN classification with K in {5, 10, 20}: plain vote, cosine-weighted vote, class-balanced cosine-weighted vote
     (weights divided by the label's frequency in the candidate pool), with top-1 accuracy, class-balanced
     (macro) accuracy and ECE from the winning-vote share as confidence
  E2 prototype matching: nearest class centroid built from every image outside the query's group
Tasks: museum process (Met; candidates decade-matched, non-stereo) and film-stock line (scans; uploader excluded).
Also: within-uploader different-day stock AUC for the hybrids, storage and brute-force latency estimates.

  .venv/bin/python scripts/e57_retrieval_methods.py            -> reports/e57_retrieval_methods.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e53_fresh_holdout as E  # noqa: E402
from eval_commons_film import boot_groups  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from style_benchmark import unit  # noqa: E402
from within_photographer_stock import within_auc  # noqa: E402

CAND = "style_fs_v2_real_f576_deg:pca32_int8"
INIT = "imagenet_init:pca64"
DINO = "dino_global:pca64"
EXPL = "explicit28_std"
KS = (5, 10, 20)
WEIGHTS = (0.5, 1.0, 2.0)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(total), 4)


def knn_classify(Eu, label, group, allowed=None, k=5, mode="plain"):
    """Per-query predicted label and confidence with the query's group excluded. mode: plain | weighted | balanced."""
    S = Eu @ Eu.T; preds, confs, qi = [], [], []
    for i in range(len(Eu)):
        ok = group != group[i]
        if allowed is not None:
            ok &= allowed[i]
        if ok.sum() < k or (label[ok] == label[i]).sum() == 0:
            continue
        pool = label[ok]
        freq = Counter(pool)
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:k]
        votes: dict = {}
        for j in order:
            w = 1.0 if mode == "plain" else max(S[i, j], 1e-6)
            if mode == "balanced":
                w = w / freq[label[j]]
            votes[label[j]] = votes.get(label[j], 0.0) + w
        best = max(votes, key=votes.get); total = sum(votes.values())
        preds.append(best); confs.append(votes[best] / total if total > 0 else 0.0); qi.append(i)
    return np.array(preds), np.array(confs), np.array(qi, int)


def prototype_classify(Eu, label, group, allowed=None):
    preds, confs, qi = [], [], []
    for i in range(len(Eu)):
        ok = group != group[i]
        if allowed is not None:
            ok &= allowed[i]
        pool = label[ok]
        if (pool == label[i]).sum() == 0 or len(set(pool)) < 2:
            continue
        classes = sorted(set(pool)); cents = unit(np.stack([Eu[ok][pool == c].mean(0) for c in classes]))
        sims = cents @ Eu[i]; j = int(np.argmax(sims))
        soft = np.exp((sims - sims.max()) * 10); soft /= soft.sum()
        preds.append(classes[j]); confs.append(float(soft[j])); qi.append(i)
    return np.array(preds), np.array(confs), np.array(qi, int)


def classification_summary(preds, confs, qi, label, group):
    if len(qi) == 0:
        return None
    truth = label[qi]; correct = (preds == truth).astype(float); g = group[qi]
    per_class = {c: float(correct[truth == c].mean()) for c in set(truth)}
    return {"accuracy": round(float(correct.mean()), 4), "ci95_over_groups": boot_groups(correct, g), "balanced_accuracy": round(float(np.mean(list(per_class.values()))), 4),
            "n_classes": len(per_class), "ece": ece(confs, correct), "mean_confidence": round(float(np.mean(confs)), 4), "n_queries": int(len(qi)),
            "chance_balanced": round(1.0 / len(per_class), 4)}


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows_m = [r for r in read_jsonl(P.MANIFESTS / "met_photographs_manifest.jsonl") if r["split"] == "analysis" and min(r["image_width"], r["image_height"]) >= 256]
    rows_s = [r for r in read_jsonl(P.MANIFESTS / "commoncatalog_film_hd_manifest.jsonl") if r.get("film_stock")]
    arrays = {"museum": E.load_arrays(rows_m), "scans": E.load_arrays(rows_s)}
    Lm, Ls = E.museum_labels(rows_m), E.scan_labels(rows_s)
    reps, frozen, _ = E.representations(arrays, dev)
    base = {k: reps[k] for k in (CAND, INIT, DINO, EXPL)}
    hybrids = {}
    for name, key in (("cand", CAND), ("init", INIT), ("dino", DINO)):
        for w in WEIGHTS:
            hybrids[f"hybrid_{name}+expl_w{w}"] = {s: np.concatenate([unit(np.nan_to_num(base[key][s].astype(np.float32))), w * unit(np.nan_to_num(base[EXPL][s].astype(np.float32)))], 1) for s in arrays}
    allreps = {**base, **hybrids}
    out = {"experiment": "E57 retrieval methods (post-hoc on the E53 fresh sets)", "n_museum": len(rows_m), "n_scans": len(rows_s), "representations": {}}
    for name, e in allreps.items():
        r = {}
        Em = unit(np.nan_to_num(e["museum"].astype(np.float32))); Es = unit(np.nan_to_num(e["scans"].astype(np.float32)))
        h, g, pr, _, _ = E.hits(Em, Lm["proc"], Lm["artist"], Lm["allowed"]); r["museum_p5"] = E.summarise(h, g, pr); r["artist_share"] = E.share(Em, Lm["artist"])
        h, g, pr, _, _ = E.hits(Es, Ls["line"], Ls["up"]); r["stock_line_p5"] = E.summarise(h, g, pr); r["uploader_share"] = E.share(Es, Ls["up"])
        a_all, g_all = within_auc(Es[Ls["kmulti"]], Ls["line"][Ls["kmulti"]], Ls["up"][Ls["kmulti"]], Ls["day"][Ls["kmulti"]], True)
        r["within_uploader_auc"] = {"auc": round(float(a_all.mean()), 4), "ci95": boot_groups(a_all, g_all), "n_queries": int(len(a_all))} if len(a_all) else None
        r["knn"] = {}
        for task, (Eu, label, group, allowed) in {"museum_process": (Em, Lm["proc"], Lm["artist"], Lm["allowed"]), "stock_line": (Es, Ls["line"], Ls["up"], None)}.items():
            r["knn"][task] = {}
            for k in KS:
                for mode in ("plain", "weighted", "balanced"):
                    p, c, qi = knn_classify(Eu, label, group, allowed, k, mode)
                    r["knn"][task][f"k{k}_{mode}"] = classification_summary(p, c, qi, label, group)
            p, c, qi = prototype_classify(Eu, label, group, allowed)
            r["knn"][task]["prototype"] = classification_summary(p, c, qi, label, group)
        r["dims"] = int(e["scans"].shape[1])
        out["representations"][name] = r
        sb = r["knn"]["stock_line"]; mb = r["knn"]["museum_process"]
        print(f"{name:28s} museum p5 {r['museum_p5']['precision@5']} art {r['artist_share']} | stock p5 {r['stock_line_p5']['precision@5']} up {r['uploader_share']} "
              f"within {r['within_uploader_auc']['auc'] if r['within_uploader_auc'] else None} | stock bal-acc k10 bal {sb['k10_balanced']['balanced_accuracy']} proto {sb['prototype']['balanced_accuracy']} "
              f"| museum bal-acc k10 bal {mb['k10_balanced']['balanced_accuracy']} proto {mb['prototype']['balanced_accuracy']} ece {mb['k10_balanced']['ece']}", flush=True)
    # storage and brute-force latency estimates (CPU, this host; the E2 Micro core is several times slower)
    est = {}
    for dims, dtype, name in ((32, np.int8, "pca32_int8"), (60, np.float16, "hybrid60_f16"), (64, np.float32, "pca64_f32")):
        for n in (10_000, 50_000, 100_000):
            bank = np.random.default_rng(0).standard_normal((n, dims)).astype(np.float32); bank /= np.linalg.norm(bank, axis=1, keepdims=True)
            q = bank[0]; t = time.perf_counter()
            for _ in range(20):
                top = np.argpartition(-(bank @ q), 5)[:5]
            ms = (time.perf_counter() - t) / 20 * 1000
            est[f"{name}_n{n}"] = {"bytes": int(n * dims * np.dtype(dtype).itemsize), "mb": round(n * dims * np.dtype(dtype).itemsize / 1e6, 1), "brute_force_ms_this_host": round(ms, 2)}
    out["storage_latency_estimates"] = est
    out["seconds"] = round(time.time() - t0, 1)
    OUT = P.REPORTS / "e57_retrieval_methods.json"; OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT} in {out['seconds']} s")


if __name__ == "__main__":
    main()
