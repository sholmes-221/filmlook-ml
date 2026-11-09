#!/usr/bin/env python
"""Does fusion help? DINOv2 only vs semantic only vs Visual DNA only vs fused, on photographers that have both
licensed reference images and a researched profile.

  python scripts/refdb_fusion_eval.py [--seed 0]

Same splits as refdb_eval.py (session-grouped 60/20/20). Signals per query image, each a score vector over the
candidates:
- dinov2:   similarity-weighted kNN vote (k=10) against the train bank, DINOv2-small CLS vectors;
- semantic: the same kNN on SigLIP image embeddings (content similarity to the photographer's works);
- dna:      Visual DNA knowledge score from the query's detected tags (SigLIP zero-shot, per-tag logit z >= 1.5 against
            the reference corpus, top 6 - uncalibrated, documented) and pixel-measured tags. No reference images are used by this signal.
Temperatures per signal and the fusion weights are chosen on val (weight grid in steps of 0.05, starting point the
brief's 0.45 / 0.35 / 0.20); test is read once. Also reported: the brief's initial weights on test, unchanged.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from refdb_eval import fit_temperature, load, session_key, softmax, split_of  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.visualdna import dna  # noqa: E402
from filmlook_ml.visualdna.semantic import measured_tags  # noqa: E402

PROFILED = ("dorothea_lange", "walker_evans", "lewis_hine", "eugene_atget", "alfred_stieglitz")


def knn_scores(Xq, Xb, yb, K, k=10):
    sims = Xq @ Xb.T; idx = np.argsort(-sims, axis=1)[:, :k]; S = np.zeros((len(Xq), K))
    for i in range(len(Xq)):
        for j in idx[i]:
            S[i, yb[j]] += max(sims[i, j], 0)
    return S / k


def metrics(S, y, K):
    o = np.argsort(-S, axis=1)
    return {"top1": float((o[:, 0] == y).mean()), "top3": float(np.mean([y[i] in o[i, :3] for i in range(len(y))])),
            "macro_f1": float(f1_score(y, o[:, 0], labels=list(range(K)), average="macro", zero_division=0)), "n": int(len(y))}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, Xd = load("dinov2s_cls_224")
    z = np.load(P.DATA / "refdb" / "siglip_v1.npz", allow_pickle=True)
    pos = {w: i for i, w in enumerate(z["work_id"])}
    keep = [i for i, r in enumerate(rows) if r["photographer_id"] in PROFILED and r["work_id"] in pos]
    rows = [rows[i] for i in keep]; Xd = Xd[keep]
    Xs = z["X"][[pos[r["work_id"]] for r in rows]]
    tags = list(z["tag_names"]); TL = z["tag_logits"][[pos[r["work_id"]] for r in rows]]
    stats = json.loads((P.DATA / "refdb" / "siglip_tag_stats.json").read_text())
    mu = np.array([stats[t][0] for t in tags]); sd = np.array([stats[t][1] for t in tags])
    TZ = (TL - mu) / np.maximum(sd, 1e-6)
    meas = [json.loads(m) for m in z["measured"][[pos[r["work_id"]] for r in rows]]]
    cands = [p for p in PROFILED if any(r["photographer_id"] == p for r in rows)]
    K = len(cands); cid = {p: i for i, p in enumerate(cands)}
    y = np.array([cid[r["photographer_id"]] for r in rows])
    split = np.array([split_of(session_key(r), a.seed) for r in rows])
    tr, va, te = split == "train", split == "val", split == "test"
    kb = dna.knowledge()

    def dna_scores(idx):
        S = np.zeros((len(idx), K))
        for n, i in enumerate(idx):
            order = np.argsort(-TZ[i])[:6]
            det = {tags[j]: min(1.0, float(TZ[i, j]) / 4) for j in order if TZ[i, j] >= 1.5}  # z mapped to (0, 1] as strength
            mt = {t for t, v in measured_tags(meas[i]).items() if v}
            for c, p in enumerate(cands):
                S[n, c] = dna.knowledge_score(p, det, mt, kb)["total"]
        return S

    sig = {}
    for name, part in (("val", va), ("test", te)):
        idx = np.where(part)[0]
        sig[name] = {"dinov2": knn_scores(Xd[idx], Xd[tr], y[tr], K), "semantic": knn_scores(Xs[idx], Xs[tr], y[tr], K), "dna": dna_scores(idx)}
    T = {k: fit_temperature(sig["val"][k], y[va]) for k in ("dinov2", "semantic", "dna")}
    P_ = {n: {k: np.apply_along_axis(lambda s: softmax(s[None], T[k])[0], 1, v) for k, v in d.items()} for n, d in sig.items()}

    def fused(part, w):
        return sum(w[k] * P_[part][k] for k in w)

    grid = [dict(zip(("dinov2", "semantic", "dna"), (a_, b_, round(1 - a_ - b_, 2)))) for a_, b_ in itertools.product(np.arange(0, 1.01, 0.05), repeat=2) if a_ + b_ <= 1.0001]
    best = max(grid, key=lambda w: (metrics(fused("val", w), y[va], K)["top1"], -abs(w["dinov2"] - 0.45)))
    initial = {"dinov2": 0.45, "semantic": 0.35, "dna": 0.20}
    res = {"candidates": cands, "counts": {p: {s: int(((y == cid[p]) & (split == s)).sum()) for s in ("train", "val", "test")} for p in cands},
           "temperatures": T, "weights_from_val": best,
           "test": {k: metrics(sig["test"][k], y[te], K) for k in ("dinov2", "semantic", "dna")}}
    res["test"]["fused_initial_weights"] = metrics(fused("test", initial), y[te], K)
    res["test"]["fused_val_weights"] = metrics(fused("test", best), y[te], K)
    # Agreement rule: how precise is 'two signals rank the same photographer first'?
    firsts = {k: P_["test"][k].argmax(1) for k in ("dinov2", "semantic", "dna")}
    top = fused("test", best).argmax(1)
    agree = np.array([sum(firsts[k][i] == top[i] for k in firsts) >= 2 for i in range(len(top))])
    res["test"]["strong_rule"] = {"share_labelled_strong": float(agree.mean()), "precision_when_strong": float((top[agree] == y[te][agree]).mean()) if agree.any() else None,
                                  "accuracy_when_not_strong": float((top[~agree] == y[te][~agree]).mean()) if (~agree).any() else None}
    res["chance_uniform"] = 1 / K
    out = P.REPORTS / "refdb" / f"fusion_seed{a.seed}.json"; out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["test"], indent=1)); print("weights", best, "T", T, "counts", res["counts"])


if __name__ == "__main__":
    main()
