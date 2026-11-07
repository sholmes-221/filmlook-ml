#!/usr/bin/env python
"""Confidence and abstention for retrieval-style outputs (benchmark v1 Tests B and D).

  python scripts/abstention_analysis.py --split test --candidates hybrid_supcon_family,student_fs_v2_64,colour_tone16

For each candidate embedding and each retrieval task, every query gets a label-free confidence score:
  top1_sim   cosine similarity to the nearest neighbour from another source (another roll for BlueNeg)
  knn_agree  share of the 5 nearest neighbours that carry the top-1 neighbour's label (index labels are known at
             inference; the query label is not)
Queries are then sorted by confidence and the rank-1 precision is reported at coverage 100/75/50/25 % (the
precision among the most confident fraction), with the chance rate and the selective-risk AURC (area under the
risk-coverage curve, lower is better; a score that does not rank errors below hits gives AURC = 1 - precision).
Tasks: B filmsynth same treatment across scenes (test sources); D BlueNeg same stock across rolls (blue-intact partition, 170 scans, 27 rolls).
Nothing is fitted on the evaluated split; the abstention threshold is a deployment choice reported as coverage.
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from style_benchmark import candidates, load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

COVERAGES = (1.0, 0.75, 0.5, 0.25)


def selective(E, labels, groups, exclude_same_group=True):
    """Return per-query (hit, top1_sim, knn_agree) with neighbours restricted to other groups."""
    S = E @ E.T
    same = groups[:, None] == groups[None]
    S = np.where(same if exclude_same_group else np.eye(len(E), dtype=bool), -np.inf, S)
    order = np.argsort(-S, axis=1)[:, :5]
    top = order[:, 0]
    hit = (labels[top] == labels).astype(float)
    top1_sim = S[np.arange(len(E)), top]
    knn_agree = (labels[order] == labels[top][:, None]).mean(1)
    chance = np.mean([(labels[~same[i]] == labels[i]).mean() for i in range(len(E))])
    return hit, top1_sim, knn_agree, float(chance)


def curve(hit, score):
    o = np.argsort(-score); h = hit[o]; n = len(h)
    prec_at = {f"{int(c * 100)}%": round(float(h[: max(1, int(round(c * n)))].mean()), 4) for c in COVERAGES}
    risk = 1 - np.cumsum(h) / np.arange(1, n + 1)
    return {"precision_at_coverage": prec_at, "aurc": round(float(risk.mean()), 4), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--candidates", default="hybrid_supcon_family,student_fs_v2_64,colour_tone16,raw_hybrid")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    F, rows = load_cache()
    cands = {k: v for k, v in candidates(F, rows, dev).items() if k in a.candidates.split(",")}
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    src = np.array([r["source"] for r in rows]); treated = np.array([bool(r.get("treated", False)) for r in rows])
    bmask = (st == "filmsynth") & (sp == a.split) & treated
    dmask = (st == "blueneg") & np.array([r.get("partition") == "blue-intact" for r in rows])
    roll = np.array([r.get("roll", "") for r in rows])
    out = {"benchmark": "content-invariance-v1", "split": a.split, "tasks": {}}
    for name, fn in cands.items():
        E = unit(fn(F).astype(np.float32)); rep = {}
        tasks = [("B_filmsynth_same_treatment", bmask, fam, src)]
        if a.split == "test":
            tasks.append(("D_blueneg_same_stock", dmask, fam, roll))
        for task, m, lab, grp in tasks:
            hit, s1, agree, chance = selective(E[m], lab[m], grp[m])
            rep[task] = {"chance_rank1": round(chance, 4), "top1_sim": curve(hit, s1), "knn_agree": curve(hit, agree),
                         "random_order_aurc": round(float(1 - hit.mean()), 4)}
        out["tasks"][name] = rep
    out["seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / f"abstention_{a.split}.json").write_text(json.dumps(out, indent=1))
    for name, rep in out["tasks"].items():
        for task, r in rep.items():
            print(f"{name:22s} {task:28s} chance {r['chance_rank1']:.3f} | top1_sim p@cov {r['top1_sim']['precision_at_coverage']} aurc {r['top1_sim']['aurc']:.3f}"
                  f" | knn_agree p@cov {r['knn_agree']['precision_at_coverage']} aurc {r['knn_agree']['aurc']:.3f} (unordered {r['random_order_aurc']:.3f})")


if __name__ == "__main__":
    main()
