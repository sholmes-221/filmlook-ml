#!/usr/bin/env python
"""Model A/B/C comparison across seeds and input resolutions (the key v1 decision).

Each configuration trains N independent seeds; metrics are computed per seed on VAL (for selection)
and TEST (for reporting), then summarised as mean +/- std. Writes reports/model_comparison.json.

  python scripts/compare_models.py --seeds 0 1 2 3 4
"""
import argparse
import json
import time

import numpy as np

from filmlook_ml import paths as P
from filmlook_ml.evaluation.metrics import evaluate
from filmlook_ml.labeling.store import pairs_by_source
from filmlook_ml.labeling.teachers import film_score_from_attributes, heuristic_attributes
from filmlook_ml.training.data import load_train_data
from filmlook_ml.training.train import predict, train_member

CONFIGS = [  # (name, kind, dino input size, texture max_side)
    ("A_texture_1280", "texture", None, 1280),
    ("A_texture_1024", "texture", None, 1024),
    ("A_texture_768", "texture", None, 768),
    ("B_dino_224", "dino", 224, 1280),
    ("C_fused_224", "fused", 224, 1280),
    ("B_dino_168", "dino", 168, 1280),
    ("C_fused_168", "fused", 168, 1280),
    ("B_dino_112", "dino", 112, 1280),
    ("C_fused_112", "fused", 112, 1280),
    # score objective variant: + film-scan > digital cross-image pairs on film_score (train split)
    ("A_texture_1280_mp", "texture", None, 1280),
    ("C_fused_224_mp", "fused", 224, 1280),
]


def summarise(r: dict) -> dict:
    """Flatten the headline numbers of one evaluate() result."""
    out = {}
    for src, p in r["pairwise"].items():
        if p.get("n"):
            out[f"pair_acc_{src}"] = p["acc"]
    cm = r["capture_medium"]
    if cm.get("n", 1):
        out["medium_auroc_head"] = cm["film_capture_head"]["auroc"]
        out["medium_auroc_score"] = cm["film_score_output"]["auroc"]
        h = cm.get("hard_subset_film_vs_filtered_digital")
        if h:
            out["medium_auroc_head_hard"] = h["film_capture_head"]["auroc"]
            out["medium_auroc_score_hard"] = h["film_score_output"]["auroc"]
    for src, ta in r["teacher_agreement"].items():
        fs = ta["film_score"]
        if fs.get("mae") is not None:
            out[f"{src}_score_mae"] = fs["mae"]
            out[f"{src}_score_spearman"] = fs["spearman"]
        maes = [v["mae"] for v in ta["attributes"].values() if v.get("mae") is not None]
        if maes:
            out[f"{src}_attr_mae"] = float(np.mean(maes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--out", default=str(P.REPORTS / "model_comparison.json"))
    a = ap.parse_args()
    cfg, lcfg = P.load_yaml("training.yaml"), P.load_yaml("labeling.yaml")
    results, baseline = {}, None
    for name, kind, size, ms in CONFIGS:
        if a.configs and name not in a.configs:
            continue
        t0 = time.time()
        c = {**cfg, "medium_pairs": {**cfg.get("medium_pairs", {}), "enabled": name.endswith("_mp")}}
        data, manifest = load_train_data(c, lcfg, with_emb=kind != "texture", emb_size=size or 224,
                                         feat_max_side=ms)
        pbs = pairs_by_source(P.LABELS, {i: k for k, i in enumerate(data.ids)}, lcfg["source_weights"])
        if baseline is None:  # the heuristic teacher itself, scored with the same metrics (no training)
            from filmlook_ml.labeling.features import FEATURE_NAMES
            fd = [dict(zip(FEATURE_NAMES, row)) for row in data.feats]
            ha = [heuristic_attributes(f) for f in fd]
            hp = {"score": np.array([film_score_from_attributes(x) / 100 for x in ha]),
                  "attrs": np.array([[0.5 if v is None else v for v in x.values()] for x in ha]),
                  "film_capture": np.array([film_score_from_attributes(x) / 100 for x in ha])}
            baseline = {s: summarise(evaluate(hp, data, manifest, s, pbs)) for s in ("val", "test")}
            results["heuristic_teacher_no_training"] = {"per_seed": {"val": [baseline["val"]],
                                                                     "test": [baseline["test"]]}}
        per = {"val": [], "test": []}
        for s in a.seeds:
            m, _ = train_member(data, kind, c, s, log=lambda *x: None)
            pred = predict([m], data)
            for sp in ("val", "test"):
                per[sp].append(summarise(evaluate(pred, data, manifest, sp, pbs)))
        results[name] = {"kind": kind, "input_size": size, "feat_max_side": ms, "per_seed": per}
        msg = {k: f"{np.mean([d[k] for d in per['test'] if d.get(k) is not None]):.3f}"
               f"±{np.std([d[k] for d in per['test'] if d.get(k) is not None]):.3f}"
               for k in per["test"][0]}
        print(f"{name} ({time.time() - t0:.0f}s) TEST {msg}", flush=True)
    for name, r in results.items():
        r["summary"] = {}
        for sp in ("val", "test"):
            keys = r["per_seed"][sp][0].keys()
            r["summary"][sp] = {k: {"mean": float(np.mean([d[k] for d in r["per_seed"][sp] if d.get(k) is not None])),
                                    "std": float(np.std([d[k] for d in r["per_seed"][sp] if d.get(k) is not None]))}
                                for k in keys}
    P.REPORTS.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write(json.dumps({"seeds": a.seeds, "results": results,
                                       "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
