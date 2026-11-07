#!/usr/bin/env python
"""Evaluate an ensemble checkpoint on val/test; writes reports/eval_report.html (+ .json)."""
import argparse
import json

import numpy as np
from scipy import stats

from filmlook_ml import paths as P
from filmlook_ml.evaluation.metrics import evaluate, failure_cases, html_report
from filmlook_ml.labeling.store import pairs_by_source
from filmlook_ml.models.filmlook_model import load_ensemble
from filmlook_ml.training.data import load_train_data
from filmlook_ml.training.train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(P.CHECKPOINTS / "best"))
    ap.add_argument("--out", default=str(P.REPORTS / "eval_report.html"))
    a = ap.parse_args()
    cfg, lcfg = P.load_yaml("training.yaml"), P.load_yaml("labeling.yaml")
    ens, meta = load_ensemble(a.ckpt)
    data, manifest = load_train_data(cfg, lcfg, with_emb=meta["kind"] != "texture",
                                     emb_size=meta.get("input_size") or 224, feat_max_side=meta["max_side"])
    pred = predict(list(ens.members), data)
    pbs = pairs_by_source(P.LABELS, {i: k for k, i in enumerate(data.ids)}, lcfg["source_weights"])
    srcs = sorted(set(data.t.source))
    sections = {"model": {k: v for k, v in meta.items() if not isinstance(v, list)}}
    for sp in ("val", "test"):
        sections[sp] = evaluate(pred, data, manifest, sp, pbs, srcs)
    # does the heuristic confidence track error? (Spearman between ensemble std and |error| on test)
    m = (data.split == "test") & (data.t.score_w > 0)
    err = np.abs(pred["score"][m] - data.t.score[m])
    sections["test"]["confidence_vs_error_spearman"] = float(stats.spearmanr(pred["score_std"][m], err)[0]) \
        if m.sum() > 3 else None
    html_report(sections, a.out, failure_cases(pred, data, manifest), title="FilmLook v1 evaluation")
    print(json.dumps({sp: {k: v for k, v in sections[sp].items() if k in ("pairwise", "capture_medium",
                                                                          "mean_film_score_by_edit_category",
                                                                          "seed_stability_points")}
                      for sp in ("test",)}, indent=1, default=str))
    print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
