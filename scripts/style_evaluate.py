#!/usr/bin/env python
"""Compare style methods against INDEPENDENT judgements (data/labels/style_judgements.jsonl).

Methods (all ranked over profile ids + "none"):
  profile   - shipped v1 engine (no training)
  linear / forest / mlp - learned on JS-reproducible style features
  dinov2    - logistic regression on frozen DINOv2-small embeddings (server-only baseline)
Learned methods: GroupKFold by uploader. They are trained on --train-source judgements and scored on
--eval-source judgements (default: train on claude, evaluate on human = independent of the teacher).
Teacher agreement (eval-source claude) is reported separately and is NOT human accuracy.

  python scripts/style_evaluate.py --train-source claude --eval-source human
"""
import argparse
import json

import numpy as np

from filmlook_ml import paths as P
from filmlook_ml.compliance.policy import THIRD_PARTY_SOURCES, claude_training_permission, policy_docs_status
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.style import learned as L
from filmlook_ml.style.engine import analyze_features
from filmlook_ml.style.features import feature_vector
from filmlook_ml.style.profiles import PROFILE_IDS

MIN_JUDGEMENTS = 30


def truth_set(j):
    return {"none"} if j["no_match"] or not j["matches"] else set(j["matches"])


def profile_ranked(f, q):
    r = analyze_features(f, q)
    ids = [m["id"] for m in r["matches"]]
    rest = [p for p in PROFILE_IDS if p not in ids]
    return (["none"] + ids + rest) if r["status"] == "uncertain" else (ids[:1] + ["none"] + ids[1:] + rest
                                                                          if not r["matches"][1]["reliable"]
                                                                          else ids + ["none"] + rest)


def run(judgements, feats, groups, q, train_source, eval_source, emb=None, seeds=(0, 1, 2)):
    from sklearn.model_selection import GroupKFold
    ev = [j for j in judgements if j["source"] == eval_source]
    tr_all = [j for j in judgements if j["source"] == train_source]
    res = {"n_eval": len(ev), "n_train_pool": len(tr_all), "eval_source": eval_source, "train_source": train_source}
    if len(ev) < MIN_JUDGEMENTS:
        res["status"] = f"BLOCKED: only {len(ev)} {eval_source} judgements (< {MIN_JUDGEMENTS})"
        return res
    truth = [truth_set(j) for j in ev]
    res["profile"] = L.ranking_metrics([profile_ranked(feats[j["image_id"]], q) for j in ev], truth)
    # learned: grouped CV over eval images; training rows = train-source judgements whose group is not in the fold
    gk = GroupKFold(n_splits=5)
    ev_groups = np.array([groups[j["image_id"]] for j in ev])
    for kind in ("linear", "forest", "mlp", "dinov2"):
        if kind == "dinov2" and emb is None:
            continue
        per_seed = []
        for seed in seeds:
            ranked = [None] * len(ev)
            for _, te in gk.split(np.zeros(len(ev)), groups=ev_groups):
                held = set(ev_groups[te])
                tr = [j for j in tr_all if groups[j["image_id"]] not in held]
                X = np.stack([emb[j["image_id"]] if kind == "dinov2" else feature_vector(feats[j["image_id"]]) for j in tr])
                y = [j["matches"][0] if j["matches"] and not j["no_match"] else "none" for j in tr]
                sc, m = L.fit("linear" if kind == "dinov2" else kind, X, y, seed=seed)
                for i in te:
                    j = ev[i]
                    x = emb[j["image_id"]] if kind == "dinov2" else feature_vector(feats[j["image_id"]])
                    pr = m.predict_proba(sc.transform(x[None]))[0]
                    ranked[i] = [str(m.classes_[k]) for k in np.argsort(-pr)] + \
                        [p for p in [*PROFILE_IDS, "none"] if p not in set(m.classes_)]
            per_seed.append(L.ranking_metrics(ranked, truth))
        res[kind] = {k: {"mean": float(np.mean([p[k] for p in per_seed])), "std": float(np.std([p[k] for p in per_seed]))}
                     for k in ("top1", "top3", "mrr")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", default="claude")
    ap.add_argument("--eval-source", default="human")
    ap.add_argument("--judgements", default=str(P.LABELS / "style_judgements.jsonl"))
    a = ap.parse_args()
    # compliance gate (docs/AI_DATA_POLICY.md): learned methods are TRAINED on --train-source judgements
    probs = policy_docs_status(P.ROOT)
    if probs:
        raise SystemExit("policy documents missing: " + "; ".join(probs))
    if a.train_source in THIRD_PARTY_SOURCES:
        status, basis = claude_training_permission(P.load_yaml("labeling.yaml")["claude"])
        if status != "permitted":
            raise SystemExit(f"--train-source {a.train_source} refused: training permission is {status} ({basis}). "
                             f"Use --train-source human, or record a written authorization reference.")
    js = read_jsonl(a.judgements)
    out = {"judgements": {s: sum(j["source"] == s for j in js) for s in ("human", "claude")}}
    if not js:
        out["status"] = ("BLOCKED: no independent style judgements exist yet (no human labels; Claude teacher not "
                         "run). Collect them with the review app (Style mode) or scripts/style_labels.py.")
        print(json.dumps(out, indent=1))
        (P.REPORTS / "style_evaluation.json").write_text(json.dumps(out, indent=1))
        return
    man = {r["image_id"]: r for r in read_jsonl(P.MANIFESTS / "manifest.jsonl")}
    F = json.loads((P.EMBEDDINGS / "style_features.json").read_text())
    feats = {iid: F[r["sha256"]] for iid, r in man.items()}
    groups = {iid: r.get("group_id", r["family_id"]) for iid, r in man.items()}
    q = json.loads((P.ROOT / "js/filmlook-style/model-data.js").read_text()
                   .split("module.exports = ", 1)[1].rstrip().rstrip(";"))["quantiles"]
    emb = None
    ep = P.EMBEDDINGS / "dinov2_small_224_ms1280.npz"
    if ep.exists():
        z = np.load(ep)
        by_sha = dict(zip(z["sha"].tolist(), z["emb"]))
        emb = {iid: by_sha[r["sha256"]] for iid, r in man.items() if r["sha256"] in by_sha}
    out["result"] = run(js, feats, groups, q, a.train_source, a.eval_source, emb)
    print(json.dumps(out, indent=1))
    (P.REPORTS / "style_evaluation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
