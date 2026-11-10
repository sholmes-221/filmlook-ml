"""Deployable species head on the production embedding, evaluated per badge group, and exported for the website.

  python -m research.species.badge_heads [--min-precision 0.95] [--ship-precision 0.90] [--ship-lb 0.80] [--min-recall 0.30]
      -> data/species/eval/badge_heads.json (metrics), dist/species/heads-species-<date>-<hash>.json (website artefact)

Model: multinomial logistic regression on data/species/feats/onnx768.npz (the website's photo-gate encoder), C chosen on
val top-1, temperature fitted on val NLL. Reported on the recorder-disjoint test split: species / genus / family top-1 and
top-5 (genus and family = summed probabilities). Badge outputs: each group in badge_groups.GROUPS (summed probability of
its member classes) plus kind-level "birds" / "flowers". Threshold = the lowest one whose validation precision reaches
--min-precision; the output ships only if on the untouched test half it keeps precision >= --ship-precision, Wilson 95 %
lower bound >= --ship-lb and recall >= --min-recall with at least 8 test positives.

Negatives for a group: every encyclopedia row outside the group, non-wildlife photos and unknown taxa from other families.
Unknown taxa from a member's own family are ignored for that group (a different gull species predicted as "gull" is not a
false badge, and is not counted as a hit either). Unknown and non-wildlife rows are split in half by recorder / path hash:
one half for threshold selection, the other for the test.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from research.species import common as C
from research.species import data as D
from research.species.badge_groups import GROUPS

DIST = Path(__file__).resolve().parents[2] / "dist" / "species"


def wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def half(key: str) -> str:
    return "val" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


def softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--ship-precision", type=float, default=0.90)
    ap.add_argument("--ship-lb", type=float, default=0.80)
    ap.add_argument("--min-recall", type=float, default=0.30)
    a = ap.parse_args()
    rows = D.load_rows()
    emb = np.load(C.DATA / "feats" / "onnx768.npz")["emb"]
    assert len(emb) == len(rows), "re-run onnx_features after splits"
    ls = D.label_space(rows); classes = ls["classes"]; index = ls["index"]
    for r in rows:
        if r["split"] in ("unknown_taxa", "unknown_nonwild"):
            r["part"] = half(r.get("recorder") or r["path"])
        else:
            r["part"] = r["split"]
    def sel(part, labelled=True):
        ids = [r["idx"] for r in rows if r["part"] == part and (not labelled or (r["split"] in ("train", "val", "test") and r["species_id"] in index))]
        return np.array(ids, int)
    tr, va, te = sel("train"), sel("val"), sel("test")
    NONE = len(classes)  # a background class trained on further non-wildlife photos (split train_none), never an output
    y = np.array([index.get(r["species_id"], -1) if r["split"] in ("train", "val", "test") else (NONE if r["split"] == "train_none" else -1) for r in rows])
    tr = np.concatenate([tr, np.array([r["idx"] for r in rows if r["split"] == "train_none"], int)])
    best = None
    for c in (1.0, 3.0, 10.0, 30.0):
        m = LogisticRegression(C=c, max_iter=3000, n_jobs=16)
        m.fit(emb[tr], y[tr])
        acc = float((m.classes_[m.decision_function(emb[va]).argmax(1)] == y[va]).mean())
        print(f"C={c}: val top-1 {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, c, m)
    _, c_best, m = best
    # full weight matrix over all classes (classes absent from train cannot be predicted)
    W = np.zeros((len(classes) + 1, 768), np.float32); b = np.full(len(classes) + 1, -30.0, np.float32)
    W[m.classes_] = m.coef_; b[m.classes_] = m.intercept_
    logits = emb @ W.T + b
    temps = np.exp(np.linspace(np.log(0.3), np.log(5), 40))
    nll = [-np.log(softmax(logits[va], t)[np.arange(len(va)), y[va]] + 1e-12).mean() for t in temps]
    T = float(temps[int(np.argmin(nll))])
    P = softmax(logits, T)

    # hierarchical accuracy on test
    def topk(idx, k, keyfn):
        groups = defaultdict(list)
        for ci, cl in enumerate(classes):
            groups[keyfn(cl)].append(ci)
        names = list(groups)
        G = np.stack([P[idx][:, groups[n]].sum(1) for n in names], 1)
        truth = [keyfn(classes[y[i]]) for i in idx]
        order = np.argsort(-G, 1)[:, :k]
        return float(np.mean([truth[j] in {names[o] for o in order[j]} for j in range(len(idx))]))
    hier = {lvl: {"top1": topk(te, 1, f), "top5": topk(te, 5, f)} for lvl, f in
            (("species", lambda c: c["species_id"]), ("genus", lambda c: c["genus"]), ("family", lambda c: c["family"]))}
    print("test", json.dumps(hier), flush=True)

    # badge outputs
    member_fam = {g: {classes[index[s]]["family"] for s in ids if s in index} for g, ids in GROUPS.items()}
    outs, decisions = [], {}
    defs = [(g, f"species:{g}", [index[s] for s in ids if s in index]) for g, ids in GROUPS.items()]
    defs += [("birds", "birds", [i for i, c in enumerate(classes) if c["kind"] == "bird"]),
             ("flowers", "flowers", [i for i, c in enumerate(classes) if c["kind"] == "flower"])]
    for name, signal, members in defs:
        if not members:
            decisions[name] = {"decision": "unsupported", "why": "no member class with training images"}; continue
        mset = set(members)
        def truth(r):
            if r["split"] == "unknown_nonwild":
                return False
            if name in ("birds", "flowers"):
                return r.get("kind") == name[:-1]
            if r["split"] == "unknown_taxa":
                return None if r.get("family") in member_fam[name] else False
            return index.get(r["species_id"]) in mset
        score = P[:, members].sum(1)
        def pr(part, thr):
            tp = fp = fn = 0
            for r in rows:
                if r["part"] != part:
                    continue
                t = truth(r)
                if t is None:
                    continue
                hit = bool(score[r["idx"]] >= thr)
                tp += hit and t; fp += hit and not t; fn += (not hit) and t
            return int(tp), int(fp), int(fn)
        thr = None
        for cand in np.linspace(0.30, 0.995, 140):
            tp, fp, fn = pr("val", cand)
            if tp + fp >= 3 and tp / (tp + fp) >= a.min_precision:
                thr = float(cand); break
        if thr is None:
            decisions[name] = {"decision": "unsupported", "why": f"validation precision never reached {a.min_precision}"}; continue
        tp, fp, fn = pr("test", thr)
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0; lb = wilson_lb(tp, tp + fp)
        ok = tp + fn >= 8 and prec >= a.ship_precision and lb >= a.ship_lb and rec >= a.min_recall
        decisions[name] = {"decision": "ship" if ok else "unsupported", "threshold": round(thr, 4), "test": {"tp": tp, "fp": fp, "fn": fn,
                           "precision": round(prec, 3), "precision_wilson_lb": round(lb, 3), "recall": round(rec, 3)}}
        if ok:
            outs.append({"signal": signal, "classes": [int(i) for i in members], "threshold": round(thr, 4), "test": decisions[name]["test"]})
        print(f"{name:16s} {decisions[name]['decision']:11s} thr {thr:.3f} test P {prec:.3f} (lb {lb:.3f}) R {rec:.3f} tp {tp} fp {fp} fn {fn}", flush=True)

    date = datetime.date.today().strftime("%Y%m%d")
    art = {"embedding": "dinov2s-cls-patch-768", "kind": "species", "activation": "softmax", "temperature": T,
           "classes": [c["species_id"] for c in classes] + ["__none__"], "weights": np.round(W, 6).tolist(), "bias": np.round(b, 6).tolist(), "outputs": outs,
           "trained": {"model": "multinomial logistic regression", "C": c_best, "train": len(tr), "val": len(va), "test": len(te),
                       "encoder": "website models/photo_gate/dinov2_small_cls_patch.onnx", "data": "GBIF occurrence images, CC0 / CC BY, recorder-disjoint splits"},
           "test_hierarchical": hier}
    blob = json.dumps(art, sort_keys=True).encode()
    art["version"] = f"species-{date}-{hashlib.sha256(blob).hexdigest()[:8]}"
    DIST.mkdir(parents=True, exist_ok=True)
    path = DIST / f"heads-{art['version']}.json"
    path.write_text(json.dumps(art))
    (C.DATA / "eval").mkdir(exist_ok=True)
    (C.DATA / "eval" / "badge_heads.json").write_text(json.dumps({"version": art["version"], "C": c_best, "temperature": T, "hierarchical_test": hier,
                                                                  "decisions": decisions, "artefact": str(path)}, indent=1))
    print("wrote", path, "shipped outputs:", len(outs))


if __name__ == "__main__":
    main()
