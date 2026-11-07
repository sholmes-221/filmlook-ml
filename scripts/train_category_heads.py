#!/usr/bin/env python
"""Train and evaluate photographer-style and film-aesthetic heads on frozen DINOv2-small embeddings.

  python scripts/train_category_heads.py              # readiness check, then train and evaluate if ready
  python scripts/train_category_heads.py --check      # readiness + leakage checks only

Runs the compliance preflight first. Uses only training_eligible annotations, the existing uploader-grouped
splits, and cached embeddings (data/embeddings/dinov2_small_224_ms1280.npz). It refuses to train a head whose
classes are too sparse, so it never reports numbers it could not measure. The headline numbers are the val/test
metrics; training-set accuracy is never reported. Output: reports/category_heads.json.
"""
import argparse
import json
import sys

import numpy as np

from filmlook_ml import paths as P
from filmlook_ml.annotation import heads as H
from filmlook_ml.annotation import qa
from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.workflow import AnnotationStore
from filmlook_ml.compliance.preflight import require_training_preflight
from filmlook_ml.datasets.preprocess import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    require_training_preflight()
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    recs = {r["image_id"]: r for r in man}
    anns = AnnotationStore(P.DATA / "annotations").load_all()
    splits = {i: r["split"] for i, r in recs.items()}
    # leakage: families (original + edits) share a split by construction; check near-duplicate originals across splits
    dups = qa.near_duplicates([recs[x.image_id] for x in anns])
    fam_split = {}
    leaks = 0
    for r in man:
        s = fam_split.setdefault(r["family_id"], r["split"])
        leaks += s != r["split"]
    rep = {"annotations": len(anns), "training_eligible": sum(x.training_eligible for x in anns),
           "needs_review": sum(x.qa.needs_review for x in anns),
           "leakage": {"families_split_across": leaks, "near_duplicate_pairs_cross_split": sum(d[3] for d in dups)},
           "heads": {}}
    for head, names in (("photographer", list(T.PHOTOGRAPHERS)), ("film", list(T.FILM_STOCKS))):
        r = H.readiness(anns, head, splits)
        rep["heads"][head] = {"readiness": r}
        if a.check or not r["ready"]:
            continue
        classes = r["trainable_classes"]
        emb = np.load(P.EMBEDDINGS / "dinov2_small_224_ms1280.npz")
        emb_by = dict(zip(emb["sha"], emb["emb"]))
        feat_by = json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text())
        d = H.build(anns, head, recs, emb_by, feat_by, classes)
        rep["heads"][head]["results"] = H.run_head(d, H.METHODS())
        rep["heads"][head]["classes"] = classes
    (P.REPORTS / "category_heads.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not any(h["readiness"]["ready"] for h in rep["heads"].values()):
        print("NOT READY: too few training-eligible labels per class; no model was trained.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
