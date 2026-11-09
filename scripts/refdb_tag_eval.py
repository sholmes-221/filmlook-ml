#!/usr/bin/env python
"""How good is zero-shot tag detection? Precision and recall against weak labels from catalogue titles.

  python scripts/refdb_tag_eval.py

A catalogue title such as "Coal miners, Birmingham, Alabama" is an archivist's record, independent of every model
here. For tags with a clear title vocabulary, a work counts as *positive* when its title contains one of the words
and *negative* otherwise. Titles omit most of what a photo shows, so recall is a lower-bound-ish figure (negatives
include unmentioned positives) and precision is a lower bound (a true detection on an unmentioned object counts as a
false positive). The numbers rank tags by reliability; they are not calibrated accuracies.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.visualdna.semantic import MAX_TAGS, Z_MIN  # noqa: E402

TITLE_WORDS = {
    "sign": r"\bsigns?\b|\bbillboards?\b", "street": r"\bstreets?\b|\bmain street\b", "alley": r"\balley", "child": r"\bchild|\bchildren\b|\bboys?\b|\bgirls?\b|\bbaby\b",
    "worker": r"\bworkers?\b|\blaborers?\b|\bminers?\b|\bpickers?\b", "farm": r"\bfarms?\b|\bfarmers?\b|\bsharecropper", "mill": r"\bmills?\b", "mine": r"\bmines?\b|\bminers?\b|\bcoal\b",
    "car": r"\bcars?\b|\bautomobiles?\b|\btrucks?\b", "train": r"\btrains?\b|\blocomotives?\b|\brailroad\b|\bfreight\b", "camp": r"\bcamps?\b", "dust storm": r"\bdust storm",
    "snow": r"\bsnow", "harbour": r"\bharbou?r\b|\bdocks?\b|\bwharf", "storefront": r"\bstores?\b|\bshops?\b|\bstorefront", "factory": r"\bfactory|\bplant\b|\bworks\b",
    "vernacular house": r"\bhouses?\b|\bhomes?\b|\bshack|\bcabin", "crowd": r"\bcrowd|\bparade|\bmeeting\b", "interior": r"\binterior|\bkitchen|\bliving room|\bbedroom",
    "railway yard": r"\byards?\b.*\brailroad|\brailroad yard|\broundhouse", "horse": r"\bhorses?\b|\bmules?\b", "church": r"$^",
}


def main() -> None:
    z = np.load(P.DATA / "refdb" / "siglip_v1.npz", allow_pickle=True)
    stats = json.loads((P.DATA / "refdb" / "siglip_tag_stats.json").read_text())
    tags = list(z["tag_names"]); mu = np.array([stats[t][0] for t in tags]); sd = np.array([stats[t][1] for t in tags])
    Z = (z["tag_logits"] - mu) / sd
    det = np.zeros_like(Z, bool)
    for i in range(len(Z)):
        for j in np.argsort(-Z[i])[:MAX_TAGS]:
            if Z[i, j] >= Z_MIN:
                det[i, j] = True
    c = sqlite3.connect(P.DATA / "refdb" / "manifest.sqlite")
    title = dict(c.execute("SELECT work_id, COALESCE(title, '') FROM works"))
    titles = [title.get(w, "").lower() for w in z["work_id"]]
    res = {}
    for tag, pat in TITLE_WORDS.items():
        if tag not in tags or pat == "$^":
            continue
        j = tags.index(tag)
        pos = np.array([bool(re.search(pat, t)) for t in titles])
        if pos.sum() < 5:
            continue
        tp = int((det[:, j] & pos).sum()); fp = int((det[:, j] & ~pos).sum()); fn = int((~det[:, j] & pos).sum())
        res[tag] = {"positives": int(pos.sum()), "detected": int(det[:, j].sum()), "precision_lower_bound": round(tp / max(tp + fp, 1), 3),
                    "recall_vs_title": round(tp / max(tp + fn, 1), 3), "base_rate": round(float(pos.mean()), 3)}
    out = {"n_images": int(len(Z)), "rule": f"z >= {Z_MIN}, top {MAX_TAGS}", "tags": dict(sorted(res.items(), key=lambda kv: -kv[1]["precision_lower_bound"]))}
    (P.REPORTS / "refdb" / "tag_eval.json").write_text(json.dumps(out, indent=1))
    for k, v in out["tags"].items():
        print(f"{k:18s} pos {v['positives']:4d} det {v['detected']:4d} P>= {v['precision_lower_bound']:.2f} R {v['recall_vs_title']:.2f} base {v['base_rate']:.2f}")


if __name__ == "__main__":
    main()
