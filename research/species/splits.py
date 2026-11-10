"""Near-duplicate removal and recorder-disjoint train / val / test splits, plus the unknown (rejection) sets.

  python -m research.species.splits [--test-frac 0.2 --val-frac 0.1]   -> data/species/dataset.jsonl, split_stats.json

A recorder (iNaturalist user / dataset recorder) lands in exactly one split for the whole dataset (hash of the
normalised recorder name), so no photographer appears on both sides. Exact (sha1) and near duplicates (pHash
Hamming distance <= 4) are dropped globally. Unknown sets: control taxa downloaded by collect.py (all held out) and
non-wildlife photos already on disk (PD12M CC0/PDM, CommonCatalog CC-BY-2.0, from data/manifests/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import imagehash
from PIL import Image

from research.species import common as C

DATASET = C.DATA / "dataset.jsonl"


def split_of(recorder: str, test_frac: float, val_frac: float) -> str:
    h = int(hashlib.sha256(recorder.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--nonwild", type=int, default=500)
    ap.add_argument("--nonwild-train", type=int, default=3000, help="further, disjoint non-wildlife photos for a 'none' training class")
    ap.add_argument("--phash-dist", type=int, default=4)
    a = ap.parse_args()
    taxa = {t["id"]: t for t in json.load(open(C.DATA / "taxa.json"))}
    unknown = {t["id"]: t for t in json.load(open(C.DATA / "unknown_taxa.json"))}
    rows = C.read_jsonl(C.DATA / "manifest.jsonl")
    print(f"{len(rows)} manifest rows")
    # duplicates
    seen_sha, hashes, kept, dropped = set(), [], [], Counter()
    for r in rows:
        if r["sha1"] in seen_sha:
            dropped["sha1"] += 1
            continue
        try:
            ph = imagehash.phash(Image.open(C.DATA / r["path"]))
        except Exception:  # noqa: BLE001
            dropped["unreadable"] += 1
            continue
        if any(ph - h <= a.phash_dist for h in hashes):
            dropped["phash"] += 1
            continue
        seen_sha.add(r["sha1"]); hashes.append(ph)
        kept.append(dict(r, phash=str(ph)))
    print("dropped", dict(dropped), "kept", len(kept))
    out = []
    for r in kept:
        sid = r["species_id"]
        if sid.startswith("unk_"):
            t = unknown[sid]
            out.append({**r, "split": "unknown_taxa", "genus": t.get("genus"), "family": t.get("family"), "canonical": t.get("canonical"), "relation": t.get("relation")})
        else:
            t = taxa[sid]
            out.append({**r, "split": split_of(r["recorder"], a.test_frac, a.val_frac), "genus": t.get("genus"), "family": t.get("family"), "canonical": t.get("canonical")})
    # non-wildlife unknowns from registry-approved local datasets
    rng = random.Random(0)
    nonwild = []
    for mf in ("raw_pd12m.jsonl", "raw_commoncatalog.jsonl"):
        for r in C.read_jsonl(C.ROOT / "data" / "manifests" / mf):
            if r.get("license") in ("CC0-1.0", "PDM-1.0", "CC-BY-2.0", "CC-BY-4.0") and os.path.exists(r["path"]):
                subj = " ".join(r.get("subjects") or []).lower() if isinstance(r.get("subjects"), list) else str(r.get("subjects") or "").lower()
                if any(w in subj for w in ("bird", "flower", "blossom", "duck", "owl", "heron", "orchid", "rose", "tulip", "garden")):
                    continue
                nonwild.append({"path": r["path"], "species_id": None, "split": "unknown_nonwild", "licence": r["license"], "source_url": r.get("source_url"),
                                "attribution": r.get("attribution"), "dataset": r.get("dataset_id")})
    rng.shuffle(nonwild)
    out += nonwild[: a.nonwild]
    out += [{**r, "split": "train_none"} for r in nonwild[a.nonwild: a.nonwild + a.nonwild_train]]
    with open(DATASET, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # statistics
    per = defaultdict(Counter)
    for r in out:
        if r["species_id"] and not r["species_id"].startswith("unk_"):
            per[r["species_id"]][r["split"]] += 1
    stats = {"n_rows": len(out), "dropped": dict(dropped), "splits": dict(Counter(r["split"] for r in out)),
             "classes_with_test": sum(1 for c in per.values() if c["test"] > 0), "classes_with_train": sum(1 for c in per.values() if c["train"] > 0),
             "classes_total": len(per), "per_class": {k: dict(v) for k, v in sorted(per.items())},
             "recorders": {s: len({r["recorder"] for r in out if r.get("recorder") and r["split"] == s}) for s in ("train", "val", "test")},
             "licences": dict(Counter(r.get("licence") for r in out))}
    json.dump(stats, open(C.DATA / "split_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
