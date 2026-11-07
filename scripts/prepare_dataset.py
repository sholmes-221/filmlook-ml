#!/usr/bin/env python
"""Validate, license-gate, dedupe, create edit families, and split by GROUP.

Group = uploader for Flickr photos (same photographer/lab/scanner never spans splits), else the photo
family. Every edit of a photo stays in its original's split."""
import argparse
import json

from filmlook_ml import paths as P
from filmlook_ml.datasets.edits import EDITS
from filmlook_ml.datasets.preprocess import group_split, prepare, read_jsonl, write_jsonl
from filmlook_ml.datasets.registry import Registry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["synthetic", "pd12m", "commoncatalog"])
    ap.add_argument("--edits", nargs="*", default=list(EDITS))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = P.load_yaml("training.yaml")
    raw = []
    for s in a.sources:
        raw += read_jsonl(P.MANIFESTS / f"raw_{s}.jsonl")
    print(f"{len(raw)} raw records from {a.sources}")
    manifest, rejected = prepare(raw, Registry.load(), P.PROCESSED, a.edits, a.max_side, seed=a.seed,
                                 workers=a.workers)
    split = group_split(manifest, cfg["splits"], a.seed, group_key="group_id")
    for r in manifest:
        r["split"] = split[r["image_id"]]
    write_jsonl(P.MANIFESTS / "manifest.jsonl", manifest)
    write_jsonl(P.QUARANTINE / "rejected.jsonl", rejected)
    P.SPLITS.mkdir(parents=True, exist_ok=True)
    (P.SPLITS / "splits.json").write_text(json.dumps(split))
    fams = {r["family_id"]: r["split"] for r in manifest}
    summary = {"images": len(manifest), "families": len(fams), "rejected": len(rejected),
               "by_split_families": {s: sum(v == s for v in fams.values()) for s in ("train", "val", "test")},
               "by_dataset": {d: sum(r["dataset_id"] == d and r["edit_type"] == "original" for r in manifest)
                              for d in sorted({r["dataset_id"] for r in manifest})},
               "by_medium": {m: sum(r.get("medium") == m and r["edit_type"] == "original" for r in manifest)
                             for m in sorted({str(r.get("medium")) for r in manifest})},
               "groups": len({r["group_id"] for r in manifest}),
               "reject_reasons": {}}
    for r in rejected:
        k = r["reason"].split(":")[0]
        summary["reject_reasons"][k] = summary["reject_reasons"].get(k, 0) + 1
    (P.MANIFESTS / "prepare_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
