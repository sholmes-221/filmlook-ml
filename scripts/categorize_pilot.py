#!/usr/bin/env python
"""Interactive categorization pilot (annotation_interface = claude_code_interactive).

A bounded pilot run inside an interactive Claude Code session at the project owner's request. Claude Code
reads each image with its built-in image reading, and every result goes through the same strict schema, QA and
provenance as the API path. Larger runs use scripts/annotate_categories.py (API key, budget cap) after the
owner approves them.

  python scripts/categorize_pilot.py select --n 20         # diverse eligible originals + full/crop views
  python scripts/categorize_pilot.py prompt                # the instructions the annotator follows
  python scripts/categorize_pilot.py record IMAGE_ID < result.json
  python scripts/categorize_pilot.py report                # QA, distributions, review sample -> reports/
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.annotation import qa
from filmlook_ml.annotation.schema import CategorizationResult
from filmlook_ml.annotation.workflow import (PROMPT_VERSION, AnnotationStore, build_annotation, discover,
                                             image_metadata, load_uploader_tags, system_prompt, validate_store)
from filmlook_ml.compliance.policy import claude_training_permission
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry
from filmlook_infer.features import center_crop_native

ANN = P.DATA / "annotations"
PILOT = ANN / "claude_exploratory" / "pilot"
MODEL_ID = "claude-opus-5-5"        # the model running this Claude Code session
FEATS = P.EMBEDDINGS / "features_ms1280_c512.json"


def _ctx():
    lcfg = P.load_yaml("labeling.yaml")
    ccfg = lcfg["claude"]
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, skipped = discover(man, Registry.load(), set(ccfg["consented_datasets"]), load_quarantine())
    return ccfg, man, eligible, skipped


def stratum(r):
    if r["dataset_id"] == "pd12m_pilot":
        return "pd12m"
    return r.get("medium") or "unknown"


def cmd_select(a):
    ccfg, man, eligible, skipped = _ctx()
    usertags = load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl")
    quota = {"film_scan": 6, "digital_camera": 4, "smartphone": 2, "smartphone_filtered": 2, "digital_filtered": 2,
             "pd12m": 4}
    rng = random.Random(a.seed)
    by = defaultdict(list)
    for r in eligible:
        by[stratum(r)].append(r)
    chosen, groups = [], set()
    for s, k in quota.items():
        pool = by.get(s, [])[:]
        rng.shuffle(pool)
        # film scans: prefer two with an uploader film-stock tag (weak check of the inferred look)
        if s == "film_scan":
            pool.sort(key=lambda r: not image_metadata(r, usertags).film_stock_metadata)
            pool = pool[:2] + rng.sample(pool[2:], len(pool) - 2)
        subj_seen = set()
        for r in pool:
            if len([c for c in chosen if stratum(c) == s]) >= k:
                break
            subj = tuple(r.get("subjects") or [])
            if (s != "pd12m" and r.get("group_id") in groups) or (subj and subj in subj_seen and len(pool) > 3 * k):
                continue
            chosen.append(r)
            groups.add(r.get("group_id"))
            subj_seen.add(subj)
    chosen = chosen[: a.n]
    PILOT.mkdir(parents=True, exist_ok=True)
    plan = []
    for r in chosen:
        img = load_image(r["path"], max_side=4096)          # same validating loader as training / inference
        full = img.copy()
        full.thumbnail((1024, 1024), Image.LANCZOS)
        crop = center_crop_native(img, 384)
        fp, cp = PILOT / f"{r['image_id']}_full.jpg", PILOT / f"{r['image_id']}_crop.jpg"
        full.save(fp, quality=92)
        crop.save(cp, quality=95)
        meta = image_metadata(r, usertags)
        plan.append({"image_id": r["image_id"], "stratum": stratum(r), "split": r["split"], "subjects": r.get("subjects"),
                     "size": list(img.size), "full_view": str(fp), "crop_view": str(cp),
                     "uploader_stock_tags": [m.value for m in meta.film_stock_metadata]})
    (PILOT / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({"eligible": len(eligible), "skipped": dict(skipped), "selected": len(plan),
                      "by_stratum": {s: sum(p["stratum"] == s for p in plan) for s in quota}}, indent=1))
    for p in plan:
        print(p["image_id"], p["stratum"], p["split"], p["size"], p["uploader_stock_tags"])


def cmd_prompt(a):
    print(system_prompt())


def cmd_record(a):
    ccfg, man, eligible, _ = _ctx()
    rec = {r["image_id"]: r for r in eligible}.get(a.image_id)
    if rec is None:
        raise SystemExit(f"{a.image_id} is not an eligible original")
    plan = {p["image_id"] for p in json.loads((PILOT / "plan.json").read_text())}
    if a.image_id not in plan:
        raise SystemExit(f"{a.image_id} is not in the pilot plan; the pilot is bounded to the selected images")
    store = AnnotationStore(ANN)
    if store.has(a.image_id):
        raise SystemExit(f"{a.image_id} already annotated; annotations are never overwritten")
    result = CategorizationResult.model_validate_json(sys.stdin.read())     # strict: raises on any violation
    feats = json.loads(FEATS.read_text())
    usertags = load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl")
    prog = store.dir / "progress.jsonl"
    last = None
    if prog.exists():
        for line in prog.read_text().splitlines():
            row = json.loads(line)
            if row.get("interface") == "claude_code_interactive":
                last = row["ts"]
    from datetime import datetime
    elapsed = round(time.time() - datetime.fromisoformat(last).timestamp(), 1) if last else None
    ann = build_annotation(result, rec, interface="claude_code_interactive", ccfg=ccfg, model_id=MODEL_ID,
                           features=feats.get(rec["sha256"]), metadata=image_metadata(rec, usertags),
                           request={"model": MODEL_ID, "interface": "Claude Code interactive session (Read tool)",
                                    "elapsed_s_since_previous_record": elapsed, "cost_usd": None, "tokens": None,
                                    "cost_note": "per-image tokens and cost are not observable from inside the session"},
                           root=P.ROOT)
    store.write(ann)
    store.progress(image_id=a.image_id, status="categorized", interface="claude_code_interactive", flags=ann.qa.flags)
    print(json.dumps({"image_id": a.image_id, "flags": ann.qa.flags, "needs_review": ann.qa.needs_review,
                      "training_eligible": ann.training_eligible}))


def cmd_start(a):
    AnnotationStore(ANN).progress(event="pilot_start", interface="claude_code_interactive", prompt_version=PROMPT_VERSION)


def cmd_report(a):
    ccfg, man, eligible, skipped = _ctx()
    store = AnnotationStore(ANN)
    anns = [x for x in store.load_all() if x.annotation_interface == "claude_code_interactive"]
    plan = json.loads((PILOT / "plan.json").read_text())
    byid = {r["image_id"]: r for r in man}
    ann_recs = [byid[x.image_id] for x in anns]
    dups_pilot = qa.near_duplicates(ann_recs)
    dups_all = qa.near_duplicates([r for r in man if r["edit_type"] == "original"])
    times = [x.request.get("elapsed_s_since_previous_record") for x in anns]
    times = [t for t in times if t is not None]
    status, basis = claude_training_permission(ccfg)
    low = [{"image_id": x.image_id, "photographer": [x.photographer_style.primary, x.photographer_style.confidence],
            "film": [x.film_aesthetic.primary, x.film_aesthetic.confidence], "flags": x.qa.flags}
           for x in anns if x.qa.needs_review]
    stock_check = [{"image_id": x.image_id, "uploader_tag": [m.value for m in x.metadata.film_stock_metadata],
                    "inferred": [x.film_aesthetic.primary, *x.film_aesthetic.alternatives]}
                   for x in anns if x.metadata.film_stock_metadata]
    rep = {"pilot_selected": len(plan), "annotated": len(anns), "validation": validate_store(store),
           "eligible_total": len(eligible), "skipped": dict(skipped), "distributions": qa.distributions(anns),
           "needs_review": low, "review_sample": qa.review_sample(anns), "uploader_stock_tag_check": stock_check,
           "near_duplicates_in_pilot": dups_pilot,
           "near_duplicate_originals_all": {"pairs": len(dups_all), "cross_split": sum(d[3] for d in dups_all)},
           "timing_s": {"median_between_records": sorted(times)[len(times) // 2] if times else None,
                        "total_recorded": round(sum(times), 1)},
           "cost": "not observable from an interactive Claude Code session (no per-request usage)",
           "training_permission": {"status": status, "basis": basis},
           "by_stratum": {s: sum(1 for x in anns for p in plan if p["image_id"] == x.image_id and p["stratum"] == s)
                          for s in sorted({p['stratum'] for p in plan})}}
    out = P.REPORTS / "categorization_pilot.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)
    sub.add_parser("prompt")
    sub.add_parser("start")
    r = sub.add_parser("record")
    r.add_argument("image_id")
    sub.add_parser("report")
    a = ap.parse_args()
    {"select": cmd_select, "prompt": cmd_prompt, "start": cmd_start, "record": cmd_record, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
