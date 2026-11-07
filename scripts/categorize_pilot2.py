#!/usr/bin/env python
"""Second categorization pilot: 30 images, schema categorization-3.0, annotator BLIND to metadata.

Blinding: `select` writes
  pilot2/blind/P2_NN.jpg      one composite per image (full view + native-resolution 384 px crop), pixels only
  pilot2/blind_plan.json      blind ids and composite paths only
  pilot2/sealed.json          blind id -> image_id + metadata (uploader tags, EXIF capture type, stratum)
and prints only counts. The annotator reads the composites and `record`s results by blind id; `record` prints
no metadata. `sealed.json` is read only by this script and, after the last record, by `report`.

  python scripts/categorize_pilot2.py select --n 30
  python scripts/categorize_pilot2.py record P2_01 < result.json
  python scripts/categorize_pilot2.py report
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

from filmlook_infer.features import center_crop_native
from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.schema_v3 import CategorizationV3
from filmlook_ml.annotation.v3 import StoreV3, build_v3, metadata_v3
from filmlook_ml.annotation.workflow import AnnotationStore, discover, load_uploader_tags
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry

ANN = P.DATA / "annotations"
DIR = ANN / "claude_exploratory" / "pilot2"
MODEL_ID = "claude-opus-5-5"
PILOT = "pilot2"
QUOTA = {"film_scan": 9, "digital_camera": 6, "smartphone": 4, "smartphone_filtered": 3, "digital_filtered": 3,
         "pd12m": 5}


def _ctx():
    lcfg = P.load_yaml("labeling.yaml")
    ccfg = lcfg["claude"]
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, skipped = discover(man, Registry.load(), set(ccfg["consented_datasets"]), load_quarantine())
    return ccfg, man, eligible, skipped


def _stratum(r):
    return "pd12m" if r["dataset_id"] == "pd12m_pilot" else (r.get("medium") or "unknown")


def composite(img: Image.Image) -> Image.Image:
    full = img.copy()
    full.thumbnail((1024, 1024), Image.LANCZOS)
    crop = center_crop_native(img, 384)
    canvas = Image.new("RGB", (full.width + 16 + 384, max(full.height, 384)), (40, 40, 40))
    canvas.paste(full, (0, 0))
    canvas.paste(crop, (full.width + 16, 0))
    return canvas


def cmd_select(a):
    ccfg, man, eligible, skipped = _ctx()
    done_p1 = {x.image_id for x in AnnotationStore(ANN).load_all()}
    usertags = load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl")
    pool = [r for r in eligible if r["image_id"] not in done_p1]
    rng = random.Random(a.seed)
    by = defaultdict(list)
    for r in pool:
        by[_stratum(r)].append(r)
    chosen, groups = [], set()
    for s, k in QUOTA.items():
        cand = by.get(s, [])[:]
        rng.shuffle(cand)
        if s == "film_scan":   # include some uploader-tagged scans so tag agreement can be measured afterwards
            tagged = [r for r in cand if metadata_v3(r, usertags).uploader_asserted_film_stock]
            cand = tagged[:4] + [r for r in cand if r not in tagged[:4]]
        subj_seen, n = Counter(), 0
        for r in cand:
            if n >= k:
                break
            subj = tuple(r.get("subjects") or [])
            if (s != "pd12m" and r.get("group_id") in groups) or (subj and subj_seen[subj] >= 2):
                continue
            chosen.append(r)
            groups.add(r.get("group_id"))
            subj_seen[subj] += 1
            n += 1
    rng.shuffle(chosen)                       # blind ids carry no stratum order
    (DIR / "blind").mkdir(parents=True, exist_ok=True)
    plan, sealed = [], {}
    for i, r in enumerate(chosen[: a.n], 1):
        bid = f"P2_{i:02d}"
        img = load_image(r["path"], max_side=4096)     # same validating loader as training (EXIF transposed)
        cp = DIR / "blind" / f"{bid}.jpg"
        composite(img).save(cp, quality=92)
        plan.append({"blind_id": bid, "composite": str(cp)})
        meta = metadata_v3(r, usertags)
        sealed[bid] = {"image_id": r["image_id"], "stratum": _stratum(r), "split": r["split"],
                       "metadata": meta.model_dump()}
    (DIR / "blind_plan.json").write_text(json.dumps(plan, indent=1))
    (DIR / "sealed.json").write_text(json.dumps(sealed, indent=1))
    print(json.dumps({"eligible": len(eligible), "skipped": dict(skipped), "excluded_pilot1": len(done_p1),
                      "selected": len(plan)}, indent=1))


def cmd_record(a):
    ccfg, man, eligible, _ = _ctx()
    sealed = json.loads((DIR / "sealed.json").read_text())
    if a.blind_id not in sealed:
        raise SystemExit(f"unknown blind id {a.blind_id}")
    rec = {r["image_id"]: r for r in eligible}[sealed[a.blind_id]["image_id"]]
    store = StoreV3(ANN)
    if store.has(rec["image_id"]):
        raise SystemExit(f"{a.blind_id} already recorded; annotations are never overwritten")
    res = CategorizationV3.model_validate_json(sys.stdin.read())       # strict: raises on any violation
    feats = json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text())
    log = DIR / "timing.jsonl"
    prev = None
    if log.exists():
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        prev = rows[-1]["ts"] if rows else None
    now = time.time()
    elapsed = round(now - prev, 1) if prev else None
    usertags = load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl")
    ann = build_v3(res, rec, interface="claude_code_interactive", ccfg=ccfg, model_id=MODEL_ID,
                   features=feats.get(rec["sha256"]), metadata=metadata_v3(rec, usertags), root=P.ROOT, blind=True,
                   pilot=PILOT, request={"model": MODEL_ID, "interface": "Claude Code interactive session (Read tool)",
                                         "blind_id": a.blind_id, "elapsed_s_since_previous_record": elapsed,
                                         "cost_usd": None, "tokens": None,
                                         "cost_note": "per-image usage is not observable inside the session"})
    store.write(ann)
    with open(log, "a") as fh:
        fh.write(json.dumps({"ts": now, "blind_id": a.blind_id}) + "\n")
    print(f"recorded {a.blind_id}")             # nothing else: keeps the annotator blind to metadata and QA


def cmd_start(a):
    with open(DIR / "timing.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.time(), "blind_id": "start"}) + "\n")


def _agree_rate(pairs):
    pairs = [p for p in pairs if p is not None]
    return {"n": len(pairs), "agree": sum(pairs), "rate": round(sum(pairs) / len(pairs), 3) if pairs else None}


def cmd_report(a):
    sealed = json.loads((DIR / "sealed.json").read_text())
    anns = [x for x in StoreV3(ANN).load_all() if x.pilot == PILOT]
    by_id = {x.image_id: x for x in anns}
    rows = [(bid, s, by_id.get(s["image_id"])) for bid, s in sealed.items()]
    missing = [bid for bid, _, x in rows if x is None]
    rows = [(b, s, x) for b, s, x in rows if x is not None]

    # attribute consistency with deterministic measurements (only where the measurement is informative)
    def m(x):
        return x.qa.measured
    cons = defaultdict(list)
    for _, _, x in rows:
        at, me = x.result.attributes, m(x)
        if not me:
            continue
        tags = set(me["tags"])
        mono = at.color.saturation == "monochrome"
        cons["monochrome"].append(mono == me["is_mono"])
        if not mono and not me["is_mono"]:
            if {"warm", "cool"} & tags:
                cons["temperature_extremes"].append(at.color.temperature in tags)
            if {"muted", "vivid"} & tags:
                cons["saturation_extremes"].append((at.color.saturation == "low") == ("muted" in tags))
        if {"high_contrast", "low_contrast"} & tags:
            cons["contrast_extremes"].append((at.tone.contrast == "high") == ("high_contrast" in tags)
                                             if at.tone.contrast != "medium" else None)
        if {"no_visible_grain", "coarse_grain"} & tags:
            g = at.texture.visible_grain
            cons["grain_extremes"].append((g == "none") == ("no_visible_grain" in tags) if g != "subtle" else None)
        if {"high_key", "low_key"} & tags:
            cons["key_extremes"].append(at.lighting.key in tags if at.lighting.key != "balanced" else None)

    # photographer ranking usefulness
    ph = [x.result.photographer for _, _, x in rows]
    analyzable = [p for p in ph if p.analyzable]
    supported = [p for p in analyzable if p.style_match_supported]
    top1_sims = [p.top_3_photographers[0].similarity for p in analyzable]
    # retrieval sanity: do DINOv2 neighbours share the closest photographer more often than chance?
    emb = np.load(P.EMBEDDINGS / "dinov2_small_224_ms1280.npz")
    E = dict(zip(emb["sha"], emb["emb"]))
    ids = [x for _, _, x in rows if x.result.photographer.analyzable and x.sha256 in E]
    nn_agree, chance = None, None
    if len(ids) > 3:
        Z = np.stack([E[x.sha256] for x in ids])
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ Z.T
        np.fill_diagonal(S, -1)
        lab = [x.result.photographer.closest_photographer for x in ids]
        nn = S.argmax(1)
        nn_agree = round(float(np.mean([lab[i] == lab[j] for i, j in enumerate(nn)])), 3)
        c = Counter(lab)
        chance = round(sum((v / len(lab)) * ((v - 1) / (len(lab) - 1)) for v in c.values()), 3)

    fi = [x.result.film for _, _, x in rows]
    # agreement with metadata (unsealed now): EXIF capture type vs apparent capture; uploader stock tags vs top 3
    cap_pairs, cap_conf = [], Counter()
    for _, s, x in rows:
        exif = s["metadata"]["capture_type_exif"]
        app = x.result.film.apparent_capture
        cap_conf[(exif or "none(pd12m)", app)] += 1
        if exif == "film_scan":
            cap_pairs.append(app == "film_look" if app != "indeterminate" else None)
        elif exif in ("digital_camera", "smartphone"):
            cap_pairs.append(app == "digital_look" if app != "indeterminate" else None)
        elif exif in ("digital_filtered", "smartphone_filtered"):
            cap_pairs.append(app in ("digital_filter_look", "digital_look") if app != "indeterminate" else None)
    tag_check = []
    for bid, s, x in rows:
        tags = [t["value"] for t in s["metadata"]["uploader_asserted_film_stock"]]
        if tags and x.result.film.analyzable:
            top3 = [r.name for r in x.result.film.top_3_film_stocks]
            tag_check.append({"blind_id": bid, "uploader_tag": tags, "top3": top3,
                              "tag_in_top1": top3[0] in tags, "tag_in_top3": bool(set(tags) & set(top3)),
                              "stock_match_supported": x.result.film.stock_match_supported})
    flags = Counter(f for _, _, x in rows for f in x.qa.flags)
    timing = [json.loads(t) for t in (DIR / "timing.jsonl").read_text().splitlines()]
    gaps = [b["ts"] - a_["ts"] for a_, b in zip(timing, timing[1:])]
    rep = {
        "selected": len(sealed), "annotated": len(rows), "missing": missing,
        "schema_valid": len(rows), "blind_to_metadata": all(x.blind_to_metadata for _, _, x in rows),
        "by_stratum": dict(Counter(s["stratum"] for _, s, _ in rows)),
        "attribute_consistency_with_measurement": {k: _agree_rate(v) for k, v in cons.items()},
        "photographer": {
            "analyzable": len(analyzable), "style_match_supported": len(supported),
            "none_of_the_above": sum(p.none_of_the_above for p in analyzable),
            "closest_distribution": dict(Counter(p.closest_photographer for p in analyzable).most_common()),
            "supported_distribution": dict(Counter(p.closest_photographer for p in supported).most_common()),
            "candidates_appearing_in_top3": dict(Counter(r.name for p in analyzable for r in p.top_3_photographers)),
            "top1_similarity": {"min": min(top1_sims), "median": float(np.median(top1_sims)), "max": max(top1_sims)}
            if top1_sims else None,
            "dinov2_nearest_neighbour_shares_closest": nn_agree, "chance_level": chance},
        "film": {
            "analyzable": sum(f.analyzable for f in fi), "stock_match_supported": sum(f.stock_match_supported for f in fi),
            "category_distribution": dict(Counter(f.aesthetic_category for f in fi).most_common()),
            "top1_stock_distribution": dict(Counter(f.top_3_film_stocks[0].name for f in fi if f.analyzable).most_common()),
            "apparent_capture": dict(Counter(f.apparent_capture for f in fi))},
        "unknown_frequency": {"photographer_not_analyzable": sum(not p.analyzable for p in ph),
                              "film_category_unknown": sum(f.aesthetic_category == "unknown_indeterminate" for f in fi)},
        "metadata_agreement": {"exif_capture_vs_apparent": _agree_rate(cap_pairs),
                               "confusion": {f"{k[0]} -> {k[1]}": v for k, v in sorted(cap_conf.items())},
                               "uploader_stock_tags": tag_check},
        "qa_flags": dict(flags.most_common()),
        "needs_review": sum(x.qa.needs_review for _, _, x in rows),
        "attributes_training_eligible": sum(x.attributes_training_eligible for _, _, x in rows),
        "timing_s": {"median_per_image": round(float(np.median(gaps)), 1) if gaps else None,
                     "total": round(sum(gaps), 1)},
        "cost": "not observable inside an interactive Claude Code session",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (P.REPORTS / "categorization_pilot2.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=11)
    sub.add_parser("start")
    r = sub.add_parser("record")
    r.add_argument("blind_id")
    sub.add_parser("report")
    a = ap.parse_args()
    {"select": cmd_select, "start": cmd_start, "record": cmd_record, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
