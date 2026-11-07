#!/usr/bin/env python
"""Select a diverse, license-checked pilot from CommonCatalog candidates and fetch ONLY those images
(parquet page range reads from the Hugging Face Hub; no Flickr scraping).

  python scripts/download_commoncatalog.py --plan        # print the selection, download nothing
  python scripts/download_commoncatalog.py               # fetch (default caps ~800 originals)

Writes data/raw/commoncatalog/<photoid>.jpg, data/manifests/raw_commoncatalog.jsonl and
data/raw/commoncatalog/ATTRIBUTION.jsonl (author, page URL, license URL for every image).
"""
import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

QUOTAS = {  # medium -> target originals
    "film_scan": 320,
    "digital_camera": 260,
    "smartphone": 120,
    "digital_filtered": 60,
    "smartphone_filtered": 60,
}
SUBJECT_ORDER = ["night", "street", "portrait", "architecture", "landscape", "other"]
PER_UPLOADER = {"film_scan": 6, "digital_camera": 3, "smartphone": 3, "digital_filtered": 4, "smartphone_filtered": 4}


def select(cands, seed=0):
    rng = random.Random(seed)
    rng.shuffle(cands)
    out, per_up = [], Counter()
    for medium, quota in QUOTAS.items():
        pool = [c for c in cands if c["medium"] == medium and min(c["width"], c["height"]) >= 1024]
        by_subj = defaultdict(list)
        for c in pool:
            by_subj[(c["subjects"] or ["other"])[0]].append(c)
        picked = []
        # round-robin over subjects so rare ones (night, street) are represented
        while len(picked) < quota and any(by_subj.values()):
            for s in SUBJECT_ORDER:
                if by_subj.get(s) and len(picked) < quota:
                    c = by_subj[s].pop()
                    if per_up[c["uid"]] >= PER_UPLOADER[medium]:
                        continue
                    per_up[c["uid"]] += 1
                    picked.append(c)
        out += picked
    return out


def fetch_group(fs, file, rg, rows, out_dir):
    from filmlook_ml.datasets.commoncatalog import REPO, fetch_rows, jpeg_ok

    done = [(r, out_dir / f"cc_{r['photoid']}.jpg") for r in rows if (out_dir / f"cc_{r['photoid']}.jpg").exists()]
    if len(done) == len(rows):  # resume: already fetched
        return [(r, p, hashlib.sha256(p.read_bytes()).hexdigest()) for r, p in done], []

    for attempt in range(4):
        try:
            got = fetch_rows(fs, f"datasets/{REPO}/{file}", rg, {r["row"] for r in rows})
            break
        except Exception as e:  # network hiccup: back off and retry
            if attempt == 3:
                return [], [(r, f"fetch failed: {e}") for r in rows]
            time.sleep(2 ** attempt * 3)
    ok, bad = [], []
    for r in rows:
        b = got.get(r["row"])
        if not b or not jpeg_ok(b) or len(b) > 30 * 2**20:
            bad.append((r, "missing/not jpeg/too large"))
            continue
        p = out_dir / f"cc_{r['photoid']}.jpg"
        p.write_bytes(b)
        ok.append((r, p, hashlib.sha256(b).hexdigest()))
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates/commoncatalog_candidates.jsonl")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import HfFileSystem

    from filmlook_ml import paths as P
    from filmlook_ml.datasets.preprocess import write_jsonl

    cands = [json.loads(x) for x in open(a.candidates)]
    sel = select(cands, a.seed)
    print("selected", len(sel), dict(Counter(c["medium"] for c in sel)))
    print("subjects", dict(Counter((c["subjects"] or ["other"])[0] for c in sel)))
    print("uploaders", len({c["uid"] for c in sel}))
    groups = defaultdict(list)
    for c in sel:
        groups[(c["file"], c["row_group"])].append(c)
    print("row groups to touch", len(groups))
    if a.plan:
        return
    out_dir = P.RAW / "commoncatalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    records, attributions, failures = [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch_group, fs, f, g, rows, out_dir) for (f, g), rows in groups.items()]
        for k, fu in enumerate(as_completed(futs)):
            ok, bad = fu.result()
            failures += [{"photoid": r["photoid"], "reason": why} for r, why in bad]
            for r, p, sha in ok:
                records.append({
                    "dataset_id": "commoncatalog_ccby", "path": str(p), "license": r["license"],
                    "license_url": r["licenseurl"], "source_url": r["pageurl"], "source_id": str(r["photoid"]),
                    "author": r["unickname"], "attribution": f"Photo by {r['unickname']} on Flickr ({r['pageurl']}), "
                                                            f"{r['license']} ({r['licenseurl']})",
                    "group_id": f"flickr:{r['uid']}", "medium": r["medium"], "subjects": r["subjects"],
                    "camera": r["camera"], "monochrome_tag": r["monochrome_tag"], "file_sha256": sha,
                    "datetaken": r["datetaken"]})
                attributions.append({"file": p.name, "author": r["unickname"], "page": r["pageurl"],
                                     "license": r["license"], "license_url": r["licenseurl"],
                                     "title": r.get("title", "")})
            if k % 25 == 0:
                print(f"[{k + 1}/{len(groups)}] images={len(records)} failed={len(failures)} "
                      f"{time.time() - t0:.0f}s", flush=True)
    write_jsonl(P.MANIFESTS / "raw_commoncatalog.jsonl", records)
    write_jsonl(out_dir / "ATTRIBUTION.jsonl", attributions)
    write_jsonl(out_dir / "failures.jsonl", failures)
    print(f"done: {len(records)} images, {len(failures)} failures, {time.time() - t0:.0f}s")
    print(dict(Counter(r["medium"] for r in records)))


if __name__ == "__main__":
    main()
