#!/usr/bin/env python
"""met_photographs: a fresh held-out museum set from The Met's public-domain photographs (CC0), session 7.

  python scripts/ingest_met_photographs.py --prepare                         # CSV -> candidate list (metadata only)
  python scripts/ingest_met_photographs.py --download [--max-minutes 90] [--target 700] [--per-artist 6]
  python scripts/ingest_met_photographs.py --status

Source: The Met Open Access initiative (https://metmuseum.github.io/): the MetObjects.csv release (CC0) filtered to
department "Photographs", Is Public Domain = True, classification "Photographs" (albums, books, negatives,
transparencies and equipment excluded). The search endpoint has returned HTTP 410 since 2026-11-07, so the CSV is the
listing route; per-object records (image URLs, public-domain flag re-checked) come from the documented objects endpoint
(no key; the Met asks for at most 80 requests per second, this client sends one every --spacing seconds, serially, and
honours Retry-After). Images are the `primaryImageSmall` renditions (web-large JPEG, comparable to the AIC 843 px
pilot), stored as received, sha256 and perceptual hash recorded, near-duplicates (phash <= 4) rejected.
Labels are museum catalogue records (artist, date, medium = physical print process) - owner records, never aesthetic
judgements and never film-stock labels. Caps: --per-artist (unknown artists count as one group) and --target.
Split by artist hash (analysis / test) so no photographer spans both halves. Project policy for this set:
training_use_allowed = false (held-out evaluation only; the CC0 licence would permit more, the policy does not).
Era: 1840s-1890s dominate (albumen, salted paper, daguerreotype, cyanotype, gelatin silver) - a documented limitation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.downloader import DownloadState, Downloader  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import hamming, phash  # noqa: E402

DS = "met_photographs"
RAW = P.RAW / DS; IMG = RAW / "images"; MANIFEST = P.MANIFESTS / f"{DS}_manifest.jsonl"; STATE = RAW / "download_state.jsonl"; LOG = RAW / "download.log"
CSV = P.DATA / "candidates" / "met" / "MetObjects.csv"; CAND = P.DATA / "candidates" / "met" / "met_photographs_candidates.json"; OBJECTS = RAW / "objects.jsonl"
CSV_URL = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
UA = {"User-Agent": "filmlook-ml-dataset-acquisition/0.1 (research; python-requests)"}
KEEP = ["Object ID", "Object Number", "Object Name", "Title", "Artist Display Name", "Constituent ID", "Artist Nationality", "Artist Begin Date", "Artist End Date",
        "Object Date", "Object Begin Date", "Object End Date", "Medium", "Dimensions", "Classification", "Credit Line", "Link Resource", "Culture", "Is Highlight", "Rights and Reproduction", "Tags"]


def log(msg):
    RAW.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"; print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def half_of(artist: str) -> str:
    return "analysis" if int.from_bytes(hashlib.sha256((DS + ":" + str(artist)).encode()).digest()[:2], "little") % 2 == 0 else "test"


def artist_key(c: dict) -> str:
    name = (c.get("Artist Display Name") or "").strip()
    if not name or name.lower() in ("unknown", "unknown artist", "anonymous"):
        return "unknown"
    return f"met_artist:{c.get('Constituent ID') or name}"


# ------------------------------------------------------------------------------------------------- listing
def prepare() -> dict:
    """MetObjects.csv (downloaded once if absent) -> candidates: public-domain works of the Photographs department whose
    classification is 'Photographs' and whose medium is recorded. Metadata only; nothing is fetched per object here."""
    CAND.parent.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        import requests
        log(f"downloading {CSV_URL}")
        with requests.get(CSV_URL, headers=UA, stream=True, timeout=(10, 120)) as r:
            r.raise_for_status(); tmp = CSV.with_suffix(".csv.part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        os.replace(tmp, CSV)
    csv.field_size_limit(1 << 30); n = 0; out = []; drop = Counter()
    with open(CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if row.get("Department") != "Photographs":
                continue
            if row.get("Is Public Domain") != "True":
                drop["not_public_domain"] += 1; continue
            if row.get("Classification") != "Photographs":
                drop["classification"] += 1; continue
            if not (row.get("Medium") or "").strip():
                drop["no_medium"] += 1; continue
            out.append({k: row.get(k) for k in KEEP})
    summary = {"csv_rows": n, "candidates": len(out), "dropped": dict(drop), "csv_sha256": _sha(CSV), "prepared": time.strftime("%Y-%m-%dT%H:%M:%S")}
    CAND.write_text(json.dumps({"summary": summary, "candidates": out})); log(f"prepare: {summary}")
    return summary


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select(cands: list[dict], target: int, per_artist: int, seed: int = 0) -> list[dict]:
    import random
    pool = sorted(cands, key=lambda c: int(c["Object ID"])); random.Random(seed).shuffle(pool)
    out, per = [], Counter()
    for c in pool:
        a = artist_key(c)
        if per[a] >= per_artist:
            continue
        per[a] += 1; out.append(c)
        if len(out) >= target:
            break
    return out


# ------------------------------------------------------------------------------------------------- per-object records
def object_records(selected: list[dict], spacing: float, deadline: float) -> dict[str, dict]:
    """objects/{id} for the selected works (image URL, public-domain flag re-checked); cached in objects.jsonl."""
    import requests
    have = {}
    if OBJECTS.exists():
        for line in OBJECTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line); have[str(r["objectID"])] = r
    todo = [c for c in selected if str(c["Object ID"]) not in have]
    log(f"objects: {len(have)} cached, {len(todo)} to fetch")
    s = requests.Session(); rl = 0
    with open(OBJECTS, "a") as fh:
        for k, c in enumerate(todo):
            if time.time() > deadline:
                log("objects: max runtime reached (resumable)"); break
            oid = str(c["Object ID"])
            try:
                r = s.get(API + oid, headers=UA, timeout=(10, 60))
            except Exception as ex:  # noqa: BLE001
                log(f"objects: network error on {oid}: {ex!r}"[:200]); time.sleep(5); continue
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After"); wait = min(float(ra) if ra and ra.replace(".", "").isdigit() else 30.0, 120.0); rl += 1
                log(f"objects: HTTP {r.status_code} on {oid}; waiting {wait}s (Retry-After honoured)"); time.sleep(wait)
                if rl >= 10:
                    log("objects: ten rate-limit responses; stopping this run (resumable)"); break
                continue
            if r.status_code != 200:
                rec = {"objectID": int(oid), "http_status": r.status_code}
            else:
                d = r.json(); rec = {"objectID": int(oid), "http_status": 200, "isPublicDomain": d.get("isPublicDomain"), "primaryImage": d.get("primaryImage"), "primaryImageSmall": d.get("primaryImageSmall"),
                                     "artistDisplayName": d.get("artistDisplayName"), "constituentID": (d.get("constituents") or [{}])[0].get("constituentID") if d.get("constituents") else None,
                                     "medium": d.get("medium"), "objectDate": d.get("objectDate"), "objectBeginDate": d.get("objectBeginDate"), "classification": d.get("classification"), "objectURL": d.get("objectURL"),
                                     "rightsAndReproduction": d.get("rightsAndReproduction"), "metadataDate": d.get("metadataDate")}
            fh.write(json.dumps(rec) + "\n"); fh.flush(); have[oid] = rec
            if (k + 1) % 50 == 0:
                log(f"objects: {k + 1}/{len(todo)} fetched")
            time.sleep(spacing)
    return have


def validate(content: bytes, tmp: Path):
    from PIL import Image
    im = Image.open(io.BytesIO(content)); im.verify()
    if len(content) < 10_000:
        raise ValueError("suspiciously small image")
    tmp.write_bytes(content)


def write_outputs(items: list[dict], state: DownloadState, complete: bool) -> dict:
    from PIL import Image
    rows, hashes = [], []
    for it in items:
        st = state.get(it["id"])
        if st.get("status") != "validated":
            continue
        c, o, p = it["cand"], it["obj"], Path(it["dst"])
        try:
            with Image.open(p) as im:
                wd, ht = im.size; ph = phash(im.convert("RGB"))
        except Exception:
            continue
        if any(hamming(ph, h) <= 4 for h in hashes):
            state.set(it["id"], "permanent_failure", error="near-duplicate (phash <= 4) of a stored image", dst=str(p)); continue
        hashes.append(ph); artist = artist_key(c); date_start = int(c["Object Begin Date"]) if (c.get("Object Begin Date") or "").lstrip("-").isdigit() else None
        rows.append({"image_id": it["id"], "source_dataset": DS, "dataset_id": DS, "path": str(p), "source_url": c.get("Link Resource") or o.get("objectURL"), "original_image_url": it["urls"][0],
                     "photographer_id": artist, "uploader_id": "metmuseum.org", "uploader": artist, "author": c.get("Artist Display Name") or "Unknown", "artist_nationality": c.get("Artist Nationality"),
                     "title": c.get("Title"), "object_number": c.get("Object Number"), "date_display": c.get("Object Date"), "date_start": date_start, "process": c.get("Medium"), "classification": c.get("Classification"),
                     "culture": c.get("Culture"), "credit_line": c.get("Credit Line"), "film_stock": None, "film_type": None, "camera": None, "scanner": None, "scan_process": "museum digitisation of a print",
                     "authenticity_class": "A", "authenticity_evidence": "museum catalogue record of the physical print (Medium); the digitised object is a print, not a negative",
                     "group": "curated_photography", "kind": "print", "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "commercial_use_allowed": True,
                     "training_use_allowed": False, "training_use_note": "held-out evaluation set (session 7 policy); CC0 would permit more",
                     "image_width": wd, "image_height": ht, "sha256": st.get("sha256"), "perceptual_hash": ph, "split": half_of(artist), "metadata_date": o.get("metadataDate")})
    rows.sort(key=lambda r: r["image_id"]); MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".jsonl.part"); tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); os.replace(tmp, MANIFEST)
    (RAW / "ATTRIBUTION.jsonl").write_text("".join(json.dumps({"file": Path(r["path"]).name, "artist": r["author"], "title": r["title"], "page": r["source_url"], "license": "CC0 (The Met Open Access)"}) + "\n" for r in rows))
    artists = Counter(r["uploader"] for r in rows); proc = Counter((r["process"] or "").lower()[:40] for r in rows)
    summary = {"dataset": DS, "complete": complete, "rows": len(rows), "target_items": len(items), "item_states": state.counts(), "n_artists": len(artists),
               "max_artist_share": round(max(artists.values()) / len(rows), 3) if rows else None, "processes_top": proc.most_common(12), "splits": dict(Counter(r["split"] for r in rows)),
               "decades": dict(Counter((r["date_start"] // 10 * 10) if isinstance(r["date_start"], int) else "?" for r in rows)), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1, default=str)); return summary


def run(max_minutes: float, target: int, per_artist: int, spacing: float, seed: int) -> int:
    IMG.mkdir(parents=True, exist_ok=True); t0 = time.time(); deadline = t0 + max_minutes * 60
    if not CAND.exists():
        prepare()
    cands = json.loads(CAND.read_text())["candidates"]
    sel = select(cands, target, per_artist, seed)
    log(f"download: {len(sel)} selected of {len(cands)} candidates ({len({artist_key(c) for c in sel})} artists)")
    objs = object_records(sel, spacing, deadline)
    items = []; skipped = Counter()
    for c in sel:
        o = objs.get(str(c["Object ID"]))
        if o is None:
            skipped["no_record_yet"] += 1; continue
        if o.get("http_status") != 200:
            skipped[f"http_{o.get('http_status')}"] += 1; continue
        if not o.get("isPublicDomain"):
            skipped["not_public_domain_live"] += 1; continue
        url = o.get("primaryImageSmall") or o.get("primaryImage")
        if not url:
            skipped["no_image"] += 1; continue
        items.append({"id": f"met_{c['Object ID']}", "urls": [url], "dst": IMG / f"met_{c['Object ID']}.jpg", "cand": c, "obj": o})
    log(f"download: {len(items)} items with image URLs; skipped {dict(skipped)}")
    remaining_s = max(60.0, deadline - time.time())
    dl = Downloader(STATE, headers=UA, spacing=spacing, max_seconds=remaining_s, log_path=LOG, checkpoint_every=50, max_retry_after=120)
    dl.on_checkpoint = lambda: write_outputs(items, dl.state, False)
    res = dl.run(items, validate)
    complete = res.get("run_status") == "complete" and not skipped.get("no_record_yet")
    s = write_outputs(items, dl.state, complete)
    print(json.dumps({"complete": complete, "run_status": res.get("run_status"), "rows": s["rows"], "n_artists": s["n_artists"], "states": s["item_states"], "skipped": dict(skipped)}), flush=True)
    return 0 if complete else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--download", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90); ap.add_argument("--target", type=int, default=700); ap.add_argument("--per-artist", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rc = 0
    if a.prepare:
        print(json.dumps(prepare()))
    if a.download:
        rc = run(a.max_minutes, a.target, a.per_artist, a.spacing, a.seed)
    if a.status:
        print((RAW / "summary.json").read_text() if (RAW / "summary.json").exists() else "no summary yet")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
