"""Collect CC0 / CC BY reference photographs per GBIF species key and download <=384 px thumbnails.

  python -m research.species.collect query     [--target 50] [--unknown-target 15]   -> data/species/candidates.jsonl
  python -m research.species.collect download  [--workers 4]                          -> data/species/images/, manifest.jsonl

Source: GBIF occurrence API (human observations with still images, record licence CC0 or CC BY). Each media item's
own licence is checked again (CC0, CC BY x.y or PDM only; NC / SA / ND / missing are rejected). Labels are the
occurrence's GBIF speciesKey (must equal the resolved key). At most one image per occurrence and two per recorder
per species, so recorder-disjoint splits stay possible. Herbarium sheets and specimens are excluded by
basisOfRecord. Etiquette: one request per second per host (shared limiter), Retry-After honoured, resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import queue
import re
import threading
import time
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from PIL import Image

from research.species import common as C

CAND = C.DATA / "candidates.jsonl"
MANIFEST = C.DATA / "manifest.jsonl"
FAILED = C.DATA / "download_failures.jsonl"
MAX_SIDE = 384
MAX_BYTES = 12 * 1024 * 1024


def norm_recorder(s: str | None) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.split("|")[0].split(";")[0].strip()


def query_taxon(sess, taxon: dict, target: int, per_recorder: int = 2, pages: int = 2) -> list[dict]:
    rows, per_rec = [], Counter()
    want = int(taxon["gbif_key"])
    for page in range(pages):
        params = {"taxonKey": want, "mediaType": "StillImage", "basisOfRecord": "HUMAN_OBSERVATION",
                  "license": ["CC0_1_0", "CC_BY_4_0"], "limit": 300, "offset": 300 * page}
        r = C.get(sess, f"{C.GBIF}/occurrence/search", params)
        if r is None or r.status_code != 200:
            print(f"  GBIF query failed for {taxon['id']} page {page}: {None if r is None else r.status_code}")
            break
        d = r.json()
        for occ in d.get("results", []):
            skey = occ.get("speciesKey") or occ.get("acceptedTaxonKey")
            if skey != want:
                continue
            media = [m for m in occ.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
            chosen = None
            for m in media:
                lid = C.licence_id(m.get("license"))
                fmt = (m.get("format") or "").lower()
                if lid and (not fmt or "jpeg" in fmt or "jpg" in fmt or "png" in fmt):
                    chosen = (m, lid)
                    break
            if not chosen:
                continue
            m, lid = chosen
            recorder = norm_recorder(occ.get("recordedBy")) or norm_recorder(occ.get("rightsHolder")) or norm_recorder(m.get("creator")) or f"occ:{occ['key']}"
            if per_rec[recorder] >= per_recorder:
                continue
            per_rec[recorder] += 1
            rows.append({"species_id": taxon["id"], "gbif_key": want, "kind": taxon["kind"], "occurrence_key": occ["key"],
                         "dataset_key": occ.get("datasetKey"), "recorder": recorder, "licence": lid, "licence_url": m.get("license"),
                         "attribution": m.get("rightsHolder") or occ.get("rightsHolder") or m.get("creator") or occ.get("recordedBy"),
                         "source_url": m.get("references") or f"https://www.gbif.org/occurrence/{occ['key']}",
                         "image_url": m["identifier"], "country": occ.get("country"), "event_date": occ.get("eventDate"),
                         "record_licence": occ.get("license"), "scientific_name": occ.get("species")})
            if len(rows) >= target:
                return rows
        if d.get("endOfRecords") or len(d.get("results", [])) < 300:
            break
    return rows


def cmd_query(a):
    sess = C.session()
    taxa = [t for t in json.load(open(C.DATA / "taxa.json")) if t["status"].startswith("ok")]
    unknown = json.load(open(C.DATA / "unknown_taxa.json")) if (C.DATA / "unknown_taxa.json").exists() else []
    done = {r["species_id"] for r in C.read_jsonl(CAND)}
    todo = [(t, int(a.target * 1.4)) for t in taxa if t["id"] not in done] + [(t, int(a.unknown_target * 1.6)) for t in unknown if t["id"] not in done]
    print(f"{len(todo)} taxa to query ({len(done)} already done)")
    for i, (t, target) in enumerate(todo):
        rows = query_taxon(sess, t, target)
        if not rows:
            C.append_jsonl(CAND, {"species_id": t["id"], "gbif_key": t["gbif_key"], "kind": t["kind"], "empty": True})
        for r in rows:
            C.append_jsonl(CAND, r)
        print(f"[{i + 1}/{len(todo)}] {t['id']:40s} {len(rows):3d} candidates, {len({r['recorder'] for r in rows})} recorders", flush=True)


def inat_variant(url: str, size: str) -> str:
    return re.sub(r"/original\.", f"/{size}.", url) if "inaturalist" in url else url


def fetch_image(sess, row: dict) -> tuple[bytes | None, str]:
    urls = [inat_variant(row["image_url"], "medium"), row["image_url"]] if "inaturalist" in row["image_url"] else [row["image_url"]]
    for u in dict.fromkeys(urls):
        r = C.get(sess, u, stream=True, timeout=60, max_attempts=3)
        if r is None:
            continue
        if r.status_code != 200:
            r.close()
            continue
        buf, size = io.BytesIO(), 0
        for chunk in r.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                r.close()
                return None, "too large"
            buf.write(chunk)
        r.close()
        return buf.getvalue(), u
    return None, "http error"


def thumbnail(data: bytes) -> tuple[bytes, int, int]:
    im = Image.open(io.BytesIO(data))
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88, optimize=True)
    return out.getvalue(), im.width, im.height


def cmd_download(a):
    cands = [r for r in C.read_jsonl(CAND) if not r.get("empty")]
    have = C.read_jsonl(MANIFEST)
    failed = {r["occurrence_key"] for r in C.read_jsonl(FAILED)}
    done_occ = {r["occurrence_key"] for r in have}
    counts = Counter(r["species_id"] for r in have)
    targets = {}
    for r in cands:
        targets[r["species_id"]] = a.unknown_target if r["species_id"].startswith("unk_") else a.target
    # round-robin order over species so a partial run stays balanced
    per_sp = defaultdict(list)
    for r in cands:
        if r["occurrence_key"] not in done_occ and r["occurrence_key"] not in failed:
            per_sp[r["species_id"]].append(r)
    order, i = [], 0
    while any(per_sp.values()):
        for sid in list(per_sp):
            if per_sp[sid]:
                order.append(per_sp[sid].pop(0))
        i += 1
    print(f"{len(order)} candidates queued; {len(have)} images already present")
    q = queue.Queue()
    for r in order:
        q.put(r)
    lock = threading.Lock()
    stats = Counter()
    t0 = time.time()

    def worker():
        sess = C.session()
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                if counts[r["species_id"]] >= targets[r["species_id"]]:
                    stats["skipped_full"] += 1
                    continue
            data, used = fetch_image(sess, r)
            if data is None:
                with lock:
                    C.append_jsonl(FAILED, {"occurrence_key": r["occurrence_key"], "species_id": r["species_id"], "reason": used})
                    stats["failed"] += 1
                continue
            try:
                jpg, w, h = thumbnail(data)
            except Exception as e:  # noqa: BLE001
                with lock:
                    C.append_jsonl(FAILED, {"occurrence_key": r["occurrence_key"], "species_id": r["species_id"], "reason": f"decode: {type(e).__name__}"})
                    stats["failed"] += 1
                continue
            sha = hashlib.sha1(jpg).hexdigest()
            rel = f"images/{r['species_id']}/{r['occurrence_key']}.jpg"
            path = C.DATA / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(jpg)
            with lock:
                if counts[r["species_id"]] >= targets[r["species_id"]]:
                    path.unlink(missing_ok=True)
                    stats["skipped_full"] += 1
                    continue
                counts[r["species_id"]] += 1
                C.append_jsonl(MANIFEST, {**r, "path": rel, "width": w, "height": h, "sha1": sha, "bytes": len(jpg), "fetched_url": used,
                                          "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                stats["ok"] += 1
                stats["bytes_thumb"] += len(jpg)
                stats["bytes_downloaded"] += len(data)
                n = stats["ok"] + stats["failed"]
                if n % 100 == 0:
                    el = time.time() - t0
                    print(f"  {n} done ({stats['ok']} ok, {stats['failed']} failed, {stats['skipped_full']} skipped) {el / 60:.1f} min, "
                          f"{stats['bytes_downloaded'] / 1e6:.0f} MB fetched, {q.qsize()} queued", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("done", dict(stats), f"{(time.time() - t0) / 60:.1f} min")
    json.dump({"stats": dict(stats), "per_species": dict(counts)}, open(C.DATA / "download_stats.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["query", "download"])
    ap.add_argument("--target", type=int, default=50, help="images per encyclopedia species")
    ap.add_argument("--unknown-target", type=int, default=15, help="images per unknown control taxon")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    C.DATA.mkdir(parents=True, exist_ok=True)
    (cmd_query if a.phase == "query" else cmd_download)(a)


if __name__ == "__main__":
    main()
