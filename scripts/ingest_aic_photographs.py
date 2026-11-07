#!/usr/bin/env python
"""aic_photographs: curated-photography pilot from the Art Institute of Chicago's public-domain photographs (CC0).

  python scripts/ingest_aic_photographs.py --download [--max-minutes 90] [--target 600] [--per-artist 8]

Source: AIC public API (https://api.artic.edu/docs/; no key; descriptive AIC-User-Agent; requests in series, >= 1.2 s apart).
Selection: department "Photography and Media", is_public_domain = true (AIC releases these images under CC0), image present.
Images: IIIF renditions at 843 px wide (the size AIC documents for API consumers), stored as received (JPEG), sha256 and
perceptual hash recorded. Caps: --per-artist (no single photographer dominates), --target total. Labels are museum
catalogue records (artist, date, medium/process such as gelatin silver print, albumen print, chromogenic print), i.e. owner
records of the physical print process, not aesthetic judgements and not film-stock labels. Group: curated_photography
(kept separate from the real-film pool and from synthetic data). Split by artist hash (analysis/test) so no artist spans
both halves. Mostly 1840s-1920s work: a documented limitation (no modern editorial photography).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.downloader import DownloadState, Downloader  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import phash  # noqa: E402

DS = "aic_photographs"
RAW = P.RAW / DS; IMG = RAW / "images"; MANIFEST = P.MANIFESTS / f"{DS}_manifest.jsonl"; STATE = RAW / "download_state.jsonl"; LOG = RAW / "download.log"
LISTING = RAW / "listing.json"
UA = {"User-Agent": "filmlook-ml-dataset-acquisition/0.1 (research; python-urllib)", "AIC-User-Agent": "filmlook-ml (research; dataset acquisition)"}
FIELDS = "id,title,artist_title,artist_id,date_display,date_start,medium_display,is_public_domain,image_id,department_title,classification_title,credit_line,place_of_origin"
API = "https://api.artic.edu/api/v1/artworks/search"


def half_of(artist: str) -> str:
    return "analysis" if int.from_bytes(hashlib.sha256((DS + ":" + str(artist)).encode()).digest()[:2], "little") % 2 == 0 else "test"


def listing(max_pages: int = 60, spacing: float = 1.2) -> list[dict]:
    """Page through the search API (metadata only) once; cached in listing.json."""
    if LISTING.exists():
        return json.loads(LISTING.read_text())["works"]
    works, page = [], 1
    while page <= max_pages:
        body = {"fields": FIELDS.split(","), "limit": 100, "page": page, "sort": [{"id": "asc"}],
                "query": {"bool": {"must": [{"term": {"is_public_domain": True}}, {"term": {"department_title.keyword": "Photography and Media"}}, {"exists": {"field": "image_id"}}]}}}
        req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={**UA, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as r:
            d = json.load(r)
        works += [w for w in d.get("data", []) if w.get("image_id") and w.get("is_public_domain")]
        if page >= d.get("pagination", {}).get("total_pages", 0):
            break
        page += 1; time.sleep(spacing)
    RAW.mkdir(parents=True, exist_ok=True); LISTING.write_text(json.dumps({"works": works, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(works)}))
    return works


def select(works: list[dict], target: int, per_artist: int, seed: int = 0) -> list[dict]:
    import random
    pool = sorted(works, key=lambda w: w["id"]); random.Random(seed).shuffle(pool)
    out, per = [], Counter()
    for w in pool:
        a = w.get("artist_title") or "unknown"
        if per[a] >= per_artist:
            continue
        per[a] += 1; out.append(w)
        if len(out) >= target:
            break
    return out


def validate(content: bytes, tmp: Path):
    from PIL import Image
    im = Image.open(io.BytesIO(content)); im.verify()
    if len(content) < 10_000:
        raise ValueError("suspiciously small image")
    tmp.write_bytes(content)


def write_outputs(items: list[dict], state: DownloadState, complete: bool) -> dict:
    rows = []
    for it in items:
        st = state.get(it["id"])
        if st.get("status") != "validated":
            continue
        w = it["work"]; p = Path(it["dst"])
        try:
            from PIL import Image
            with Image.open(p) as im:
                wd, ht = im.size; ph = phash(im.convert("RGB"))
        except Exception:
            wd = ht = None; ph = None
        artist = w.get("artist_title") or "unknown"
        rows.append({"image_id": it["id"], "source_dataset": DS, "dataset_id": DS, "path": str(p), "source_url": f"https://www.artic.edu/artworks/{w['id']}", "original_image_url": it["urls"][0],
                     "photographer_id": f"aic_artist:{w.get('artist_id') or artist}", "uploader_id": "artic.edu", "uploader": f"aic_artist:{w.get('artist_id') or artist}", "author": artist,
                     "title": w.get("title"), "date_display": w.get("date_display"), "date_start": w.get("date_start"), "process": w.get("medium_display"), "classification": w.get("classification_title"),
                     "place_of_origin": w.get("place_of_origin"), "credit_line": w.get("credit_line"), "film_stock": None, "film_type": None, "camera": None, "scanner": None, "scan_process": "museum digitisation of a print",
                     "authenticity_class": "A", "authenticity_evidence": "museum catalogue record of the physical print (medium_display); the digitised object is a print, not a negative",
                     "group": "curated_photography", "kind": "print", "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "commercial_use_allowed": True, "training_use_allowed": True,
                     "image_width": wd, "image_height": ht, "sha256": st.get("sha256"), "perceptual_hash": ph, "split": half_of(artist)})
    rows.sort(key=lambda r: r["image_id"]); MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".jsonl.part"); tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); os.replace(tmp, MANIFEST)
    (RAW / "ATTRIBUTION.jsonl").write_text("".join(json.dumps({"file": Path(r["path"]).name, "artist": r["author"], "title": r["title"], "page": r["source_url"], "license": "CC0 (AIC Open Access)"}) + "\n" for r in rows))
    artists = Counter(r["author"] for r in rows); proc = Counter((r["process"] or "").lower()[:40] for r in rows)
    summary = {"dataset": DS, "complete": complete, "rows": len(rows), "target_items": len(items), "item_states": state.counts(), "n_artists": len(artists),
               "max_artist_share": round(max(artists.values()) / len(rows), 3) if rows else None, "processes_top": proc.most_common(12), "splits": dict(Counter(r["split"] for r in rows)),
               "decades": dict(Counter((r["date_start"] // 10 * 10) if isinstance(r["date_start"], int) else "?" for r in rows)), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1, default=str)); return summary


def run(max_minutes: float, target: int, per_artist: int, spacing: float = 1.2) -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    try:
        works = listing()
    except Exception as ex:
        print(f"AIC API unavailable: {ex!r} (resumable)"); return 3
    sel = select(works, target, per_artist)
    items = [{"id": f"aic_{w['id']}", "urls": [f"https://www.artic.edu/iiif/2/{w['image_id']}/full/843,/0/default.jpg"], "dst": IMG / f"aic_{w['id']}.jpg", "work": w} for w in sel]
    dl = Downloader(STATE, headers=UA, spacing=spacing, max_seconds=max_minutes * 60, log_path=LOG, checkpoint_every=50, max_retry_after=120)
    dl.on_checkpoint = lambda: write_outputs(items, dl.state, False)
    res = dl.run(items, validate)
    complete = res.get("run_status") == "complete"
    s = write_outputs(items, dl.state, complete)
    print(json.dumps({"complete": complete, "run_status": res.get("run_status"), "rows": s["rows"], "n_artists": s["n_artists"], "states": s["item_states"]}), flush=True)
    return 0 if complete else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true"); ap.add_argument("--max-minutes", type=float, default=90); ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--per-artist", type=int, default=8); ap.add_argument("--spacing", type=float, default=1.2)
    a = ap.parse_args()
    raise SystemExit(run(a.max_minutes, a.target, a.per_artist, a.spacing) if a.download else 0)


if __name__ == "__main__":
    main()
