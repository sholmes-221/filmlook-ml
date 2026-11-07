#!/usr/bin/env python
"""Ingest stock-labelled real-film photographs from Wikimedia Commons (evaluation-only dataset `commons_film`).

  python scripts/ingest_commons_film.py --list      # enumerate categories + files, fetch per-file metadata -> index
  python scripts/ingest_commons_film.py --download  # download <=1024 px renditions for eligible files -> manifest

Source: the Commons category tree "Photographs by film brand" ("Photographs taken on <stock> film", "Taken on
<stock>"), whose members are photographs the uploader states were shot on that stock, plus the file's own
"Scanned with <scanner>" category when present. Labels are uploader-asserted (like CommonCatalog tags), not owner
records (BlueNeg); they are kept as `film_stock_claim` with `stock_label_source: commons_category`.

Rights: every file carries its own licence on Commons (extmetadata LicenseShortName). Only CC0, public domain,
CC BY and CC BY-SA files are kept; everything else is skipped. Author, licence and page URL are written into every
manifest row and into ATTRIBUTION.jsonl. The dataset is registered with `training_allowed: false` (ShareAlike scope
for trained models is unresolved; the project does not train on it). Access policy: the MediaWiki API is used with a
descriptive User-Agent, one request at a time, honouring 429/maxlag (API:Etiquette); files come from
the thumbnail service as server-side renditions, never originals (originals are rate-limited per client). Digital film *emulation* categories (DxO FilmPack)
are excluded from the film set and indexed separately as `digital_emulation` controls.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filmlook_ml import paths as P  # noqa: E402

UA = {"User-Agent": "FilmLook-ML-research/0.1 (research dataset ingest; one request at a time) python-requests"}
API = "https://commons.wikimedia.org/w/api.php"
ROOT_CAT = "Category:Photographs by film brand"
EMULATION_CATS = ["Category:DxO PhotoLab FilmPack color negative film emulation", "Category:DxO PhotoLab FilmPack color positive film emulation",
                  "Category:DxO PhotoLab FilmPack black and white film emulation"]
OUT = P.RAW / "commons_film"
INDEX = OUT / "index.json"
MANIFEST = P.MANIFESTS / "commons_film_manifest.jsonl"
MAX_WIDTH = 1024
LICENSE_MAP = [("cc0", "CC0-1.0"), ("public domain", "PDM-1.0"), ("cc by-sa 4.0", "CC-BY-SA-4.0"), ("cc by-sa 3.0", "CC-BY-SA-3.0"),
               ("cc by-sa 2.5", "CC-BY-SA-2.5"), ("cc by-sa 2.0", "CC-BY-SA-2.0"), ("cc by-sa 1.0", "CC-BY-SA-1.0"), ("cc by 4.0", "CC-BY-4.0"),
               ("cc by 3.0", "CC-BY-3.0"), ("cc by 2.5", "CC-BY-2.5"), ("cc by 2.0", "CC-BY-2.0"), ("cc by 1.0", "CC-BY-1.0")]
TYPE_PARENTS = {"Category:Photographs taken on color negative film": "colour_negative", "Category:Photographs taken on black-and-white negative film": "bw_negative",
                "Category:Photographs taken on color reversal films": "colour_reversal", "Category:Photographs taken on Kodachrome film": "colour_reversal",
                "Category:Photographs taken on Ektachrome and Elite Chrome film": "colour_reversal", "Category:Photographs taken on black-and-white reversal film": "bw_reversal",
                "Category:Photographs taken on motion picture film": "cine_negative", "Category:Photographs taken on instant film": "instant"}


def q(**p):
    p.update(format="json", maxlag=5)
    for attempt in range(8):
        time.sleep(0.8)
        r = requests.post(API, data=p, headers=UA, timeout=60)        # POST: long title lists exceed the GET URI limit
        if r.status_code == 200:
            d = r.json()
            if "error" in d and d["error"].get("code") == "maxlag":
                time.sleep(5 * (attempt + 1)); continue
            return d
        if r.status_code in (429, 503):
            time.sleep(15 * (attempt + 1)); continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Commons API: too many retries")


def members(title, cmtype):
    out, cont = [], {}
    while True:
        d = q(action="query", list="categorymembers", cmtitle=title, cmtype=cmtype, cmlimit=500, **cont)
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" not in d:
            return out
        cont = d["continue"]


def walk(root, depth, seen):
    rows = []
    for s in members(root, "subcat"):
        if s in seen:
            continue
        seen.add(s); rows.append((s, root))
        if depth > 1:
            rows += walk(s, depth - 1, seen)
    return rows


def norm_stock(cat):
    s = re.sub(r"^Category:(Photographs taken on |Taken on |Photographs by [^/]+/\d[\d-]* ?)", "", cat)
    s = re.sub(r"\s+film(s)?$", "", s, flags=re.I).strip()
    return s


def film_type(cat, parents_of):
    seen, stack = set(), [cat]
    while stack:
        c = stack.pop()
        if c in TYPE_PARENTS:
            return TYPE_PARENTS[c]
        for p in parents_of.get(c, []):
            if p not in seen:
                seen.add(p); stack.append(p)
    return "unknown"


def parse_license(em):
    s = (em.get("LicenseShortName", {}).get("value") or "").lower()
    for key, lic in LICENSE_MAP:
        if s.startswith(key):
            return lic
    return None


def do_list():
    OUT.mkdir(parents=True, exist_ok=True)
    listing = OUT / "listing.json"                                     # enumeration is slow (serial API): persisted and reused
    if listing.exists():
        L = json.loads(listing.read_text()); cats = [tuple(c) for c in L["cats"]]; parents_of = L["parents_of"]; files = L["files"]; emu = L["emu"]; stock_of_cat = {k: tuple(v) for k, v in L["stock_of_cat"].items()}
        print(f"reusing {listing}")
    else:
        cats, parents_of, files, emu, stock_of_cat = enumerate_tree()
        listing.write_text(json.dumps({"cats": cats, "parents_of": parents_of, "files": files, "emu": emu, "stock_of_cat": stock_of_cat}))
    print(f"{len(cats)} categories, {len(stock_of_cat)} stock/roll categories, {len(files)} film files, {len(emu)} emulation files", flush=True)
    index = {"generated": time.strftime("%Y-%m-%d"), "root": ROOT_CAT, "categories": {c: {"parent": p, "stock": stock_of_cat.get(c, (None, None))[0], "film_type": film_type(c, parents_of)} for c, p in cats},
             "files": {}, "emulation_files": {}}
    for kind, pool in (("files", files), ("emulation_files", emu)):
        titles = list(pool)
        for i in range(0, len(titles), 50):
            d = q(action="query", prop="imageinfo|categories", iiprop="extmetadata|size|url|sha1|user|mime", iiurlwidth=MAX_WIDTH, clshow="!hidden", cllimit=100,
                  iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired|DateTimeOriginal|Model|Make|Software", titles="|".join(titles[i:i + 50]))
            for pg in d["query"]["pages"].values():
                ii = (pg.get("imageinfo") or [{}])[0]; em = ii.get("extmetadata", {})
                rec = pool.get(pg["title"], {"title": pg["title"], "stock_cats": [], "roll_cat": None})
                cats_f = [x["title"] for x in pg.get("categories", [])]
                rec.update({"pageid": pg.get("pageid"), "mime": ii.get("mime"), "width": ii.get("width"), "height": ii.get("height"), "sha1": ii.get("sha1"),
                            "uploader": ii.get("user"), "url": ii.get("url"), "thumburl": ii.get("thumburl"), "descriptionurl": ii.get("descriptionurl"),
                            "license": parse_license(em), "license_raw": em.get("LicenseShortName", {}).get("value"), "license_url": em.get("LicenseUrl", {}).get("value"),
                            "artist_html": em.get("Artist", {}).get("value"), "credit_html": em.get("Credit", {}).get("value"),
                            "date_original": em.get("DateTimeOriginal", {}).get("value"), "exif_model": em.get("Model", {}).get("value"), "exif_make": em.get("Make", {}).get("value"),
                            "software": em.get("Software", {}).get("value"), "scanner_cats": [c for c in cats_f if c.startswith("Category:Scanned with ")],
                            "camera_cats": [c for c in cats_f if c.startswith("Category:Taken with ")], "all_cats_n": len(cats_f)})
                index[kind][pg["title"]] = rec
            print(f"  {kind}: {min(i + 50, len(titles))}/{len(titles)}", flush=True)
    INDEX.write_text(json.dumps(index))
    lic = Counter(r["license"] for r in index["files"].values()); mimes = Counter(r["mime"] for r in index["files"].values())
    print(json.dumps({"files": len(index["files"]), "licences": dict(lic), "mimes": dict(mimes), "uploaders": len({r["uploader"] for r in index["files"].values()}),
                      "with_scanner_cat": sum(bool(r["scanner_cats"]) for r in index["files"].values()), "emulation": len(index["emulation_files"])}, indent=1))


def enumerate_tree():
    seen = set([ROOT_CAT]); cats = walk(ROOT_CAT, 3, seen)
    parents_of = {}
    titles = [c for c, _ in cats]
    for i in range(0, len(titles), 50):
        d = q(action="query", prop="categories", titles="|".join(titles[i:i + 50]), cllimit=500)
        for pg in d["query"]["pages"].values():
            parents_of[pg["title"]] = [x["title"] for x in pg.get("categories", [])]
    # a per-stock category is one whose normalised name is not a user/roll category
    stock_of_cat = {}
    for c, parent in cats:
        if re.match(r"^Category:Photographs by [^/]+/", c):          # roll-level user category nested under a stock
            stock_of_cat[c] = (norm_stock(parent) if parent != ROOT_CAT else norm_stock(c), c)
        elif c.startswith("Category:Photographs taken on ") or c.startswith("Category:Taken on "):
            stock_of_cat[c] = (norm_stock(c), None)
    files = {}
    for c, (stock, roll_cat) in stock_of_cat.items():
        if stock.lower().endswith(("kodak", "fujifilm", "ilford", "agfa", "agfaphoto", "lomography", "orwo", "konica", "svema", "fomapan", "rollei", "film washi", "foma", "color negative", "black-and-white negative", "color reversal")):
            continue                                                 # brand-level or type-level container, not a stock
        for f in members(c, "file"):
            files.setdefault(f, {"title": f, "stock_cats": [], "roll_cat": None})
            files[f]["stock_cats"].append(c); files[f]["roll_cat"] = files[f]["roll_cat"] or roll_cat
    emu = {}
    for root in EMULATION_CATS:
        for c in [root] + [s for s, _ in walk(root, 2, set([root]))]:
            for f in members(c, "file"):
                emu.setdefault(f, {"title": f, "stock_cats": [c], "roll_cat": None})
    return cats, parents_of, files, emu, stock_of_cat


THUMB_BUCKETS = (1280, 960, 500, 330, 250)     # Wikimedia standard thumbnail widths (T414805): any other width is HTTP 400


def rendition_urls(r):
    """Candidate URLs for a stored <= MAX_WIDTH copy. Originals, and the "thumbnail_unscaled" alias the API returns
    whenever the requested width rounds up to at least the original size, are rate-limited per client on
    upload.wikimedia.org (HTTP 429, Retry-After 600). Scaled renditions from the thumbnail service are not, but only at
    the standard widths. So: the largest standard width strictly below the original (1280 for larger files, downsized
    locally to MAX_WIDTH). Files of 250 px or less get no URL and are skipped."""
    w = r.get("width") or 0
    base = (r.get("url") or "").split("?")[0]
    if not base:
        return []
    name = base.rsplit("/", 1)[-1]
    thumb_base = base.replace("/wikipedia/commons/", "/wikipedia/commons/thumb/")
    suffix = ".jpg" if r.get("mime") == "image/tiff" else ""
    widths = [b for b in THUMB_BUCKETS if b < w]
    urls = [f"{thumb_base}/{b}px-{name}{suffix}" for b in widths[:2]]
    tb = r.get("thumburl") or ""
    if "thumbnail_unscaled" not in tb and "/thumb/" in tb and tb.split("?")[0] not in [u.split("?")[0] for u in urls]:
        urls.insert(0, tb)                                        # the rendition the API itself offered (standard size)
    return urls


def store_image(content, tmp):
    """Validator for the downloader: decode, bound to MAX_WIDTH, write JPEG q92 to tmp (renamed atomically by the caller)."""
    import io as _io
    from PIL import Image
    im = Image.open(_io.BytesIO(content)); im.load(); im = im.convert("RGB")
    if min(im.size) < 64:
        raise ValueError(f"too small: {im.size}")
    if max(im.size) > MAX_WIDTH:
        im.thumbnail((MAX_WIDTH, MAX_WIDTH), Image.LANCZOS)
    im.save(tmp, "JPEG", quality=92)


def write_outputs(index, items, state):
    """Manifest, attribution and summary for every validated item (called at checkpoints and at the end: resumable)."""
    from PIL import Image
    cat_info = index["categories"]; rows = []
    for it in items:
        rec = state.get(it["id"]); dst = Path(it["dst"])
        if rec["status"] != "validated" or not dst.exists():
            continue
        r, kind = it["rec"], it["kind"]
        with Image.open(dst) as im:
            w, h = im.size
        stock_cat = r["stock_cats"][0] if r["stock_cats"] else None
        info = cat_info.get(stock_cat, {}) if stock_cat else {}
        rows.append({"image_id": it["id"], "path": str(dst), "sha256": rec.get("sha256"), "width": w, "height": h,
                     "dataset_id": "commons_film", "kind": kind, "license": r["license"], "license_raw": r["license_raw"], "license_url": r["license_url"],
                     "attribution": re.sub(r"<[^>]+>", "", r.get("artist_html") or r.get("uploader") or ""), "uploader": r["uploader"], "source_url": r["descriptionurl"],
                     "commons_title": r["title"], "commons_sha1": r["sha1"], "original_size": [r.get("width"), r.get("height")], "rendition_max_width": MAX_WIDTH,
                     "download_url": rec.get("url"), "medium": "film_scan" if kind == "film" else "digital_emulation",
                     "film_stock_claim": info.get("stock") or (norm_stock(stock_cat) if stock_cat else None), "stock_label_source": "commons_category",
                     "film_type": info.get("film_type", "unknown"), "stock_category": stock_cat, "roll_category": r.get("roll_cat"),
                     "scanner_category": (r.get("scanner_cats") or [None])[0], "camera_category": (r.get("camera_cats") or [None])[0],
                     "date_original": r.get("date_original"), "exif_make": r.get("exif_make"), "exif_model": r.get("exif_model"), "software": r.get("software"),
                     "group_id": f"commons:{r['uploader']}", "split": "external_eval", "edit_type": "original", "family_id": it["id"]})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".jsonl.part")
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(MANIFEST)
    with open(OUT / "ATTRIBUTION.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in ("image_id", "commons_title", "attribution", "license", "license_url", "source_url")}) + "\n")
    film = [r for r in rows if r["kind"] == "film"]; counts = state.counts()
    pending = sum(counts.get(k, 0) for k in ("pending", "downloading", "retryable_failure")) + (len(items) - sum(counts.values()))
    summary = {"rows": len(rows), "film": len(film), "digital_emulation": len(rows) - len(film), "target_items": len(items), "pending_items": pending,
               "complete": pending == 0, "item_states": counts,
               "licences": dict(Counter(r["license"] for r in rows)), "film_types": dict(Counter(r["film_type"] for r in film)),
               "stocks": dict(Counter(r["film_stock_claim"] for r in film).most_common()), "uploaders": len({r["uploader"] for r in film}),
               "with_scanner": sum(bool(r["scanner_category"]) for r in film), "with_roll": sum(bool(r["roll_category"]) for r in film),
               "bytes": sum(Path(r["path"]).stat().st_size for r in rows), "manifest": str(MANIFEST), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def download_items(index, limit=None):
    img_dir = OUT / "images"; emu_dir = OUT / "emulation"
    items = []
    for kind, pool, d in (("film", index["files"], img_dir), ("digital_emulation", index["emulation_files"], emu_dir)):
        for r in pool.values():
            if not r.get("license") or r.get("mime") not in ("image/jpeg", "image/png", "image/tiff"):
                continue
            fid = "commons_" + hashlib.sha1(r["title"].encode()).hexdigest()[:16]          # stable id: no duplicate downloads across runs
            items.append({"id": fid, "urls": rendition_urls(r), "dst": d / f"{fid}.jpg", "rec": r, "kind": kind})
    return items[: limit or None]


def do_download(limit=None, max_minutes=180, stall_minutes=20, spacing=0.5):
    """Resumable download (research.downloader): per-file state in download_state.jsonl, atomic writes, bounded retries,
    429/Retry-After honoured with a bounded wait, no-progress (no HTTP response) timeout, wall-clock cap. Exit status
    0 when every item is settled, 3 when paused/stalled/rate-limited (re-run to resume)."""
    from research.downloader import Downloader
    index = json.loads(INDEX.read_text())
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "emulation").mkdir(parents=True, exist_ok=True)
    items = download_items(index, limit)
    dl = Downloader(OUT / "download_state.jsonl", headers=UA, spacing=spacing, max_attempts=4, max_retry_after=120, max_rate_limit_waits=20,
                    stall_seconds=stall_minutes * 60, max_seconds=max_minutes * 60 if max_minutes else None, heartbeat_seconds=60,
                    log_path=OUT / "download.log", checkpoint_every=100)
    dl.on_checkpoint = lambda: write_outputs(index, items, dl.state)
    st = dl.run(items, store_image)
    summary = write_outputs(index, items, dl.state); summary["run_status"] = st["run_status"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("rows", "film", "digital_emulation", "target_items", "pending_items", "complete", "item_states", "run_status")}, indent=1))
    return 0 if st["run_status"] == "complete" else 3


def do_hidden_cats():
    """Backfill scanner / camera categories: "Scanned with ..." and "Taken with ..." are HIDDEN categories on Commons,
    which the listing's clshow=!hidden query skipped. Writes into index.json (and the manifest if it exists)."""
    index = json.loads(INDEX.read_text())
    for kind in ("files", "emulation_files"):
        titles = list(index[kind]); n_s = 0
        for i in range(0, len(titles), 50):
            d = q(action="query", prop="categories", clshow="hidden", cllimit=200, titles="|".join(titles[i:i + 50]))
            for pg in d["query"]["pages"].values():
                cats_f = [x["title"] for x in pg.get("categories", [])]; rec = index[kind].get(pg["title"])
                if rec is None:
                    continue
                rec["scanner_cats"] = [c for c in cats_f if c.startswith("Category:Scanned with ")]; rec["camera_cats"] = [c for c in cats_f if c.startswith("Category:Taken with ")]
                rec["hidden_cats"] = cats_f; n_s += bool(rec["scanner_cats"])
        print(f"{kind}: {n_s} with scanner category", flush=True)
    INDEX.write_text(json.dumps(index))
    if MANIFEST.exists():
        by_title = {**index["files"], **index["emulation_files"]}; rows = [json.loads(l) for l in open(MANIFEST)]
        for r in rows:
            rec = by_title.get(r["commons_title"], {}); r["scanner_category"] = (rec.get("scanner_cats") or [None])[0]; r["camera_category"] = (rec.get("camera_cats") or [None])[0]
        with open(MANIFEST, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("manifest updated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--download", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hidden-cats", action="store_true", help="backfill scanner/camera (hidden) categories into the index and manifest")
    ap.add_argument("--max-minutes", type=float, default=180, help="wall-clock cap for one download run (0 = none); re-run to resume")
    ap.add_argument("--stall-minutes", type=float, default=20, help="exit when no HTTP response arrives for this long")
    ap.add_argument("--spacing", type=float, default=0.5, help="seconds between file requests")
    a = ap.parse_args()
    if a.list:
        do_list()
    if a.hidden_cats:
        do_hidden_cats()
    if a.download:
        raise SystemExit(do_download(a.limit, a.max_minutes or None, a.stall_minutes, a.spacing))
