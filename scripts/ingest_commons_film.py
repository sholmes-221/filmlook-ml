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
upload.wikimedia.org as server-side renditions, not originals. Digital film *emulation* categories (DxO FilmPack)
are excluded from the film set and indexed separately as `digital_emulation` controls.
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

from filmlook_ml import paths as P

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
        time.sleep(0.6)
        r = requests.get(API, params=p, headers=UA, timeout=60)
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
    print(f"{len(cats)} categories, {len(stock_of_cat)} stock/roll categories, {len(files)} film files, {len(emu)} emulation files")
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


def do_download(limit=None):
    index = json.loads(INDEX.read_text()); cat_info = index["categories"]
    img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    emu_dir = OUT / "emulation"; emu_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    todo = [("film", r, img_dir) for r in index["files"].values()] + [("digital_emulation", r, emu_dir) for r in index["emulation_files"].values()]
    for kind, r, d in todo[: limit or None]:
        if not r.get("license") or r.get("mime") not in ("image/jpeg", "image/png", "image/tiff"):
            skipped["licence_or_mime"] += 1; continue
        url = r.get("thumburl") or r.get("url")
        if not url:
            skipped["no_url"] += 1; continue
        fid = "commons_" + hashlib.sha1(r["title"].encode()).hexdigest()[:16]
        dst = d / f"{fid}.jpg"
        if not dst.exists():
            for attempt in range(5):
                time.sleep(0.4)
                resp = requests.get(url, headers=UA, timeout=120)
                if resp.status_code == 200:
                    dst.write_bytes(resp.content); break
                if resp.status_code in (429, 503):
                    time.sleep(15 * (attempt + 1)); continue
                skipped[f"http_{resp.status_code}"] += 1; break
            if not dst.exists():
                continue
        try:
            from PIL import Image
            with Image.open(dst) as im:
                w, h = im.size
        except Exception:
            dst.unlink(missing_ok=True); skipped["unreadable"] += 1; continue
        stock_cat = r["stock_cats"][0] if r["stock_cats"] else None
        info = cat_info.get(stock_cat, {}) if stock_cat else {}
        rows.append({"image_id": fid, "path": str(dst), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "width": w, "height": h,
                     "dataset_id": "commons_film", "kind": kind, "license": r["license"], "license_raw": r["license_raw"], "license_url": r["license_url"],
                     "attribution": re.sub(r"<[^>]+>", "", r.get("artist_html") or r.get("uploader") or ""), "uploader": r["uploader"], "source_url": r["descriptionurl"],
                     "commons_title": r["title"], "commons_sha1": r["sha1"], "original_size": [r.get("width"), r.get("height")], "rendition_max_width": MAX_WIDTH,
                     "medium": "film_scan" if kind == "film" else "digital_emulation",
                     "film_stock_claim": info.get("stock") or (norm_stock(stock_cat) if stock_cat else None), "stock_label_source": "commons_category",
                     "film_type": info.get("film_type", "unknown"), "stock_category": stock_cat, "roll_category": r.get("roll_cat"),
                     "scanner_category": (r.get("scanner_cats") or [None])[0], "camera_category": (r.get("camera_cats") or [None])[0],
                     "date_original": r.get("date_original"), "exif_make": r.get("exif_make"), "exif_model": r.get("exif_model"), "software": r.get("software"),
                     "group_id": f"commons:{r['uploader']}", "split": "external_eval", "edit_type": "original", "family_id": fid})
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} downloaded", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with open(OUT / "ATTRIBUTION.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in ("image_id", "commons_title", "attribution", "license", "license_url", "source_url")}) + "\n")
    film = [r for r in rows if r["kind"] == "film"]
    summary = {"rows": len(rows), "film": len(film), "digital_emulation": len(rows) - len(film), "skipped": dict(skipped),
               "licences": dict(Counter(r["license"] for r in rows)), "film_types": dict(Counter(r["film_type"] for r in film)),
               "stocks": dict(Counter(r["film_stock_claim"] for r in film).most_common()), "uploaders": len({r["uploader"] for r in film}),
               "with_scanner": sum(bool(r["scanner_category"]) for r in film), "with_roll": sum(bool(r["roll_category"]) for r in film),
               "bytes": sum(Path(r["path"]).stat().st_size for r in rows), "manifest": str(MANIFEST)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--download", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.list:
        do_list()
    if a.download:
        do_download(a.limit)
