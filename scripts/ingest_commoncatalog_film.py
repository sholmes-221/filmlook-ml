#!/usr/bin/env python
"""commoncatalog_film: an evaluation-only real-film set from CommonCatalog CC-BY film scans whose uploader tagged the stock.

  python scripts/ingest_commoncatalog_film.py --scan --shards 240          # metadata-only shard scan (resumable; no pixels)
  python scripts/ingest_commoncatalog_film.py --download [--max-minutes 120] # select + fetch (resumable; exit 0 complete, 3 resumable)
  python scripts/ingest_commoncatalog_film.py --status

Source: https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by (YFCC100M Flickr photos, CC-BY per row; the
Hub's dataset tag is not relied on). Rows qualify when (1) the per-row licenceurl is creativecommons.org/licenses/by/<v>/,
(2) the EXIF capture device is a film scanner / minilab (filmlook_ml.datasets.commoncatalog.classify -> medium film_scan),
(3) the uploader's own tags name a film stock (filmlook_ml.datasets.film_stock.stock_claim; brand-only rows are kept with
film_stock = null, never as a negative label), (4) the uploader is not in the training manifest (no overlap with the
students' distillation inputs). Authenticity class B (scanner EXIF + uploader claim); the claim is never verified from
pixels. Images come from the Hub parquet files by page-range reads (official mechanism; no Flickr scraping), are stored
as the original JPEG bytes (no recompression), sha256-verified, perceptual-hashed, and split into analysis/test halves by
uploader hash. Caps: per uploader 20, per uploader x stock 6, per stock 150 (no single-stock dominance).
Provenance: author, page URL, licence URL per row (ATTRIBUTION.jsonl). training_allowed: false (evaluation only).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.downloader import DownloadState, sha256_file  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.commoncatalog import REPO  # noqa: E402
from filmlook_ml.datasets.film_stock import stock_claim  # noqa: E402
from filmlook_ml.datasets.preprocess import hamming, phash, read_jsonl  # noqa: E402

DS = "commoncatalog_film"
RAW = P.RAW / DS; IMG = RAW / "images"; CAND = P.DATA / "candidates" / "commoncatalog_film_candidates.jsonl"; SCANNED = P.DATA / "candidates" / "commoncatalog_film_scanned.json"
MANIFEST = P.MANIFESTS / f"{DS}_manifest.jsonl"; STATE = RAW / "download_state.jsonl"; LOG = RAW / "download.log"
CAPS = {"per_uploader": 20, "per_uploader_stock": 6, "per_stock": 150}
MIN_SIDE = 1024


def log(msg):
    RAW.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"; print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def half_of(uploader: str) -> str:
    return "analysis" if int.from_bytes(hashlib.sha256((DS + ":" + str(uploader)).encode()).digest()[:2], "little") % 2 == 0 else "test"


def training_uploaders() -> set[str]:
    man = P.MANIFESTS / "manifest.jsonl"
    return {r["group_id"].split(":", 1)[1] for r in read_jsonl(man) if str(r.get("group_id", "")).startswith("flickr:")} if man.exists() else set()


# ------------------------------------------------------------------------------------------------- metadata scan
def scan_shard(path: str) -> list[dict]:
    """Metadata columns only (never the jpg column); returns qualifying rows with the stock claim attached."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from filmlook_ml.datasets.commoncatalog import META_COLS, classify
    fs = HfFileSystem(); pf = pq.ParquetFile(fs.open(f"datasets/{REPO}/{path}", block_size=1 << 20)); out = []
    for g in range(pf.metadata.num_row_groups):
        for i, r in enumerate(pf.read_row_group(g, columns=META_COLS).to_pylist()):
            c = classify(r)
            if not c["license_ok"] or c["exclude"] or c["medium"] != "film_scan":
                continue
            claim = stock_claim(r.get("usertags"))
            if not claim["film_stock"] and not claim["film_brand"]:
                continue
            r.pop("exif", None); r["datetaken"] = str(r.get("datetaken"))
            out.append({"file": path, "row_group": g, "row": i, **{k: r.get(k) for k in META_COLS if k != "exif"}, **c, **claim})
    return out


def do_scan(shards: int, rng_range: str, seed: int, workers: int = 4) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".parquet") and rng_range in s.rfilename)
    random.Random(seed).shuffle(files)
    done = set(json.loads(SCANNED.read_text())["files"]) if SCANNED.exists() else set()
    todo = [f for f in files if f not in done][:shards]; CAND.parent.mkdir(parents=True, exist_ok=True)
    log(f"scan: {len(files)} shards in {rng_range}, {len(done)} scanned before, {len(todo)} now")
    n = 0
    with open(CAND, "a") as fh, ThreadPoolExecutor(workers) as ex:
        for k, (f, rows) in enumerate(zip(todo, ex.map(lambda f: _safe(scan_shard, f), todo))):
            if rows is None:
                log(f"  shard failed (will retry next run): {f[-40:]}"); continue
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
            n += len(rows); done.add(f)
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                SCANNED.write_text(json.dumps({"files": sorted(done), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})); log(f"  [{k + 1}/{len(todo)}] +{n} qualifying rows")
    SCANNED.write_text(json.dumps({"files": sorted(done), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return {"scanned": len(done), "new_rows": n}


def _safe(fn, *a):
    for attempt in range(3):
        try:
            return fn(*a)
        except Exception as ex:                                          # network hiccup: bounded retry
            time.sleep(5 * 2 ** attempt); err = ex
    log(f"  giving up: {err!r}"); return None


# ------------------------------------------------------------------------------------------------- selection
def select(cands: list[dict], seed: int = 0, caps: dict = CAPS, exclude_uploaders: set[str] | None = None) -> list[dict]:
    exclude_uploaders = exclude_uploaders or set(); seen = set(); pool = []
    for c in cands:
        if c["photoid"] in seen or c["uid"] in exclude_uploaders or min(int(c["width"]), int(c["height"])) < MIN_SIDE or c.get("conflict"):
            continue
        seen.add(c["photoid"]); pool.append(c)
    random.Random(seed).shuffle(pool)
    pool.sort(key=lambda c: (c["film_stock"] is None,))                 # stock-labelled rows first, brand-only after
    per_up, per_us, per_stock, out = Counter(), Counter(), Counter(), []
    for c in pool:
        s = c["film_stock"] or "brand_only"
        if per_up[c["uid"]] >= caps["per_uploader"] or per_us[(c["uid"], s)] >= caps["per_uploader_stock"] or per_stock[s] >= caps["per_stock"]:
            continue
        per_up[c["uid"]] += 1; per_us[(c["uid"], s)] += 1; per_stock[s] += 1; out.append(c)
    return out


# ------------------------------------------------------------------------------------------------- download
def record_for(c: dict, path: Path, sha: str, ph: str) -> dict:
    return {"image_id": f"cc_{c['photoid']}", "source_dataset": DS, "dataset_id": DS, "path": str(path), "source_url": c["pageurl"], "original_image_url": None,
            "photographer_id": f"flickr:{c['uid']}", "uploader_id": f"flickr:{c['uid']}", "uploader": f"flickr:{c['uid']}", "author": c.get("unickname"),
            "film_stock": c["film_stock"], "film_stock_claim": c["film_stock"], "film_stock_line": c.get("film_stock_line"), "film_brand": c["film_brand"], "film_type": c["film_type"] or "unknown",
            "stock_label_source": "uploader_tag", "stock_evidence_tags": c.get("evidence_tags"), "film_format": None, "camera": None, "scanner": c.get("camera"), "scanner_category": _scanner_family(c.get("camera")),
            "scan_process": c.get("software"), "development_process": None, "authenticity_class": "B",
            "authenticity_evidence": "EXIF capture device is a film scanner/minilab; film stock is the uploader's own tag",
            "kind": "film", "license": c["license"], "license_url": c["licenseurl"], "commercial_use_allowed": True, "training_use_allowed": False,
            "image_width": int(c["width"]), "image_height": int(c["height"]), "sha256": sha, "perceptual_hash": ph, "split": half_of(c["uid"]),
            "datetaken": c.get("datetaken"), "caption": c.get("blip2_caption"), "subjects": c.get("subjects"), "monochrome_tag": c.get("monochrome_tag")}


def _scanner_family(camera):
    sys.path.insert(0, str(Path(__file__).parent))
    from scanner_effects import family
    return family(camera) if camera else ""


def write_outputs(state: DownloadState, selected: list[dict], complete: bool) -> dict:
    by = {c["photoid"]: c for c in selected}; rows, attr = [], []
    for pid, st in state.rows.items():
        if st.get("status") != "validated" or pid not in by:
            continue
        c = by[pid]; rows.append(record_for(c, Path(st["dst"]), st["sha256"], st.get("phash", "")))
        attr.append({"file": Path(st["dst"]).name, "author": c.get("unickname"), "page": c["pageurl"], "license": c["license"], "license_url": c["licenseurl"], "title": c.get("title")})
    rows.sort(key=lambda r: r["image_id"]); MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".jsonl.part"); tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows)); os.replace(tmp, MANIFEST)
    (RAW / "ATTRIBUTION.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attr))
    stocks = Counter(r["film_stock"] for r in rows if r["film_stock"]); ups = Counter(r["uploader"] for r in rows)
    summary = {"dataset": DS, "complete": complete, "rows": len(rows), "target_items": len(selected), "item_states": state.counts(), "n_uploaders": len(ups),
               "n_stocks": len(stocks), "stocks_top": stocks.most_common(20), "max_uploader_share": round(max(ups.values()) / len(rows), 3) if rows else None,
               "splits": dict(Counter(r["split"] for r in rows)), "film_types": dict(Counter(r["film_type"] for r in rows)), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1)); return summary


def do_download(max_minutes: float, seed: int, workers: int = 2) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfFileSystem

    from filmlook_ml.datasets.commoncatalog import fetch_rows, jpeg_ok
    if not CAND.exists():
        log("no candidates: run --scan first"); return 2
    cands = [json.loads(l) for l in open(CAND)]; selected = select(cands, seed, exclude_uploaders=training_uploaders())
    IMG.mkdir(parents=True, exist_ok=True); state = DownloadState(STATE); t0 = time.time(); deadline = t0 + max_minutes * 60
    todo = [c for c in selected if state.get(c["photoid"]).get("status") not in ("validated", "permanent_failure")]
    log(f"download: {len(selected)} selected from {len(cands)} candidates ({len({c['uid'] for c in selected})} uploaders); {len(todo)} to fetch; state {state.counts()}")
    groups = defaultdict(list)
    for c in todo:
        groups[(c["file"], int(c["row_group"]))].append(c)
    known_hashes = {st["phash"] for st in state.rows.values() if st.get("phash")}
    fs = HfFileSystem(); done = fails = 0; stopped = False

    def fetch(file, rg, rows):
        for attempt in range(4):
            try:
                return fetch_rows(fs, f"datasets/{REPO}/{file}", rg, {int(r["row"]) for r in rows}), None
            except Exception as ex:
                if attempt == 3:
                    return None, repr(ex)[:200]
                time.sleep(3 * 2 ** attempt)

    with ThreadPoolExecutor(workers) as ex:
        pending = {}; it = iter(groups.items())
        def submit_next():
            try:
                (f, rg), rows = next(it)
            except StopIteration:
                return False
            pending[ex.submit(fetch, f, rg, rows)] = rows; return True
        for _ in range(workers * 2):
            submit_next()
        while pending:
            if time.time() > deadline:
                stopped = True; log("max runtime reached; pausing (resumable)"); break
            fut = next(as_completed(pending)); rows = pending.pop(fut); got, err = fut.result()
            for c in rows:
                pid = c["photoid"]; dst = IMG / f"cc_{pid}.jpg"
                if err or not got:
                    state.set(pid, "retryable_failure", error=err or "no data", dst=str(dst)); fails += 1; continue
                b = got.get(int(c["row"]))
                if not b or not jpeg_ok(b) or len(b) > 30 * 2 ** 20:
                    state.set(pid, "permanent_failure", error="missing/not jpeg/too large", dst=str(dst)); fails += 1; continue
                try:
                    from PIL import Image
                    im = Image.open(io.BytesIO(b)); im.verify(); im = Image.open(io.BytesIO(b)).convert("RGB")
                    if min(im.size) < 64:
                        raise ValueError("too small")
                    ph = phash(im)
                except Exception as e2:
                    state.set(pid, "permanent_failure", error=f"decode: {e2!r}"[:160], dst=str(dst)); fails += 1; continue
                if any(hamming(ph, h) <= 4 for h in known_hashes):
                    state.set(pid, "permanent_failure", error="near-duplicate (phash <= 4) of a stored image", dst=str(dst)); fails += 1; continue
                tmp = dst.with_suffix(".jpg.part"); tmp.write_bytes(b); os.replace(tmp, dst)
                state.set(pid, "validated", dst=str(dst), sha256=hashlib.sha256(b).hexdigest(), phash=ph, bytes=len(b)); known_hashes.add(ph); done += 1
            submit_next()
            if (done + fails) % 50 == 0:
                write_outputs(state, selected, False); log(f"  progress: {done} stored, {fails} failed this run, state {state.counts()}, {round(time.time() - t0)}s")
    remaining = [c for c in selected if state.get(c["photoid"]).get("status") not in ("validated", "permanent_failure")]
    complete = not remaining and not stopped
    s = write_outputs(state, selected, complete)
    log(f"run finished: {'complete' if complete else 'paused/partial'}; stored {done}, failed {fails}; rows {s['rows']}/{s['target_items']}, uploaders {s['n_uploaders']}, stocks {s['n_stocks']}")
    return 0 if complete else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true"); ap.add_argument("--download", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--shards", type=int, default=120); ap.add_argument("--range", default="least_dim_range=1024-2048"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=120); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); rc = 0
    if a.scan:
        print(json.dumps(do_scan(a.shards, a.range, a.seed)))
    if a.download:
        rc = do_download(a.max_minutes, a.seed, a.workers)
    if a.status:
        print((RAW / "summary.json").read_text() if (RAW / "summary.json").exists() else "no summary yet")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
