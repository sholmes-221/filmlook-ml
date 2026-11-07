#!/usr/bin/env python
"""artefact_film: ARTeFACT 35mm scans (Ivanova et al., Eurographics 2023) from figshare, evaluation-only.

  python scripts/ingest_artefact_film.py --download [--max-minutes 60]   # resumable; exit 0 complete, 3 resumable, 2 permanent

Source: https://figshare.com/articles/dataset/Authentically_damaged_film_scans/21803304 (doi 10.6084/m9.figshare.21803304),
CC BY 4.0, author Daniela Ivanova. 44 damaged 35mm scans (slides and negatives) of 11 named emulsions, each with a manually
restored counterpart (88 JPEG files). File names carry the emulsion and format: <emulsion><speed>_<135|half>_<n>[_restored].jpg.
Authenticity class A: the paper documents the physical film, emulsions, scanning and manual restoration. Files are stored as
the original bytes (no recompression); the figshare md5 and our sha256 are recorded. Single photographer: every retrieval
metric on this set must be read as within-photographer, and the set serves authenticity / per-emulsion sanity checks, not
selection. training_allowed: false (evaluation only, by project policy; the licence itself permits more).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.downloader import DownloadState, Downloader  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.film_stock import stock_claim  # noqa: E402
from filmlook_ml.datasets.preprocess import phash  # noqa: E402

DS = "artefact_film"; ARTICLE = 21803304
RAW = P.RAW / DS; IMG = RAW / "images"; MANIFEST = P.MANIFESTS / f"{DS}_manifest.jsonl"; STATE = RAW / "download_state.jsonl"; LOG = RAW / "download.log"
NAME_RE = re.compile(r"^(?P<emulsion>[a-z]+?)(?P<speed>\d+[a-z]?)_(?P<fmt>135|half)_(?P<n>\d+)(?P<restored>_restored)?\.jpg$")
UA = {"User-Agent": "filmlook-ml-dataset-acquisition/0.1 (research; python-urllib)"}


def article() -> dict:
    with urllib.request.urlopen(urllib.request.Request(f"https://api.figshare.com/v2/articles/{ARTICLE}", headers=UA), timeout=40) as r:
        return json.load(r)


def parse_name(name: str) -> dict | None:
    m = NAME_RE.match(name)
    if not m:
        return None
    emu, speed = m["emulsion"], m["speed"]; claim = stock_claim(f"{emu} {speed}")
    return {"emulsion": emu + speed, "film_stock": claim["film_stock"] or f"{emu}_{speed}", "film_brand": claim["film_brand"], "film_type": claim["film_type"] or "unknown",
            "film_format": "35mm half-frame" if m["fmt"] == "half" else "35mm", "frame": int(m["n"]), "restored": bool(m["restored"]), "frame_id": f"{emu}{speed}_{m['fmt']}_{m['n']}"}


def make_validator(md5_by_name: dict[str, str]):
    """One validator for all items: the figshare md5 is looked up by the destination file name (tmp = <name>.part)."""
    def validate(content: bytes, tmp: Path):
        from PIL import Image
        md5 = md5_by_name.get(tmp.name[:-5] if tmp.name.endswith(".part") else tmp.name)
        if md5 and hashlib.md5(content).hexdigest() != md5:
            raise ValueError("md5 mismatch against figshare record")
        im = Image.open(io.BytesIO(content)); im.verify()
        tmp.write_bytes(content)                                   # original bytes, no recompression
    return validate


def write_outputs(files: list[dict], state: DownloadState, complete: bool) -> dict:
    rows, attr = [], []
    for f in files:
        st = state.get(f["id"])
        if st.get("status") != "validated":
            continue
        p = Path(f["dst"]); meta = f["meta"]
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size; ph = phash(im.convert("RGB"))
        except Exception:
            w = h = None; ph = None
        rows.append({"image_id": f["id"], "source_dataset": DS, "dataset_id": DS, "path": str(p), "source_url": f"https://figshare.com/articles/dataset/Authentically_damaged_film_scans/{ARTICLE}",
                     "original_image_url": f["url"], "photographer_id": "ivanova_daniela", "uploader_id": "figshare:Daniela_Ivanova", "uploader": "ivanova_daniela", "author": "Daniela Ivanova",
                     "film_stock": meta["film_stock"], "film_stock_claim": meta["film_stock"], "film_brand": meta["film_brand"], "film_type": meta["film_type"], "stock_label_source": "owner_record (file name, paper)",
                     "film_format": meta["film_format"], "camera": None, "scanner": None, "scanner_category": "", "scan_process": "high-resolution film scan (paper)", "development_process": None,
                     "authenticity_class": "A", "authenticity_evidence": "Eurographics 2023 paper: physical 35mm slides/negatives of named emulsions, scanned at 4K, manually restored counterparts",
                     "kind": "film_restored" if meta["restored"] else "film", "derived_from": meta["frame_id"] if meta["restored"] else None, "frame_id": meta["frame_id"],
                     "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "commercial_use_allowed": True, "training_use_allowed": False,
                     "image_width": w, "image_height": h, "sha256": st.get("sha256"), "md5_figshare": f["md5"], "perceptual_hash": ph, "split": "analysis"})
        attr.append({"file": p.name, "author": "Daniela Ivanova", "page": rows[-1]["source_url"], "license": "CC BY 4.0", "doi": "10.6084/m9.figshare.21803304"})
    rows.sort(key=lambda r: r["image_id"]); MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".jsonl.part"); tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); os.replace(tmp, MANIFEST)
    (RAW / "ATTRIBUTION.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attr))
    orig = [r for r in rows if r["kind"] == "film"]
    summary = {"dataset": DS, "complete": complete, "rows": len(rows), "target_items": len(files), "n_originals": len(orig), "n_restored": len(rows) - len(orig),
               "item_states": state.counts(), "stocks": dict(Counter(r["film_stock"] for r in orig)), "n_uploaders": 1, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1)); return summary


def run(max_minutes: float, spacing: float = 1.0) -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    try:
        art = article()
    except Exception as ex:
        print(f"figshare API unavailable: {ex!r} (resumable)"); return 3
    lic = (art.get("license") or {}).get("name", "")
    if "CC BY 4.0" not in lic:
        print(f"licence changed on figshare: {lic!r}; refusing to download"); return 2
    files = [{"id": "artefact_" + f["name"][:-4], "urls": [f["download_url"]], "dst": IMG / f["name"], "md5": f.get("computed_md5"), "url": f["download_url"], "meta": parse_name(f["name"])}
             for f in art["files"] if parse_name(f["name"])]
    dl = Downloader(STATE, headers=UA, spacing=spacing, max_seconds=max_minutes * 60, log_path=LOG, checkpoint_every=10)
    dl.on_checkpoint = lambda: write_outputs(files, dl.state, False)
    res = dl.run(files, make_validator({f["dst"].name: f["md5"] for f in files}))
    complete = res.get("run_status") == "complete"
    s = write_outputs(files, dl.state, complete)
    print(json.dumps({"complete": complete, "run_status": res.get("run_status"), "rows": s["rows"], "states": s["item_states"]}), flush=True)
    return 0 if complete else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true"); ap.add_argument("--max-minutes", type=float, default=60); ap.add_argument("--spacing", type=float, default=1.0)
    a = ap.parse_args()
    raise SystemExit(run(a.max_minutes, a.spacing) if a.download else 0)


if __name__ == "__main__":
    main()
