#!/usr/bin/env python
"""Download eligible works from the licence manifest (resumable), validate them and deduplicate.

  python scripts/refdb_download.py [--cap 300] [--only id,id] [--max-minutes 120]

Only rows with rights_status = 'eligible' (public domain or CC0, see refdb.rights) are fetched. Per photographer at
most --cap works are taken, chosen by a stable hash of the work id (not by popularity) so reruns pick the same set.
Files are the sources' scaled renditions (Commons thumbnails at 1024 px width, AIC IIIF at 843 px), never the
originals. Each file is decoded (corrupt files are quarantined), sized, hashed (sha256 and a perceptual hash) and
marked monochrome or colour. Near-duplicates (phash Hamming distance <= 6 within the same photographer, or <= 4
across photographers) are marked duplicate_of and left out of every later step: archives often hold several scans of
one negative. Rate limits: one request at a time, --spacing seconds apart, Retry-After honoured
(research.downloader.Downloader).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from research.downloader import Downloader  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import hamming, phash  # noqa: E402
from filmlook_ml.refdb import manifest  # noqa: E402
from filmlook_ml.refdb.sources import UA  # noqa: E402


def is_monochrome(img: Image.Image) -> bool:
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    chroma = a.max(axis=2) - a.min(axis=2)
    return float(np.percentile(chroma, 95)) < 18.0  # Toned prints (sepia) count as monochrome: low chroma spread, one hue.


def toned_monochrome(img: Image.Image) -> bool:
    """Sepia/albumen prints: high chroma but a single hue. Treated as monochrome for the colour flag."""
    hsv = np.asarray(img.convert("RGB").resize((128, 128)).convert("HSV"), dtype=np.float32)
    sat = hsv[..., 1] > 30
    if sat.mean() < 0.2:
        return True
    hue = hsv[..., 0][sat]
    return float(np.std(hue)) < 12.0


def pick(rows: list, cap: int) -> list:
    return sorted(rows, key=lambda r: hashlib.sha256(r["work_id"].encode()).hexdigest())[:cap]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=300); ap.add_argument("--only", default="")
    ap.add_argument("--max-minutes", type=float, default=180); ap.add_argument("--spacing", type=float, default=1.0)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    with manifest.connect() as c:
        rows = [dict(r) for r in c.execute("SELECT work_id, photographer_id, image_url FROM works WHERE rights_status = 'eligible' AND image_url IS NOT NULL")]
    by_p = defaultdict(list)
    for r in rows:
        if not only or r["photographer_id"] in only:
            by_p[r["photographer_id"]].append(r)
    chosen = [r for p, rs in sorted(by_p.items()) for r in pick(rs, a.cap)]
    items = [{"id": r["work_id"], "urls": [r["image_url"]], "dst": manifest.image_path(r["work_id"])} for r in chosen]

    def validate(content: bytes, tmp: Path) -> None:
        img = Image.open(io.BytesIO(content)); img.load()
        if min(img.size) < 256:
            raise ValueError(f"too small {img.size}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(tmp, "JPEG", quality=92)  # The downloader moves it into place atomically.

    dl = Downloader(P.DATA / "refdb" / "download_state.jsonl", headers={"User-Agent": UA["User-Agent"]}, spacing=a.spacing,
                    max_seconds=a.max_minutes * 60, log_path=P.REPORTS / "refdb" / "download.log")
    status = dl.run(items, validate)

    # Record file facts and deduplicate.
    with manifest.connect() as c:
        seen: list[tuple[str, str, str]] = []  # (phash, photographer, work_id)
        for r in chosen:
            path = manifest.image_path(r["work_id"])
            if not path.exists():
                continue
            img = Image.open(path); img.load()
            h = phash(img)
            mono = is_monochrome(img) or toned_monochrome(img)
            dup = None
            for ph, pid, wid in seen:
                d = hamming(h, ph)
                if (pid == r["photographer_id"] and d <= 6) or d <= 4:
                    dup = wid; break
            if dup is None:
                seen.append((h, r["photographer_id"], r["work_id"]))
            c.execute("UPDATE works SET sha256 = ?, phash = ?, width = ?, height = ?, is_monochrome = ?, duplicate_of = ? WHERE work_id = ?",
                      (hashlib.sha256(path.read_bytes()).hexdigest(), h, img.width, img.height, int(mono), dup, r["work_id"]))
    print(status)


if __name__ == "__main__":
    main()
