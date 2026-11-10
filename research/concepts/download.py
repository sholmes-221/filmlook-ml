"""Fetch the selected Open Images photographs from the CVDF-hosted copy and keep only a small local thumbnail.

  python -m research.concepts.download [--workers 8]

Source: the Common Visual Data Foundation's public S3 copy of Open Images (the distribution route the Open Images team
documents for bulk image download), https://open-images-dataset.s3.amazonaws.com/<split>/<id>.jpg. Each image is
decoded (images flagged as needing rotation were excluded at selection), resized so its shorter side is 320 px (DINOv2 preprocessing then only
downsizes to 256 as in production), saved as JPEG q90; the original bytes are discarded. Bounded worker pool; HTTP 429/503
honour Retry-After; a generic User-Agent. Writes data/concepts/manifest.jsonl (licence, author, source URLs per image).
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

from research.concepts.common import CVDF, DATA, IMAGES, USER_AGENT, write_jsonl

SHORT = 320
_local = threading.local()


def sess() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = USER_AGENT
    return _local.s


def fetch(row) -> tuple[str, str]:
    out = IMAGES / row.image_id[:2] / f"{row.image_id}.jpg"
    if out.exists():
        return row.image_id, "ok"
    split = {"train": "train", "validation": "validation", "test": "test"}[row.oi_split]
    for attempt in range(5):
        try:
            r = sess().get(CVDF.format(split=split, id=row.image_id), timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1)); continue
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After")
            time.sleep(min(300, float(ra) if ra and ra.isdigit() else 15 * (attempt + 1))); continue
        if r.status_code != 200:
            return row.image_id, f"http_{r.status_code}"
        try:
            im = Image.open(io.BytesIO(r.content)); im.load(); im = im.convert("RGB")
        except Exception:
            return row.image_id, "decode"
        s = SHORT / min(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=90); tmp.rename(out)
        return row.image_id, "ok"
    return row.image_id, "retries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = pd.read_parquet(DATA / "selection.parquet")
    status = {}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (iid, st) in enumerate(ex.map(fetch, rows.itertuples()), 1):
            status[iid] = st
            if n % 2000 == 0:
                print(f"{n}/{len(rows)} {time.time() - t0:.0f}s fails={sum(v != 'ok' for v in status.values())}", flush=True)
    man = []
    for r in rows.itertuples():
        if status.get(r.image_id) == "ok":
            man.append({"image_id": r.image_id, "path": f"images/{r.image_id[:2]}/{r.image_id}.jpg", "license": "CC-BY-2.0",
                        "license_url": r.License, "author": r.author, "author_url": r.AuthorProfileURL,
                        "source_page": r.OriginalLandingURL, "original_url": r.OriginalURL,
                        "fetched_from": CVDF.format(split=r.oi_split, id=r.image_id),
                        "labels": "Open Images V7 human-verified image-level labels (CC BY 4.0, Google LLC)"})
    write_jsonl(DATA / "manifest.jsonl", man)
    print(f"done {len(man)}/{len(rows)} in {time.time() - t0:.0f}s; failures: {pd.Series([v for v in status.values() if v != 'ok']).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
