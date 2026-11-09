#!/usr/bin/env python
"""Independent test photographs from PD12M (CC0 / public domain per row): rows whose *source* is iNaturalist, a
citizen-science collection of field photographs. The label comes from the source, not from PD12M's model-written
captions. Images are fetched from PD12M's own S3 bucket (no Wikimedia load), serially, resized to 1024 px.

  python scripts/photo_gate_pd12m.py [--n 220]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402

OUT = P.DATA / "raw" / "photo_gate" / "pd12m_inaturalist"
UA = {"User-Agent": "FilmLook-ML-research/0.2 (https://github.com/sholmes-221/filmlook-ml; photo-gate test set)"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=220)
    a = ap.parse_args()
    import glob
    import pyarrow.parquet as pq
    f = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--Spawning--PD12M/snapshots/*/metadata/pd12m.000.parquet"))[0]
    rows = [r for r in pq.read_table(f, columns=["id", "url", "license", "source"]).to_pylist() if r["source"] == "iNaturalist"]
    random.Random(0).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True); man = []
    s = requests.Session()
    for r in rows:
        if len(man) >= a.n:
            break
        p = OUT / f"{r['id']}.jpg"
        if not p.exists():
            time.sleep(0.3)
            try:
                resp = s.get(r["url"], headers=UA, timeout=60)
                if resp.status_code != 200:
                    continue
                im = Image.open(io.BytesIO(resp.content)); im.load(); im = im.convert("RGB"); im.thumbnail((1024, 1024)); im.save(p, quality=90)
            except Exception:  # noqa: BLE001 - a broken file is skipped, not retried
                continue
        man.append({"path": str(p.relative_to(P.DATA)), "kind": "photo_inaturalist", "family": "pd12m", "label": 1, "split": "test", "generator": "pd12m:iNaturalist", "licence": r["license"]})
    (OUT / "manifest_pd12m.jsonl").write_text("\n".join(json.dumps(m) for m in man) + "\n")
    print(len(man))


if __name__ == "__main__":
    main()
