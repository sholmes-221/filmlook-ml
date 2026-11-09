#!/usr/bin/env python
"""Museum artworks (non-photographs) from the Art Institute of Chicago, public domain (CC0), by catalogue class.

  python scripts/photo_gate_aic.py [--per-class 220]

Label comes from the museum's own classification (an owner record): painting and print -> training negatives;
"drawing and watercolor" and poster -> held-out test negatives (classes never seen in training). Requests are
serial, >= 1.2 s apart, with the AIC-User-Agent header; images are IIIF renditions 600 px wide.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402

UA = {"User-Agent": "FilmLook-ML-research/0.2 (https://github.com/sholmes-221/filmlook-ml; photo-gate dataset)", "AIC-User-Agent": "filmlook-ml (photo-gate dataset)"}
OUT = P.DATA / "raw" / "photo_gate" / "aic"
CLASSES = {"painting": "train", "print": "train", "drawings (visual works)": "test", "poster": "test"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=220)
    a = ap.parse_args()
    s = requests.Session(); man = []
    for cls, split in CLASSES.items():
        d = OUT / cls.replace(" ", "_").replace("(", "").replace(")", ""); d.mkdir(parents=True, exist_ok=True)
        q = {"query": {"bool": {"must": [{"term": {"is_public_domain": True}}, {"match_phrase": {"classification_titles": cls}}], "must_not": [{"match_phrase": {"classification_titles": "photograph"}}]}},
             "fields": "id,title,artist_title,image_id,classification_title,date_display", "limit": 100}
        rows, page = [], 1
        while len(rows) < a.per_class and page <= 5:
            time.sleep(1.2)
            r = s.post("https://api.artic.edu/api/v1/artworks/search", json={**q, "page": page}, headers=UA, timeout=60); r.raise_for_status()
            rows += [x for x in r.json()["data"] if x.get("image_id")]; page += 1
        for x in rows[: a.per_class]:
            p = d / f"aic_{x['id']}.jpg"
            if not p.exists():
                time.sleep(1.2)
                r = s.get(f"https://www.artic.edu/iiif/2/{x['image_id']}/full/600,/0/default.jpg", headers=UA, timeout=60)
                if r.status_code != 200 or len(r.content) < 5000:
                    continue
                p.write_bytes(r.content)
            man.append({"path": str(p.relative_to(P.DATA)), "kind": "artwork_" + cls.split(" (")[0].replace(" ", "_"), "family": cls, "label": 0, "split": split,
                        "generator": "aic", "licence": "CC0 (AIC public domain)", "source_url": f"https://www.artic.edu/artworks/{x['id']}", "classification": x.get("classification_title")})
        print(cls, len([m for m in man if m["family"] == cls]), flush=True)
    (OUT / "manifest_aic.jsonl").write_text("\n".join(json.dumps(m) for m in man) + "\n")


if __name__ == "__main__":
    main()
