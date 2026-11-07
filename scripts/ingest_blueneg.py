#!/usr/bin/env python
"""Ingest the BlueNeg previews (ttgroup/blueneg-release) into a FilmLook-ML manifest. Read-only on the cache.

  python scripts/ingest_blueneg.py     # -> data/manifests/blueneg_manifest.jsonl, data/raw/blueneg/ATTRIBUTION.txt

Licence (BlueNeg Data Set License v1.0, 2024-06-12): academic and commercial use permitted with the credit
"Copyrighted by Tien-Tsin Wong" in any publication, reproduction, redistribution or derivative. The credit is written
into every manifest row and into ATTRIBUTION.txt. Film-stock labels come from the owner's records (meta.json), the
only owner-verified stock labels this project holds. BlueNeg is an EXTERNAL evaluation set: one photographer, so it is
never split into train/val/test and never used as a synthetic-training source.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from filmlook_ml import paths as P

CREDIT = "Copyrighted by Tien-Tsin Wong"
CITATION = ("Liu, Li, Xie, Wang, Liang, Leung, Wong. BlueNeg: A 35mm Negative Film Dataset for Restoring "
            "Channel-Heterogeneous Deterioration. ICCV 2025.")


def main():
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("ttgroup/blueneg-release", repo_type="dataset", cache_dir=str(P.DATA / "hf_cache"),
                                  local_files_only=True,
                                  allow_patterns=["LICENSE", "README.md", "meta.json", "negative-preview-8bit/*",
                                                  "pseudogt-8bit/*"]))
    meta = json.loads((root / "meta.json").read_text())
    out = P.MANIFESTS / "blueneg_manifest.jsonl"
    rows, missing = [], 0
    for m in meta:
        p = root / m["preview_path"]
        if not p.exists():
            missing += 1
            continue
        with Image.open(p) as im:
            w, h = im.size
        data = p.read_bytes()
        rows.append({"image_id": f"blueneg_{m['filename']}", "path": str(p), "sha256": hashlib.sha256(data).hexdigest(),
                     "width": w, "height": h, "dataset_id": "blueneg_release", "license": "BlueNeg-1.0 (commercial with credit)",
                     "attribution": CREDIT, "citation": CITATION, "source_url": "https://huggingface.co/datasets/ttgroup/blueneg-release",
                     "medium": "film_scan", "film_stock_verified": m["film_type"], "stock_label_source": "owner_record",
                     "partition": m["partition"], "is_testset": m["is_testset"], "roll_id": m["roll_id"],
                     "date": m["date"], "scene": m["scene_property"], "group_id": f"blueneg:{m['roll_id']}",
                     "split": "external_eval", "edit_type": "original", "family_id": f"blueneg_{m['filename']}",
                     "pseudogt_path": str(root / m["pseudogt_path"]) if m.get("pseudogt_path") and (root / m["pseudogt_path"]).exists() else None})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (P.RAW / "blueneg").mkdir(parents=True, exist_ok=True)
    (P.RAW / "blueneg" / "ATTRIBUTION.txt").write_text(
        f"{CREDIT}\n{CITATION}\nLicence: BlueNeg Data Set License v1.0 (2024-06-12), see the LICENSE file in the dataset.\n"
        f"Rows: {len(rows)} previews; stocks: {dict(Counter(r['film_stock_verified'] for r in rows))}\n")
    print(json.dumps({"rows": len(rows), "missing_previews": missing,
                      "stocks": dict(Counter(r["film_stock_verified"] for r in rows).most_common()),
                      "partitions": dict(Counter(r["partition"] for r in rows)), "manifest": str(out)}, indent=1))


if __name__ == "__main__":
    main()
