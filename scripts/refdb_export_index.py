#!/usr/bin/env python
"""Package the compact, versioned serving index (no images) for the website.

  python scripts/refdb_export_index.py --calibration reports/refdb/eval_dinov2s_cls_224_seed0.json [--out dist/refdb]

Contents: dinov2_small_cls.onnx (fp32; int8 failed the 0.99 cosine gate), vectors_cls.f16.npy (train+val works of
the known photographers), works.sqlite (metadata and licence per row), visual_dna.sqlite, manifest.json (encoder,
preprocessing, index version, calibration, sha256 of every file). The test split is excluded from the index so
the serving bank matches the evaluated bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from refdb_eval import load, session_key, split_of  # noqa: E402

from filmlook_infer.style_match import PREPROCESSING  # noqa: E402
from filmlook_ml.refdb import manifest  # noqa: E402
from filmlook_ml.visualdna.build import DB as VDNA  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--calibration", required=True); ap.add_argument("--out", default="dist/refdb"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ev = json.loads(Path(a.calibration).read_text())
    rows, X = load("dinov2s_cls_224")
    keep = [i for i, r in enumerate(rows) if r["photographer_id"] in ev["known"] and split_of(session_key(r), a.seed) != "test"]
    with manifest.connect() as c:
        version = c.execute("SELECT version FROM embeddings WHERE model = 'dinov2s_cls_224' LIMIT 1").fetchone()[0]
        meta = {r["work_id"]: dict(r) for r in c.execute("SELECT work_id, photographer_id, title, year, source_url, license, license_evidence_url, credit FROM works")}
    index_version = time.strftime("refdb-%Y%m%d-") + hashlib.sha256("".join(rows[i]["work_id"] for i in keep).encode()).hexdigest()[:8]
    out = Path(a.out) / index_version; out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "checkpoints" / "refdb" / "dinov2_small_cls.onnx", out / "dinov2_small_cls.onnx")
    np.save(out / "vectors_cls.f16.npy", X[keep].astype(np.float16))
    db = sqlite3.connect(out / "works.sqlite")
    db.execute("CREATE TABLE works (row INTEGER PRIMARY KEY, work_id TEXT, photographer_id TEXT, title TEXT, year TEXT, source_url TEXT, license TEXT, license_evidence_url TEXT, credit TEXT)")
    for n, i in enumerate(keep):
        m = meta[rows[i]["work_id"]]
        db.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?)", (n, m["work_id"], m["photographer_id"], m["title"], m["year"], m["source_url"], m["license"], m["license_evidence_url"], m["credit"]))
    db.commit(); db.close()
    shutil.copy(VDNA, out / "visual_dna.sqlite")
    d = ev["methods"]["A_knn"]
    cal = {"knn_k": ev["chosen"]["knn_k"], "temperature_dinov2": d["temperature"],
           "abstain_below_dinov2": d.get("open_set", {}).get("threshold_from_val", 0.0), "source": Path(a.calibration).name}
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"}
    man = {"index_version": index_version, "encoder": f"facebook/dinov2-small@{version}", "preprocessing": PREPROCESSING, "vector": "CLS token, L2-normalised, float16",
           "n_works": len(keep), "photographers": ev["known"], "calibration": cal, "files": files, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "licence_note": "only public-domain and CC0 works; per-row licence and evidence URL in works.sqlite; no images included"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(out, man["n_works"], round(sum((out / f).stat().st_size for f in files) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
