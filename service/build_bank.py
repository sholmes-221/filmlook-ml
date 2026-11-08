#!/usr/bin/env python
"""Build a similarity bank for the style service from a manifest (ids, 32-d int8 embeddings, optional labels, attribution).

  python service/build_bank.py --manifest data/manifests/<set>_manifest.jsonl --out service/bank.npz [--label-key film_stock_line] [--max 2000]

Only rows whose licence allows the intended use belong in a bank that is shown to users: the script refuses rows whose
`license` is not CC0 / public domain / CC-BY unless --allow-any-license is given, and always records the attribution string
(author + source URL) next to each entry. No pixels are stored, only embeddings and metadata.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "service"))
import filmlook_style_service as S  # noqa: E402

OK_LICENCES = ("CC0", "PDM", "CC-BY-2.0", "CC-BY-4.0", "CC-BY-3.0")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label-key", default="")
    ap.add_argument("--max", type=int, default=5000); ap.add_argument("--allow-any-license", action="store_true"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest) if l.strip()][: a.max]
    if not a.allow_any_license:
        bad = [r for r in rows if not any(str(r.get("license", "")).upper().startswith(k) for k in OK_LICENCES)]
        if bad:
            raise SystemExit(f"{len(bad)} rows have a licence outside {OK_LICENCES} (first: {bad[0].get('license')}); pass --allow-any-license only for an internal bank")
    m = S.Model(S.CFG["model_dir"], 1); m.load(); ids, emb, labels, attr = [], [], [], []
    for r in rows:
        try:
            im = S.read_image(Path(r["path"]).read_bytes()); q, _ = m.embed(im)
        except Exception as e:  # noqa: BLE001
            print("skip", r.get("image_id"), repr(e)[:80]); continue
        ids.append(r["image_id"]); emb.append(q); labels.append(str(r.get(a.label_key) or "") if a.label_key else ""); attr.append(f"{r.get('author') or r.get('uploader') or ''} | {r.get('source_url') or ''} | {r.get('license') or ''}")
    np.savez(a.out, ids=np.array(ids), emb_int8=np.stack(emb), scale=m.scale, labels=np.array(labels), attribution=np.array(attr), model=m.sha)
    print(json.dumps({"rows": len(ids), "out": a.out, "model": m.sha[:12], "labelled": int(sum(1 for l in labels if l))}))


if __name__ == "__main__":
    main()
