#!/usr/bin/env python
"""Scan CommonCatalog CC-BY *metadata only* (no image bytes) and write classified candidates.

  python scripts/discover_commoncatalog.py --shards 60
"""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def scan(path):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from filmlook_ml.datasets.commoncatalog import META_COLS, REPO, classify

    fs = HfFileSystem()
    full = f"datasets/{REPO}/{path}"
    pf = pq.ParquetFile(fs.open(full, block_size=1 << 20))
    out = []
    for g in range(pf.metadata.num_row_groups):
        rows = pf.read_row_group(g, columns=META_COLS).to_pylist()
        for i, r in enumerate(rows):
            c = classify(r)
            if not c["license_ok"] or c["exclude"] or c["medium"] == "unknown":
                continue
            r.pop("exif", None)
            r["datetaken"] = str(r["datetaken"])
            out.append({"file": path, "row_group": g, "row": i, **r, **c})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--range", default="least_dim_range=1024-2048")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/candidates/commoncatalog_candidates.jsonl")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    from filmlook_ml.datasets.commoncatalog import REPO

    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings
                   if s.rfilename.endswith(".parquet") and a.range in s.rfilename)
    random.Random(a.seed).shuffle(files)
    files = files[: a.shards]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n, cnt = 0, Counter()
    with open(a.out, "w") as f, ThreadPoolExecutor(4) as ex:
        for k, rows in enumerate(ex.map(scan, files)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                cnt[r["medium"]] += 1
            n += len(rows)
            print(f"[{k + 1}/{len(files)}] candidates={n} {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
