#!/usr/bin/env python
"""Metadata-only scan of CommonCatalog CC-BY for photos whose OWN uploader tagged them with a
photographer's name (e.g. "saulleiter", "fredherzog"): a weak, teacher-independent signal that the uploader
considers their photo to resemble that style. Used for EVALUATION only.

Rows that look like reproductions of the photographer's own work (exhibitions, prints, books, posters)
are excluded: those images would be the photographer's copyrighted work, not the uploader's.

  python scripts/scan_style_tags.py --shards 600
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TAGS = {
    "saul_leiter": r"saulleiter|saul_leiter|leiteresque",
    "william_eggleston": r"williameggleston|eggleston",
    "daido_moriyama": r"daidomoriyama|moriyama|arebureboke",
    "alex_webb": r"alexwebb",
    "fan_ho": r"fanho",
    "stephen_shore": r"stephenshore",
    "fred_herzog": r"fredherzog",
    "ernst_haas": r"ernsthaas",
    "rinko_kawauchi": r"rinkokawauchi|kawauchi",
    "vivian_maier": r"vivianmaier",
    "martin_parr": r"martinparr",
}
REPRO = re.compile(r"exhibit|gallery|museum|retrospective|book|poster|print(s|ed)?\b|framed|frame|wall|"
                   r"signage|sign\b|magazine|screen|catalog|postcard", re.I)
COLS = ["photoid", "uid", "unickname", "usertags", "licenseurl", "pageurl", "blip2_caption", "title",
        "width", "height", "capturedevice", "status"]


def scan(path):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from filmlook_ml.datasets.commoncatalog import LICENSE_RE, REPO

    fs = HfFileSystem()
    out = []
    try:
        with fs.open(f"datasets/{REPO}/{path}", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            for g in range(pf.metadata.num_row_groups):
                tags = pf.read_row_group(g, columns=["usertags"]).column(0).to_pylist()
                hits = [i for i, t in enumerate(tags) if t and any(re.search(rx, t.lower()) for rx in TAGS.values())]
                if not hits:
                    continue
                rows = pf.read_row_group(g, columns=COLS).to_pylist()
                for i in hits:
                    r = rows[i]
                    if r["status"] != "success" or not LICENSE_RE.search((r["licenseurl"] or "").strip()):
                        continue
                    t = r["usertags"].lower()
                    styles = [k for k, rx in TAGS.items() if re.search(rx, t)]
                    text = f"{r['blip2_caption']} {r['title']} {t}"
                    out.append({"file": path, "row_group": g, "row": i, **r, "styles": styles,
                                "repro_suspect": bool(REPRO.search(text))})
    except Exception as e:  # keep scanning other shards
        print("ERR", path, e, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/candidates/style_tag_candidates.jsonl")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    from filmlook_ml.datasets.commoncatalog import REPO
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".parquet")
                   and ("1024-2048" in s.rfilename or "768-1024" in s.rfilename or "2048-4096" in s.rfilename))
    random.Random(a.seed).shuffle(files)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(8) as ex:
        for k, rows in enumerate(ex.map(scan, files[: a.shards])):
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
            n += len(rows)
            if k % 20 == 0:
                print(f"[{k + 1}/{a.shards}] hits={n}", flush=True)
    print("done", n)


if __name__ == "__main__":
    main()
