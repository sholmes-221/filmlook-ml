"""Open Images V7 metadata: stream the large train CSVs and keep only the rows this study needs (nothing large on disk).

  python -m research.concepts.oi_meta labels     -> data/concepts/oi_meta/train-labels-concepts.csv
  python -m research.concepts.oi_meta images     -> data/concepts/oi_meta/train-images-needed.csv

Labels: human-verified image-level labels (CC BY 4.0, Google). Images: Flickr photos the Open Images team lists as
CC BY 2.0, with author, original URL and licence per image; fetched later from the CVDF-hosted copy.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd
import requests

from research.concepts.concept_map import CONCEPTS
from research.concepts.common import META, USER_AGENT, related_mids

BASE = "https://storage.googleapis.com/openimages/"


def stream_csv(url: str):
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20):
            yield line.decode("utf-8")


def wanted_mids() -> set[str]:
    cd = pd.read_csv(META / "oidv7-class-descriptions.csv")
    n2m = dict(zip(cd.DisplayName, cd.LabelName))
    mids = {n2m[n] for _, names in CONCEPTS.values() for n in names if n in n2m}
    return mids | related_mids()


def labels() -> None:
    mids = wanted_mids()
    out = META / "train-labels-concepts.csv"
    n = kept = 0
    with open(out, "w") as fh:
        it = stream_csv(BASE + "v7/oidv7-train-annotations-human-imagelabels.csv")
        fh.write(next(it) + "\n")
        for line in it:
            n += 1
            parts = line.split(",")
            if len(parts) >= 4 and parts[2] in mids:
                fh.write(line + "\n"); kept += 1
            if n % 10_000_000 == 0:
                print(f"{n:,} rows, kept {kept:,}", flush=True)
    print(f"done: {n:,} rows, kept {kept:,} -> {out}")


def images() -> None:
    need = set(pd.read_csv(META / "train-ids-needed.txt", header=None)[0])
    out = META / "train-images-needed.csv"
    n = kept = 0
    with open(out, "w") as fh:
        it = stream_csv(BASE + "v6/oidv6-train-images-with-labels-with-rotation.csv")
        fh.write(next(it) + "\n")
        for line in it:
            n += 1
            if line[:16] in need:
                fh.write(line + "\n"); kept += 1
    print(f"done: {n:,} rows, kept {kept:,} -> {out}")


if __name__ == "__main__":
    {"labels": labels, "images": images}[sys.argv[1]]()
