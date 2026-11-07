#!/usr/bin/env python
"""Download a SMALL, license-checked pilot. Large sources require --approve-large."""
import argparse
import json

from filmlook_ml import paths as P
from filmlook_ml.datasets.acquire import download_pilot, ingest_user_consented, pd12m_candidates
from filmlook_ml.datasets.preprocess import write_jsonl
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.datasets.synthetic import write_synthetic_bases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "pd12m", "user", "unsplash_lite"], required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between image requests")
    ap.add_argument("--approve-large", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    reg = Registry.load()
    if a.limit > 2000 and not a.approve_large:
        raise SystemExit("refusing >2000 images without --approve-large")

    if a.source == "synthetic":
        recs = write_synthetic_bases(P.RAW / "synthetic", a.limit, a.seed)
    elif a.source == "pd12m":
        cands = pd12m_candidates(a.shard, a.limit)
        print(f"{len(cands)} photographic CC0/PDM candidates in shard {a.shard}")
        recs = download_pilot(cands, P.RAW / "pd12m", reg, a.limit, delay_s=a.delay)
    elif a.source == "user":
        recs = ingest_user_consented(P.RAW / "user_consented")
    else:
        d = reg.datasets["unsplash_lite"]
        if d.get("requires_approval_to_download") and not a.approve_large:
            raise SystemExit("Unsplash Lite is ~700MB and its terms restrict use to INTERNAL models with no "
                             "redistribution. Re-run with --approve-large after confirming that fits FilmLook.")
        raise SystemExit("Unsplash Lite: download https://unsplash.com/data/lite/latest manually into "
                         "data/raw/unsplash_lite/ (requires accepting their terms in a browser).")
    out = P.MANIFESTS / f"raw_{a.source}.jsonl"
    write_jsonl(out, recs)
    print(f"{len(recs)} raw records -> {out}")


if __name__ == "__main__":
    main()
