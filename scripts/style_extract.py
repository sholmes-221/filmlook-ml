#!/usr/bin/env python
"""Compute style features (v1, JS-reproducible) for every manifest image (+ optional extra records).
Cached by sha256 in data/embeddings/style_features.json."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl


def _one(path):
    from filmlook_ml.style.features import features_from_path
    try:
        return features_from_path(path)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def extract(rows, workers=16, cache_path=P.EMBEDDINGS / "style_features.json"):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [r for r in rows if r["sha256"] not in cache]
    if todo:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            for r, f in zip(todo, ex.map(_one, [r["path"] for r in todo], chunksize=8)):
                cache[r["sha256"]] = f
        cache_path.write_text(json.dumps(cache))
    return {r["image_id"]: cache[r["sha256"]] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl manifests (e.g. style eval set)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    for e in a.extra:
        rows += read_jsonl(e)
    f = extract(rows, a.workers)
    print(len(f), "images;", sum("error" in v for v in f.values()), "errors")


if __name__ == "__main__":
    main()
