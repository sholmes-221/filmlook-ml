"""CPU inference cost of the preset classifiers on one thread: latency, peak RSS, artefact sizes.

    OMP_NUM_THREADS=1 python -m research.lut.bench --models colour_mlp,dinov2_colour_mlp --image <jpeg>

Pipeline timed per image: JPEG decode + resize (PIL draft), colour features (312 values), classifier forward; for
DINOv2 models additionally ImageNet preprocessing and the ONNX session (intra-op threads = 1). 20 repetitions,
median and p90 in milliseconds; peak RSS from resource.getrusage; artefact sizes from disk.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageOps

from research.lut.features import DINO_ONNX, Dino, colour_features_from_array
from research.lut.train import NumpyClassifier, COLOUR_SLICES


def load_small(path, short=384):
    im = Image.open(path)
    im.draft("RGB", (short * 2, short * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = short / min(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BOX)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="reports/lut/bench.json")
    ap.add_argument("--engine", action="store_true", help="also time the standalone engine (analyze + film_likeness) on one thread")
    a = ap.parse_args()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out = {"image": a.image, "image_size": list(Image.open(a.image).size), "rss_mb_at_start": rss0, "models": {}}
    dino = None
    for name in a.models.split(","):
        clf = NumpyClassifier.load(f"data/lut/models/{name}.npz")
        fs = str(clf.d["feature_set"])
        need_dino = "dinov2" in fs
        if need_dino and dino is None:
            dino = Dino(threads=1)
        times = {"decode_resize": [], "colour_features": [], "dinov2": [], "classifier": [], "total": []}
        for _ in range(a.reps):
            t0 = time.perf_counter()
            img = load_small(a.image)
            t1 = time.perf_counter()
            col = colour_features_from_array(img)[None].astype(np.float32)
            t2 = time.perf_counter()
            parts = []
            if need_dino:
                d = dino([Image.fromarray(img)])
            t3 = time.perf_counter()
            for part in fs.split("+"):
                parts.append(d if part == "dinov2" else col[:, COLOUR_SLICES[part]])
            P = clf.predict_proba(np.nan_to_num(np.concatenate(parts, 1)))
            t4 = time.perf_counter()
            for k, v in (("decode_resize", t1 - t0), ("colour_features", t2 - t1), ("dinov2", t3 - t2), ("classifier", t4 - t3), ("total", t4 - t0)):
                times[k].append(1000 * v)
        res = {k: {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90))} for k, v in times.items()}
        res["model_file_mb"] = os.path.getsize(f"data/lut/models/{name}.npz") / 1e6
        res["dinov2_onnx_mb"] = os.path.getsize(DINO_ONNX) / 1e6 if need_dino else 0.0
        res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        res["top3"] = [(clf.classes[i], round(float(P[0, i]), 3)) for i in np.argsort(-P[0])[:3]]
        out["models"][name] = res
        print(f"{name:22} total {res['total']['median_ms']:.1f} ms (p90 {res['total']['p90_ms']:.1f}) | decode {res['decode_resize']['median_ms']:.1f} "
              f"colour {res['colour_features']['median_ms']:.1f} dinov2 {res['dinov2']['median_ms']:.1f} clf {res['classifier']['median_ms']:.2f} | "
              f"peak RSS {res['peak_rss_mb']:.0f} MB | model {res['model_file_mb']:.2f} MB + onnx {res['dinov2_onnx_mb']:.0f} MB | top3 {res['top3']}")
    if a.engine:
        import cv2
        cv2.setNumThreads(1)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        from research.lut.engine import FilmStyleEngine
        t0 = time.perf_counter()
        eng = FilmStyleEngine()
        t_load = 1000 * (time.perf_counter() - t0)
        times = {"analyze": [], "film_likeness": []}
        for _ in range(a.reps):
            t0 = time.perf_counter()
            r = eng.analyze(a.image)
            t1 = time.perf_counter()
            fl = eng.film_likeness(a.image)
            t2 = time.perf_counter()
            times["analyze"].append(1000 * (t1 - t0))
            times["film_likeness"].append(1000 * (t2 - t1))
        res = {k: {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90))} for k, v in times.items()}
        res["load_ms"] = t_load
        res["method"] = eng.method
        res["db_mb"] = sum(os.path.getsize(f) for f in Path(eng.db_dir).glob("*")) / 1e6
        res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        res["rss_mb_before_engine"] = rss_before
        res["top3"] = [(s["id"], s["similarity"]) for s in r["styles"]]
        res["film_likeness"] = {"estimate": fl.get("estimate"), "band": fl.get("band")}
        out["engine"] = res
        print(f"engine ({eng.method}) analyze {res['analyze']['median_ms']:.1f} ms (p90 {res['analyze']['p90_ms']:.1f}) | film_likeness {res['film_likeness']['median_ms']:.1f} ms | "
              f"load {t_load:.0f} ms | db {res['db_mb']:.1f} MB | peak RSS {res['peak_rss_mb']:.0f} MB (before engine {rss_before:.0f})")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
