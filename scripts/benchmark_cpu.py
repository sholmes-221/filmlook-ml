#!/usr/bin/env python
"""End-to-end CPU benchmark (decode + preprocessing + features + ONNX + JSON), per model artifact.

Each configuration runs in a FRESH subprocess importing only ``filmlook_infer`` (torch/filmlook_ml must not
load), so cold start and peak RSS (ru_maxrss of the whole process) are measured honestly. Inputs are the
ORIGINAL camera JPEGs (not the 1280 px training copies).

  python scripts/benchmark_cpu.py --models checkpoints/texture_ms1280 checkpoints/fused_224 --threads 1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import json, resource, sys, time
t0 = time.perf_counter()
from filmlook_infer import FilmLookAnalyzer
t_import = time.perf_counter() - t0
args = json.loads(sys.argv[1])
assert "torch" not in sys.modules and "filmlook_ml" not in sys.modules, "training code leaked into runtime"
t1 = time.perf_counter()
an = FilmLookAnalyzer(args["model"], threads=args["threads"], prefer_int8=args["int8"])
t_load = time.perf_counter() - t1
imgs = args["images"]
t2 = time.perf_counter(); an.analyze(imgs[0], "en"); t_first = time.perf_counter() - t2
lat, raw = [], []
for p in imgs:
    s = time.perf_counter(); an.analyze(p, "zh"); lat.append(time.perf_counter() - s)
for p in imgs[: args["n_quality"]]:
    r = an.predict_raw(p); raw.append({"score": r["score"], "attrs": r["attrs"].tolist()})
from filmlook_infer.imageio import load_image
from filmlook_infer.features import compute_features, feature_vector
from filmlook_infer.pixels import dinov2_pixels
s = time.perf_counter(); im = load_image(imgs[0], max_side=an.max_side); t_decode = time.perf_counter() - s
s = time.perf_counter(); fv = feature_vector(compute_features(im, an.crop))[None]; t_feat = time.perf_counter() - s
px = dinov2_pixels(im, an.input_size)[None]
s = time.perf_counter()
for _ in range(5): an._run(px, fv)
t_model = (time.perf_counter() - s) / 5
lat.sort()
print(json.dumps({"onnx_file": an.onnx_file, "threads": args["threads"], "import_s": t_import, "load_s": t_load,
  "cold_first_request_s": t_first, "cold_start_total_s": t_import + t_load + t_first,
  "warm_p50_s": lat[len(lat)//2], "warm_p95_s": lat[min(len(lat)-1, int(len(lat)*0.95))], "warm_max_s": lat[-1],
  "decode_s": t_decode, "features_s": t_feat, "model_only_s": t_model,
  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "raw": raw}))
"""


def run_child(model, images, threads, int8, n_quality, mem_limit_mb=None):
    cmd = [sys.executable, "-c", CHILD, json.dumps({"model": model, "images": images, "threads": threads,
                                                    "int8": int8, "n_quality": n_quality})]
    pre = None
    if mem_limit_mb:
        import resource

        def pre():
            lim = mem_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    p = subprocess.run(cmd, capture_output=True, text=True, preexec_fn=pre, env=env)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1] if p.stderr else f"exit {p.returncode}"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def quality_delta(a, b):
    import numpy as np
    return {"film_score_mae_points": float(np.mean([abs(x["score"] - y["score"]) * 100 for x, y in zip(a, b)])),
            "attr_mae": float(np.mean([np.abs(np.array(x["attrs"]) - np.array(y["attrs"])).mean()
                                       for x, y in zip(a, b)]))}


def bench_images(n: int) -> list[str]:
    """Original camera JPEGs from the TEST split (largest first = worst case), plus PD12M scans."""
    from filmlook_ml.datasets.preprocess import read_jsonl
    man = [r for r in read_jsonl(Path("data/manifests/manifest.jsonl"))
           if r.get("split") == "test" and r["edit_type"] == "original" and r["dataset_id"] == "commoncatalog_ccby"]
    raw = {r["source_id"]: r["path"] for r in read_jsonl(Path("data/manifests/raw_commoncatalog.jsonl"))}
    paths = [raw[r["source_id"]] for r in man if r.get("source_id") in raw]
    paths.sort(key=lambda p: -os.path.getsize(p))
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--rlimit-mb", type=int, nargs="*", default=[512],
                    help="virtual-memory caps (RLIMIT_AS; stricter than RSS)")
    ap.add_argument("--out", default="reports/cpu_benchmark.json")
    a = ap.parse_args()
    imgs = bench_images(a.images)
    from PIL import Image
    sizes = [Image.open(p).size for p in imgs]
    res = {"n_images": len(imgs), "image_megapixels": [round(w * h / 1e6, 1) for w, h in sizes],
           "host_cpu": os.cpu_count(), "runs": [], "note": "dev workstation, not deployment hardware"}
    for model in a.models:
        variants = [False] + ([True] if (Path(model) / "model.int8.onnx").exists() else [])
        raws = {}
        for int8 in variants:
            for t in a.threads:
                r = run_child(model, imgs, t, int8, n_quality=len(imgs))
                raws.setdefault(int8, r.get("raw"))
                row = {"model": model, "int8": int8, **{k: v for k, v in r.items() if k != "raw"}}
                row["model_mb"] = (Path(model) / r["onnx_file"]).stat().st_size / 1e6 if "onnx_file" in r else None
                res["runs"].append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if raws.get(False) and raws.get(True):
            res.setdefault("int8_vs_fp32", {})[model] = quality_delta(raws[False], raws[True])
            print("int8 vs fp32", model, res["int8_vs_fp32"][model])
        for lim in a.rlimit_mb:
            r = run_child(model, imgs[:3], 1, variants[-1], 0, mem_limit_mb=lim)
            res.setdefault("memory_cap_tests", []).append({"model": model, "rlimit_as_mb": lim, "ok": "error" not in r,
                                                           **{k: v for k, v in r.items()
                                                              if k in ("peak_rss_mb", "warm_p50_s", "error")}})
            print("RLIMIT_AS", lim, model, res["memory_cap_tests"][-1])
    res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
