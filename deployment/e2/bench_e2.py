#!/usr/bin/env python
"""Constrained-host benchmark for compact ONNX models (stdlib + numpy + Pillow + onnxruntime only).

  python bench_e2.py --model <dir with model[.int8].onnx + meta.json> --images <dir of JPEGs> [--int8] [--runs 2]

Prints one JSON object with:
- import time and RSS after imports;
- model-load time and RSS after load;
- per-image end-to-end latency (decode + preprocess + inference): first-image (cold) and warm p50/p95/max;
- process peak RSS (ru_maxrss) and, inside a container, the cgroup's memory.peak.

It writes nothing to disk. To measure on the real server, run it there only with the owner's approval, ideally in a
throw-away container with a hard memory limit so the website keeps priority:
  docker run --rm --memory=256m --memory-swap=256m --cpus=0.125 -v $PWD:/b:ro <image> python /b/bench_e2.py ...
"""
import argparse
import json
import os
import resource
import sys
import time

T0 = time.perf_counter()


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def cgroup_peak_mb():
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(p) as fh:
                return int(fh.read().strip()) / 1e6
        except (OSError, ValueError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, os.environ.get("FILMLOOK_SRC", "/src"))
    t = time.perf_counter()
    import numpy as np  # noqa: F401
    import onnxruntime  # noqa: F401
    from PIL import Image  # noqa: F401
    from filmlook_infer.compact import CompactAnalyzer
    out = {"python_start_to_imports_s": round(time.perf_counter() - T0, 3),
           "imports_s": round(time.perf_counter() - t, 3), "rss_after_imports_mb": round(rss_mb(), 1)}
    t = time.perf_counter()
    an = CompactAnalyzer(a.model, int8=a.int8, threads=1)
    out.update({"model_file": an.model_file, "model_mb": round(os.path.getsize(os.path.join(a.model, an.model_file)) / 1e6, 2),
                "load_s": round(time.perf_counter() - t, 3), "rss_after_load_mb": round(rss_mb(), 1)})
    imgs = sorted(os.path.join(a.images, f) for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg")))
    lat, per = [], {}
    for r in range(a.runs):
        for p in imgs:
            t = time.perf_counter()
            res = an.analyze(p)
            dt = time.perf_counter() - t
            lat.append(dt)
            per.setdefault(os.path.basename(p), []).append(round(dt, 3))
            if r == 0 and p == imgs[0]:
                out["first_image_s"] = round(dt, 3)
                out["first_image_timing_ms"] = res["timing_ms"]
    warm = sorted(lat[1:])
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
    out.update({"n_inferences": len(lat), "warm_p50_s": round(q(warm, 0.5), 3), "warm_p95_s": round(q(warm, 0.95), 3),
                "warm_max_s": round(warm[-1], 3), "per_image_s": per,
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "cgroup_peak_mb": cgroup_peak_mb(), "cold_start_total_s": round(
                    out["python_start_to_imports_s"] + out["load_s"] + out["first_image_s"], 3)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
