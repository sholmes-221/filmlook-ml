"""CPU benchmark of ONNX artefacts: 1 thread, latency per image, peak RSS, and test accuracy of each variant.

  python -m research.species.bench --models dist/species/mobilenet_v3_large_last.onnx dist/species/mobilenet_v3_large_last-int8.onnx [--n-acc 800]

Every model runs in a fresh subprocess (so RSS is that model's own) with ORT intra/inter-op threads = 1 and the
default CPU execution provider. Latency = median / p90 of 100 single-image runs after 10 warm-ups (model only),
plus the median of decode + preprocessing + model on real JPEG thumbnails. Accuracy = species top-1 on the first
--n-acc recorder-disjoint test images (same order for every model, so the int8 drop is paired). Writes
data/species/bench.json.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

from research.species import common as C


def child(model: str, n_acc: int) -> dict:
    import numpy as np
    import onnxruntime as ort
    import psutil
    from PIL import Image

    from research.species import data as D

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        sess.run(None, {name: x})
    lat = []
    for _ in range(100):
        t = time.perf_counter(); sess.run(None, {name: x}); lat.append(time.perf_counter() - t)
    rows = D.load_rows()
    labels = json.load(open(model.replace("-int8dyn", "").replace("-int8", "").replace(".onnx", ".labels.json")))
    index = {c["species_id"]: i for i, c in enumerate(labels["classes"])}
    test = [r for r in rows if r["split"] == "test" and r["species_id"] in index][:n_acc]
    tf = D.eval_transform(224)
    hits, e2e, top5 = [], [], []
    for r in test:
        t = time.perf_counter()
        inp = tf(Image.open(D.abs_path(r)).convert("RGB")).unsqueeze(0).numpy()
        lg = sess.run(None, {name: inp})[0][0]
        e2e.append(time.perf_counter() - t)
        y = index[r["species_id"]]
        hits.append(int(lg.argmax() == y)); top5.append(int(y in np.argsort(-lg)[:5]))
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"model": model, "bytes": int(__import__("os").path.getsize(model)), "load_seconds": load_s,
            "latency_ms_median": 1000 * float(np.median(lat)), "latency_ms_p90": 1000 * float(np.percentile(lat, 90)),
            "e2e_ms_median": 1000 * float(np.median(e2e)), "rss_mb_baseline": rss0 / 2**20, "rss_mb_after": proc.memory_info().rss / 2**20,
            "rss_mb_peak": rss_peak / 2**20, "n_acc": len(test), "top1": float(np.mean(hits)), "top5": float(np.mean(top5)),
            "ort_version": ort.__version__, "threads": 1}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2], int(sys.argv[3]))))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-acc", type=int, default=800)
    a = ap.parse_args()
    results = []
    for m in a.models:
        out = subprocess.run([sys.executable, "-m", "research.species.bench", "--child", m, str(a.n_acc)], capture_output=True, text=True,
                             cwd=C.ROOT, env={**__import__("os").environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
        if out.returncode != 0:
            print(m, "FAILED", out.stderr[-2000:])
            results.append({"model": m, "error": out.stderr[-500:]})
            continue
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(rec)
        print(json.dumps(rec))
    prev = json.load(open(C.DATA / "bench.json")) if (C.DATA / "bench.json").exists() else []
    json.dump(prev + results, open(C.DATA / "bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
