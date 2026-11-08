#!/usr/bin/env python
"""Measure the style service as it would run on a small host: cold start, latency, peak RSS, bounded concurrency.

  python service/bench_service.py --cpus 1 --image <jpg> [--port 8766] [--clients 6] [--requests 40]   -> reports/service_bench.json

Starts the service pinned to the given CPUs (taskset), waits for /health, times /ready (model load), then sequential
/embed requests (with and without the 28 statistics), then `--clients` concurrent clients for `--requests` requests each,
counting HTTP 503 (bounded concurrency) and latencies; RSS and peak RSS (VmHWM) are read from /proc/<pid>/status.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def post(url, body, timeout=60):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "image/jpeg"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def rss_kb(pid):
    peak = cur = None
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmHWM"):
            peak = int(line.split()[1])
        if line.startswith("VmRSS"):
            cur = int(line.split()[1])
    return cur, peak


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--cpus", type=int, default=1); ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--clients", type=int, default=6); ap.add_argument("--requests", type=int, default=40); ap.add_argument("--max-inflight", type=int, default=2); ap.add_argument("--out", default="service_bench.json"); a = ap.parse_args()
    body = Path(a.image).read_bytes(); base = f"http://127.0.0.1:{a.port}"
    env = {**os.environ, "FLS_PORT": str(a.port), "FLS_MAX_INFLIGHT": str(a.max_inflight), "FLS_THREADS": "1", "OMP_NUM_THREADS": "1"}
    cmd = ["taskset", "-c", ",".join(str(i) for i in range(a.cpus)), sys.executable, str(ROOT / "service" / "filmlook_style_service.py")]
    t0 = time.perf_counter(); proc = subprocess.Popen(cmd, env=env, stderr=subprocess.DEVNULL)
    while True:
        try:
            get(base + "/health", 2); break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    t_health = time.perf_counter() - t0; rss_idle = rss_kb(proc.pid)
    t = time.perf_counter(); _, ready = get(base + "/ready"); t_ready = time.perf_counter() - t; rss_loaded = rss_kb(proc.pid)
    lat = {"embed_full_ms": [], "embed_model_only_ms": []}
    for key, url in (("embed_full_ms", "/embed"), ("embed_model_only_ms", "/embed?explicit=0")):
        for _ in range(15):
            t = time.perf_counter(); code, out = post(base + url, body); lat[key].append(1000 * (time.perf_counter() - t)); assert code == 200, code
    results = {"ok": 0, "busy_503": 0, "other": 0, "lat_ms": []}; lock = threading.Lock()

    def client():
        for _ in range(a.requests):
            t = time.perf_counter(); code, _ = post(base + "/embed?explicit=0", body); dt = 1000 * (time.perf_counter() - t)
            with lock:
                if code == 200:
                    results["ok"] += 1; results["lat_ms"].append(dt)
                elif code == 503:
                    results["busy_503"] += 1
                else:
                    results["other"] += 1
    t = time.perf_counter(); th = [threading.Thread(target=client) for _ in range(a.clients)]; [x.start() for x in th]; [x.join() for x in th]; wall = time.perf_counter() - t
    rss_after = rss_kb(proc.pid); _, h1 = get(base + "/health"); proc.terminate(); proc.wait(5)
    rep = {"cpus": a.cpus, "max_inflight": a.max_inflight, "cold_start_to_health_s": round(t_health, 3), "model_load_to_ready_s": round(t_ready, 3), "service_load_ms": ready.get("load_ms"),
           "rss_mb": {"idle_no_model": round(rss_idle[0] / 1024, 1), "after_model_load": round(rss_loaded[0] / 1024, 1), "after_concurrency": round(rss_after[0] / 1024, 1), "peak_vmhwm": round(rss_after[1] / 1024, 1)},
           "sequential_latency_ms": {k: {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1), "n": len(v)} for k, v in lat.items()},
           "concurrency": {"clients": a.clients, "requests_per_client": a.requests, "ok": results["ok"], "busy_503": results["busy_503"], "other": results["other"], "wall_s": round(wall, 2),
                           "throughput_ok_per_s": round(results["ok"] / wall, 1), "lat_ok_ms_p50": round(float(np.percentile(results["lat_ms"], 50)), 1) if results["lat_ms"] else None,
                           "lat_ok_ms_p95": round(float(np.percentile(results["lat_ms"], 95)), 1) if results["lat_ms"] else None},
           "health_after": h1, "image_bytes": len(body), "python": sys.version.split()[0]}
    out = ROOT / "reports" / a.out; out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
