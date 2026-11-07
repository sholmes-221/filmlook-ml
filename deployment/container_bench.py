"""Run inside the deployment container: cold start + warm latency + cgroup peak memory.

docker run --rm --memory=256m --memory-swap=256m --cpus=0.25 -v <imgs>:/imgs:ro \
  -v $PWD/deployment/container_bench.py:/app/cb.py:ro --entrypoint python filmlook-infer:1.0.0 /app/cb.py /imgs
"""
import glob
import json
import sys
import time

t0 = time.perf_counter()
from filmlook_infer import FilmLookAnalyzer  # noqa: E402

an = FilmLookAnalyzer("/app/model", threads=1)
imgs = sorted(glob.glob(sys.argv[1] + "/*.jpg"))
an.analyze(imgs[0])
cold = time.perf_counter() - t0
lat = []
for _ in range(2):
    for p in imgs:
        s = time.perf_counter()
        an.analyze(p, "zh")
        lat.append(time.perf_counter() - s)
lat.sort()


def rd(p):
    try:
        return int(open(p).read().split()[0])
    except Exception:
        return None


print(json.dumps({"n": len(lat), "cold_start_s": round(cold, 2), "warm_p50_s": round(lat[len(lat) // 2], 3),
                  "warm_p95_s": round(lat[int(len(lat) * 0.95)], 3), "warm_max_s": round(lat[-1], 3),
                  "cgroup_peak_mb": round((rd("/sys/fs/cgroup/memory.peak") or 0) / 2**20, 1)}))
