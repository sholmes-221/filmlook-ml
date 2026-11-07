# CPU deployment: film-authenticity model v1 (server-side, Xiaohongshu backend)

Nothing in this repo deploys to, or modifies, the FilmLook application. The deliverable is the package from
`python scripts/build_package.py --verify` → `dist/filmlook-infer-1.0.0/` (see `INTEGRATION.md`, copied into the
package as its README). The photographer-style module is separate: pure JavaScript, running inside the Mini Tool
(`js/filmlook-style/`).

## What ships

| item | size |
|---|---|
| `model/model.onnx`: Model A, a 3-seed ensemble MLP over 28 measured statistics (fp32; INT8 is pointless at this size) | **0.62 MB** |
| `filmlook_infer/` runtime code + JSON schema + `meta.json` | ~0.4 MB |
| package total (unpacked / tar.gz) | 1.1 MB / 0.8 MB |
| runtime deps: numpy, pillow, scipy, onnxruntime (fresh venv site-packages) | 258 MB on disk |
| Docker image `filmlook-infer:1.0.0` (python:3.12-slim) | 594 MB |

## Measured (2026-11-06, development workstation: i9-14900KF; NOT deployment hardware)

End-to-end = decode original camera JPEG (4-6 MP) + features + ONNX + JSON, fresh process per configuration,
1 thread (`scripts/benchmark_cpu.py`, `reports/cpu_benchmark.json`):

| model | artifact | peak RSS | cold start | warm p50 / p95 | features | network |
|---|---|---|---|---|---|---|
| **A texture (shipped)** | 0.62 MB | **160 MB** | **0.48 s** | **0.30 / 0.37 s** | 0.22 s | <1 ms |
| C fused DINOv2 INT8 (comparator) | 26.9 MB | 205 MB | 0.60 s | 0.37 / 0.39 s | 0.24 s | 45 ms |
| C fused DINOv2 fp32 | 92.6 MB | 295 MB | 0.68 s | 0.40 / 0.44 s | 0.23 s | 93 ms |

Both run under `RLIMIT_AS` = 512 MB. INT8 vs fp32 for C: 0.30 film-score points MAE, attribute MAE 0.011.

Docker with hard cgroup limits (`--memory` = `--memory-swap`), Model A package, 8 photos × 2 passes. These are
**simulations** of a small VM:

| limits | cgroup peak | cold start | warm p50 | warm p95 / max |
|---|---|---|---|---|
| 1 GB, 1 CPU | 114 MB | 0.69 s | 0.31 s | 0.33 s |
| 512 MB, 1 CPU | 109 MB | 0.65 s | 0.30 s | 0.31 s |
| 256 MB, 0.25 CPU | 109 MB | 5.0 s | 1.30 s | 2.48 s |
| 256 MB, 0.125 CPU | 112 MB | 5.4 s | 2.50 s | 2.70 s |

Reproduce: `docker build -f deployment/Dockerfile --build-arg PKG=dist/filmlook-infer-1.0.0 -t filmlook-infer:1.0.0 .`
then `docker run --rm --memory=256m --memory-swap=256m --cpus=0.25 -v <imgs>:/imgs:ro -v $PWD/deployment/container_bench.py:/app/cb.py:ro --entrypoint python filmlook-infer:1.0.0 /app/cb.py /imgs`.

## Against the targets

| target | status |
|---|---|
| artifact < 30 MB | met: 0.62 MB |
| peak memory < 256 MB | met: 160 MB process RSS, ~110 MB cgroup |
| warm < 2 s on a constrained backend | met at 0.25 CPU (1.3 s p50, 2.5 s worst); at 0.125 CPU the p50 is 2.5 s, so not met |
| cold < 5 s | borderline: 5.0-5.4 s at 0.125-0.25 CPU, mostly Python import of numpy/scipy/onnxruntime. Keep one warm worker |
| no GPU, no external API | met |

**Actual host (confirmed 2026-11-07): Oracle `VM.Standard.E2.1.Micro`, 1/8 OCPU burstable, 1 GB RAM, x86-64.** It is
shared with the website, which has priority. See `E2_MICRO_PLAN.md` for the memory budget and the measured compact
models. Real deployment hardware was **not** measured. Its burstable fractional vCPU is expected
to be between the 0.125 and 1 CPU rows. Run `deployment/container_bench.py` on the target instance before
launch.

## Operating notes

- Load `FilmLookAnalyzer` once per worker process. Serialise requests on 1-vCPU instances (one worker, a queue).
- About 75% of request time is NumPy/SciPy feature extraction on the ≤1280 px image. The ONNX network is negligible. To go faster, recompute the features at lower resolution, but the comparison showed resolution costs quality (1280 > 1024 > 768 on every metric), so this needs retraining and re-evaluation.
- Inputs are bounded: JPEG DCT draft decoding, downscale to `max_side` = 1280, reject > 60 MP.
