# Deployment plan for the actual FilmLook host: Oracle `VM.Standard.E2.1.Micro`

Host as confirmed by the owner (2026-11-07):
- `filmlook-e2-micro`: AMD x86-64, **1/8 OCPU burstable, 1 GB RAM**, no GPU, 50 GB boot volume.
- This replaces the earlier A1 (1 OCPU / 6 GB) assumption, which came from the app's deploy guide.

Nothing has been installed on or changed on the host. All numbers below are **simulations** on the development
workstation (i9-14900KF) in Docker with a CPU quota and a hard memory limit with no swap (`reports/e2_simulation.json`).
The E2's AMD EPYC cores are slower per core, so **expect higher real latency**. Run `deployment/e2/bench_e2.py` on the
host (after approval) before deciding.

## Memory budget on the 1 GB host

The website has priority.
- The app's own measurements: it idles at about 150 MB and peaks at about 400 MB while preparing large uploads; its
  guide recommends `mem_limit: 700m` and concurrency 1 on the micro shape.
- Caddy, the OS and Docker take most of the rest; the installer adds swap.
- **Realistic headroom for ML: about 100–200 MB, and not guaranteed during upload peaks.**

## Measured (simulation): one process, 1 thread, ONNX Runtime 1.31.1, 7 real JPEGs + one 48 MP JPEG, 2 passes

| model | file | RSS after imports → after load → peak | cgroup peak | cold start | warm p50 / max, 1/8 CPU | warm p50 / max, 1 CPU (burst) |
|---|---|---|---|---|---|---|
| **MobileNetV3-Small student (fp32)** | 4.3 MB | 78 → 96 → 106 MB | 67–125 MB | 2.7–2.9 s | 0.10 / 0.99 s | 0.01 / 0.11 s |
| DINOv2-small + heads, INT8 | 24.3 MB | 78 → 129 → 139 MB | 99–126 MB | 3.7–7.2 s | 0.49–0.90 / 1.4 s | 0.06 / 0.16 s |
| DINOv2-small + heads, fp32 | 88.5 MB | 78 → 208–222 → 218–232 MB | 185–200 MB | 4.5–5.6 s | 0.81–1.70 / 1.7–3.5 s | 0.11 / 0.22 s |
| existing texture model (`filmlook-infer` 1.0.0, earlier run) | 0.6 MB | — | 112 MB | 5.4 s | 2.5 s | 0.3 s |

What the table shows:
- **Minimum RAM:**
  - the Python + numpy + Pillow + onnxruntime runtime alone is about 78 MB;
  - the student adds about 18 MB, DINOv2 INT8 about 50 MB, DINOv2 fp32 about 140 MB;
  - peak RAM during preprocessing stays bounded, even for 48 MP, because JPEG draft decoding avoids materialising the
    full image.
- **Latency under the 1/8 quota:**
  - the work is bursty: CFS quota periods turn 98 ms of DINOv2 CPU into about 0.8–1.9 s of wall time;
  - the student needs 1.5 ms of CPU, so its time is dominated by JPEG decoding (about 0.9 s for 48 MP) and cold imports
    (about 2.5 s).
- **Quantization:**
  - DINOv2 INT8 is 3.6× smaller, but agrees with fp32 on the top-1 label only 83–96% of the time;
  - dynamic INT8 **broke** the student (top-1 agreement 11–47%), so the student ships as fp32 (4.3 MB);
  - static, calibrated quantization was not tested.
- **Quality:**
  - the student's test macro-F1 on the attribute heads (0.50–0.90) is far above frozen DINOv2 + linear heads
    (0.40–0.60);
  - on capture type it is roughly level (0.35 vs 0.40, overlapping CIs);
  - the attribute labels are measured-statistics proxies, not perception;
  - distillation from DINOv2 added nothing (`reports/student_{sup,kd}.json`).

## Options

| option | extra RAM on E2 | latency (expected) | cost | risk |
|---|---|---|---|---|
| A. Student inside the existing app process (reuses the app's prepared 224 px image; adds onnxruntime + a 4 MB model) | about 20–40 MB (estimated from the deltas above; to be measured in the app) | about 10–50 ms CPU per request (likely under 0.5 s wall) | $0 | lowest; no extra process |
| B. Student as a separate container (capped at 160 MB) | about 100–125 MB | 0.1–1 s warm; about 3 s cold | $0 | memory contention at upload peaks |
| C. DINOv2 on E2 (INT8 or fp32) | 130–230 MB | 0.5–3.5 s warm (simulated; slower on EPYC) | $0 | **not recommended:** squeezes the website on 1 GB |
| D. Remote DINOv2 on the home GPU (RTX 4070) | about 0 | network RTT + about 3 ms GPU; dominated by the home uplink | $0 with a free tunnel (Cloudflare Tunnel or Tailscale) | availability tied to the home machine; photos leave Oracle (privacy notice needed) |
| E. Existing analysis service (current LLM-based pipeline) | 0 | as today | paid per call | unchanged |

## Recommended architecture (zero additional hosting cost)

1. **On E2: option A, only after validation.**
   - Load the MobileNetV3-Small ONNX student lazily inside the app, with one thread and the session reused.
   - It answers only the visual-attribute and broad film-category fields.
   - Hard rule: if the model is not loaded, or the request exceeds its budget, skip the field and keep today's
     behaviour.
2. **Optional remote DINOv2 (option D) for photographer and film matching:**
   - the home RTX 4070 runs the model service behind an outbound-only free tunnel;
   - the app sends a 512 px JPEG with a 2–3 s timeout, in **shadow mode** first (logged, not shown);
   - fallback order: remote model → existing analysis service.
3. **No DINOv2 on the E2 without explicit approval.** No public endpoint, and no change to cards, badges or the
   database: the model returns the app's ids, and the app keeps rendering.

## Before deciding

- Run `deployment/e2/bench_e2.py` on the E2 inside a capped throw-away container: `--memory=256m --memory-swap=256m`,
  at a quiet hour, with your approval.
- Measure the app-process delta for option A in a staging copy of the app, not in production.
- Measure the remote path from Oracle to home: RTT, tunnel overhead, and uptime over a week.
- I could not measure your RTX 4070: this workstation has an RTX 4090. The GPU numbers here (DINOv2 2.5 ms at batch 1)
  are from the 4090.
