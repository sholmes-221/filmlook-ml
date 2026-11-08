# FilmLook photographic-style similarity service (experimental MVP)

A standalone CPU inference service for "similar look" retrieval. It runs beside the existing FilmLook application and
does not replace the film-look scoring or the LLM-based photography analysis. It never identifies a film stock or a
photographer: a bank label is surfaced only as "looks like", and only when the abstention rule passes.

## What it serves

| endpoint | purpose |
|---|---|
| `GET /health` | liveness and counters; never loads the model |
| `GET /ready` | loads the model lazily (first call) and answers 200 after a warm-up input |
| `POST /embed[?explicit=0]` | image bytes -> 32-d int8 style embedding (+ per-component scale) and the 28 deterministic statistics (`explicit=0` skips them; they cost about 100 ms on one core) |
| `POST /similar?k=5[&explicit=0]` | the same, plus the k nearest bank entries by cosine, each with its attribution; `decision` is `similar_look_only`, `label_claim` (at least 3 of the top 5 bank labels agree and the top-1 cosine is at least `FLS_TAU`) or `abstain` |

Model: `checkpoints/frozen_session7/style_fs_v2_real_f576_deg` (MobileNetV3-Small student, 4.4 MB ONNX, SHA-256 in
`checkpoints/frozen_session7/freeze.json`) with the PCA-32 int8 reduction from `pca.npz`. The embedding is 32 bytes per image.
Preprocessing matches the evaluation pipeline (short side 384 px, 224 px centre crop, ImageNet normalisation).

## Resource design (Oracle Always Free E2 Micro, about 1 GB RAM)

- one ORT intra-op thread (`FLS_THREADS=1`), arena and memory pattern off; standard-library `ThreadingHTTPServer`, no web framework;
- bounded in-flight requests (`FLS_MAX_INFLIGHT`, default 2): the next request gets HTTP 503 with `Retry-After: 1`;
- request body limit (`FLS_MAX_BYTES`, default 8 MB), pixel limit (`FLS_MAX_PIXELS`, default 30 MP), JPEG draft decoding;
- lazy model load (`/ready` or first request; `FLS_PRELOAD=1` loads at start), graceful fallback to the 28 statistics
  (`"fallback": "explicit28"`) when the network fails, JSON errors (400 invalid image, 413 too large, 503 busy, 500 internal).

Measured on the development host pinned to one CPU core (`python service/bench_service.py --cpus 1 ...`,
`reports/service_bench.json`): RSS 65 MB before the model, 104 MB after, peak 135 MB under six concurrent clients;
`/embed?explicit=0` 21 ms p50, `/embed` with the statistics 113 ms p50; with `FLS_MAX_INFLIGHT=2` and six clients, two of
every six requests were served (47 req/s) and the rest received 503 immediately. The E2 Micro core is several times slower
than this host's; plan for roughly 100-250 ms per embedding and 0.5-1.2 s with the statistics, and for a 400 MB
`MemoryMax` (the unit file sets 400M/300M high).

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r service/requirements.txt      # onnxruntime, numpy, pillow
FLS_MODEL_DIR=checkpoints/frozen_session7/style_fs_v2_real_f576_deg FLS_PORT=8765 FLS_BANK=bank.npz python service/filmlook_style_service.py
curl -s localhost:8765/ready
curl -s --data-binary @photo.jpg -H "Content-Type: image/jpeg" "localhost:8765/similar?k=5"
```
The service needs `src/filmlook_infer` (the runtime package, pure CPU) on the path next to `service/`; a systemd unit is in
`service/filmlook-style.service`. Put it behind the existing reverse proxy on `127.0.0.1`; do not expose it directly.

## Bank

`python service/build_bank.py --manifest <manifest.jsonl> --out bank.npz [--label-key film_stock_line]` embeds the manifest's
images (no pixels are stored) and refuses licences outside CC0 / public domain / CC-BY unless `--allow-any-license` (internal
banks only). Each entry carries `author | source URL | licence` for display. A bank built from the 601 CC-BY train originals
exists for testing; the product bank should be the application's own consented uploads.

## Limits (read before integrating)

- "Similar look" is exploratory: on fresh real photographs the embedding separates physical print processes above chance
  and keeps its nearest neighbour under crops, flips, JPEG and rescaling, but same-film-stock retrieval across photographers
  is near chance for every representation tested (`docs/MVP_DECISION.md`). Never present a neighbour's stock or photographer
  as an identification.
- Mounted or bordered scans (museum-style digitisations) shift the embedding when cropped; user photographs do not.
- No human evaluation of "look" exists; the evidence is retrieval against catalogue metadata.
