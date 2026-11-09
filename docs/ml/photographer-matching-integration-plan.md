# Photographer matching on real photographs: Oracle storage and website integration plan

Written 2026-11-09. Status: plan. Nothing here is deployed; integration waits for the quality gate in
`docs/research/photographer-refdb-report.md` and the resource benchmark below.

## What runs where

| step | where | why |
|---|---|---|
| discovery, licence checks, downloads | workstation | network etiquette, full-size files stay local |
| DINOv2-small reference embeddings, SigLIP features | workstation GPU (RTX 4090) | heavy, one-off, versioned |
| Visual DNA research and knowledge graph build | workstation | reviewed JSON is the source of truth |
| user photo: DINOv2-small embedding | Oracle app container, CPU, ONNX | ~22M parameters; must be measured (below) |
| user photo: semantic tags | **not on the Micro** | SigLIP-base is ~200M parameters, ~800 MB fp32; too large for 952 MB RAM shared with the app |
| similarity search, knowledge scoring, fusion, text | Oracle app container | NumPy dot products over a few thousand vectors; SQLite reads |

The semantic signal therefore has two options, decided by measurement, not assumed: (a) a small int8 image tagger
fitted to the vocabulary if one passes the same evaluation, or (b) the website's existing provider call already
returns scene words, which can be mapped onto the controlled vocabulary and labelled "model-generated" (kept apart
from detected tags). Until one of them is evaluated, the website shows only DINOv2 and knowledge evidence and the
fusion runs without the semantic signal (the result then cannot be labelled `strong`).

## Instance facts (read-only inspection, 2026-11-09 02:13 UTC)

x86_64, 2 vCPU threads (VM.Standard.E2.1.Micro), 952 MB RAM (≈ 556 MB available), 2 GB swap, 41 GB free disk; app
container 115 MB of a 640 MB limit, Caddy 19 MB; 21 local database backups.

## Artifacts shipped to Oracle (compact, versioned, read-only)

```
refdb/<index_version>/
  manifest.json            # versions: encoder (facebook/dinov2-small rev), preprocessing, index, knowledge build, created_at, sha256 of every file
  vectors_cls.f16.npy      # N x 384 float16, L2-normalised (≈ 0.75 KB per work; 4,000 works ≈ 3 MB)
  works.sqlite             # work_id, photographer_id, series_id, title, year, source_url, licence, credit (no images)
  prototypes.f16.npy       # photographer and series prototypes (k-means centres)
  visual_dna.sqlite        # knowledge graph (profiles, motifs, style traits, books, series, sources)
  calibration.json         # per-signal temperatures, fusion weights, abstention thresholds (fitted on validation data)
  model/dinov2_small.onnx  # ≈ 88 MB fp32 / ≈ 23 MB int8 (int8 only if its embeddings match fp32 within the tested tolerance)
```

No reference images are uploaded. Thumbnails, if the UI ever shows a reference work, link to the source page
(Commons / AIC) instead of copying the file. The production database (`results/filmlook.db`) is not touched: the
index lives in its own directory, mounted read-only into the app container, and is backed up with the existing
backup script's directory list (one more path), so a rollback is "point the setting back to the previous
`index_version`".

## Encoder compatibility rule

A query vector is compared only with reference vectors that carry the same `encoder` and `preprocessing` strings in
`manifest.json` (model id + revision + "dinov2_pixels 256→224 bicubic, ImageNet norm"). The API refuses a mismatch
with an error rather than comparing incompatible vectors. Text embeddings are never substituted for image
embeddings.

## Read-only matching API (website side)

`POST /api/style/match` (session required, same quota as analysis) → computes the DINOv2 embedding of the prepared
JPEG in process, then returns:

```json
{"index_version": "...", "encoder": "facebook/dinov2-small@<rev>", "found": {"zh": "乌鸦 · 黑白 · 高反差", "en": "...", "kind": "detected|measured"},
 "matches": [{"photographer_id": "...", "name_zh": "...", "similarity": 0.41, "signals": {"dinov2": 0.52, "dna": 0.31},
              "label": "strong|subject_and_style|subject_only|style_only|no_match", "related_work": {"zh": "《鸦》", "en": "Ravens"},
              "reason": {"zh": "...", "en": "..."}, "visual_characteristics": ["monochrome", "high contrast"]}],
 "limitations": ["similarity is not a probability", "..."]}
```

An internal variant accepts a precomputed 384-d vector plus its `encoder` string (for batch tests); it rejects any
other encoder. Similarity values are shown as relative closeness, never as a percentage chance, unless the
calibration report shows ECE ≤ 0.10 on held-out data for that exact signal.

## Resource benchmark required before integration

1. ONNX export of DINOv2-small; fp32 vs int8 cosine agreement on 200 reference works (int8 only if mean cosine ≥ 0.99).
2. On the instance, inside a throwaway container with the app's limits: cold load time, p50/p95 latency over 30
   prepared JPEGs, peak RSS. Gate: p95 < 3 s, added RSS < 150 MB, no swap storm, existing analysis latency unchanged.
3. Index load: < 50 MB RSS for the vectors and both SQLite files.

## Rollout

Behind `FILMLOOK_STYLE_MATCH=off|shadow|on`. `shadow` computes and logs only the index version and timing (no user
data) next to the existing provider result; `on` shows the 小胶发现 / 摄影师风格匹配 / 关联作品 / 匹配理由 block. Each step
needs tests, a backup and the live verification used for every deploy.
