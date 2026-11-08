# Architecture of the Gemini-free analysis (statistics-first MVP, 2026-11-08)

```
upload (website: validation, 2048 px copy, EXIF stripped)
  └─ filmlook_infer.retrieval_mvp.RetrievalMVP.analyze(bytes, language, allowed_looks)
       ├─ load_image (bounded: 60 MP guard, JPEG draft decode, long side 1280)
       ├─ 28 measured statistics  → style tags (monochrome, warm/cool/neutral, muted/vivid, soft/crisp, key, contrast, grain)
       ├─ v1 texture model (checkpoints/best, 0.6 MB ONNX, features in)  → six rendering attributes → strengths / weaknesses templates
       ├─ supervised student (checkpoints/student_sup, 4.3 MB ONNX fp32, 224 px)  → P(film scan) + temperature/saturation/contrast/key/sharpness
       ├─ Platt map (checkpoints/retrieval_mvp/calibration.json)  → film-likeness estimate 0..1, banded wording
       ├─ rules over measured character within the caller's look allowlist  → one look suggestion or abstention
       └─ deterministic zh / en text  → schema filmlook_output_v2.json
website: photographer matching = existing measured 17-axis ranking under the library rules (unchanged)
```

## What each component contributes, and the evidence

| component | output | evidence (independent data) | where |
|---|---|---|---|
| 28 statistics | style tags, grain visibility, basis for rules | strongest representation for rendering-family retrieval on fresh museum data (0.85 balanced accuracy within a collection), ties the student within one photographer's stocks (AUC 0.69) | E53, E57 |
| `student_sup` capture head | film-likeness estimate | AUROC 0.86 [0.79, 0.92] film scans vs digital captures on the E58 READ half, ECE 0.07 after Platt; JPEG q50 and 0.5x rescale flip the 50 % decision in about 5 % of cases | E58 |
| `student_sup` attribute heads | temperature, saturation, contrast, key, sharpness | 0.81 to 0.89 accuracy against known edits on held-out originals (grain 0.59, so grain is taken from the statistics) | reports/student_sup.json |
| v1 texture model | six rendering attributes, strengths/weaknesses | within-photo edit ordering agreement 0.99; **its scalar score reads at chance (0.49) against real film and is not returned** | REPORT.md, E58 |
| rules | look suggestion within the library | deterministic; no accuracy claim; abstains by design | code |
| frozen style student (PCA-32 int8) | optional similar-look retrieval with the E55 gate | robust to re-encoding; no licensed reference bank exists yet, so the feature is not wired into the MVP | E53, E55, service/ |

Not included, with the reason: rendering-family prototypes (fit on one museum collection, read on another:
0.23 balanced accuracy against 0.20 chance; digitisation pipelines dominate), stock classification (0.12 vs
0.043), photographer identification (no legal reference images), any aesthetic score (no label), any text model.

## Versioning

Every result carries `versions`: schema 2.0, the v1 model version with the ONNX file hash, the student version
with its hash, and the artifact version of the calibration and prototype files. The website stores the result
JSON with the record, so historical results are never rewritten; a new artifact version means new analyses
only.

## Serving footprint (development host, one pinned core, 2048 px uploads)

Load 0.16 s; per photo p50 259 ms, p95 303 ms; peak RSS 165 MB for the whole Python process (ORT + NumPy +
Pillow + both models). The E2 Micro core is several times slower: plan for roughly 1 to 1.5 s per photo and the
same memory. The website's work gate (one image job at a time on the Micro) bounds concurrency; the model is
loaded once per worker; no temporary files; nothing is fetched.

## Interfaces

- Python: `RetrievalMVP(v1_dir, sup_dir, artifacts_dir).analyze(path_or_buffer, language="zh"|"en", allowed_looks=None|set)`.
- Errors: `InvalidImage`, `UnsupportedLanguage` (typed, stable codes); a missing artifact fails at construction,
  never at request time, and never falls back to a remote model.
- Schema: `src/filmlook_infer/schema/filmlook_output_v2.json`; tests in `tests/test_retrieval_mvp.py`.
