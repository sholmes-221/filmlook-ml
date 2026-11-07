# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`,
`cat reports/orchestrator/loop_state.json`). Conversation context is not the record. Validated claims with their
evidence are in `docs/FINDINGS.md`; the research model card is `docs/MODEL_CARD.md`; the selection instrument is
`docs/BENCHMARK.md`; pipelines, commands and recovery are in `docs/PIPELINES.md`; data sources in
`docs/REAL_FILM_DATASETS.md` and `docs/CURATED_PHOTOGRAPHY_SOURCES.md`.

Last updated: 2026-11-08 00:00 (session 4, in progress; the 20-hour orchestrator runs until 2026-11-08 17:47:57).

## Current objective
A lightweight, scientifically defensible film-aesthetic representation that transfers to real photographs
(DINOv2-Small research teacher, MobileNetV3 students for deployment): explainable attributes, cross-photo
film-style similarity, badge attributes. Film authenticity is a separate task with the EXIF student as its unchanged
baseline. The standing question is whether the representation captures treatment rather than content, generator
fingerprints, photographer, scanner or dataset shortcuts; benchmark v1 answers it per candidate on synthetic data,
and the evaluation pipeline answers it on real film as soon as an independent real-film set is ready.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 4 has used about 0.2 metered
GPU-minutes; all sessions together 85.2 metered GPU-minutes, 0.46 GB (`docs/RESOURCE_USAGE.md`). Nothing paid.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo style similarity (research) | hybrid (explicit + DINOv2 L4+L8) + family SupCon | benchmark v1 test: B fs 0.44 [0.39, 0.48], rank-1 0.33, gen share 0.59; BlueNeg same-stock p@5 0.25 [0.16, 0.31] (chance 0.12) | `reports/style_benchmark_test.json` |
| abstention for the above | top-1 cosine similarity, answer the 50 % most confident | BlueNeg rank-1 0.22 -> 0.37 at 50 % coverage; AURC 0.59 vs 0.78 | `reports/abstention_test.json` |
| lightweight student, candidate | `student_style_fs_v2_real` (E23b) | val B fs 0.557 vs 0.520; real teacher agreement 0.62 vs 0.53; **not promoted**: awaits a real-film evaluation | `reports/student_style_fs_v2_real.json` |
| lightweight student, shipped research baseline | `student_style_fs_v2` (4.4 MB, 1.2 ms) | best on synthetic but BlueNeg 0.18 = DINOv2 reference; not promoted | `style_benchmark_test.json` |

## What session 4 established (so far)
- The layer-3 teacher's synthetic gain is bound to the filmsynth-trained projection (E28: under a generator-swapped
  projection both teachers collapse to 0.19-0.20 on filmsynth); L3 is *not* a scanner or photographer shortcut
  (scanner probe 0.423 vs 0.457, uploader share 0.014 vs 0.019 on 265 real scans). Teacher target stays L4+L8; L3 is
  only an E27/E27b candidate on real film.
- Pooling two synthetic pipelines in the family SupCon does not give generator invariance (E30: held-out zimage
  0.342 -> 0.351, generator share 0.541 -> 0.635). Rejected.
- Dataset discovery (API-based, en/zh/ja, 187 sources recorded, `research/dataset_discovery.py`): the only sources
  that satisfy both authenticity and licence are CommonCatalog CC-BY film scans with uploader stock tags (class B,
  thousands available), ARTeFACT (class A, 44 frames, 11 emulsions, one photographer), BlueNeg (have) and Commons
  (throttled). Chinese and Japanese platforms yielded nothing reusable. PhotoVogue, DEHAZED and Iconic Artist are
  metadata-only: their terms prohibit automated access and ML use; the openly licensed curated alternative is the
  Art Institute of Chicago's CC0 photographs (`aic_photographs`, pilot 600).
- Orchestrator runtime is 20 h (`--max-hours`, persisted `loop_state.json`, graceful SIGTERM, tick errors never end
  the loop, `--started-at/--resume` keep the original start).

## Live state at the time of writing
- Orchestrator loop PID in `reports/orchestrator/loop_state.json` (started 2026-11-07 21:47:57, deadline 2026-11-08
  17:47:57, 15-minute ticks). Restart after a crash: `python -m research.orchestrator loop --interval 900 --resume`.
- `data_commoncatalog_film_scan` (300 shards, metadata only) -> `data_commoncatalog_film_acquire` (resumable) ->
  `eval_commoncatalog_film_analysis` (automatic when >= 300 rows and complete). First bounded run: 78 rows, 49
  uploaders, 33 stocks.
- `data_artefact_film_acquire`: 37/88 files, resumable, retried by the loop. `data_aic_photographs_acquire`: first
  attempts hit an intermittent HTTP 403 from the AIC API; retried with backoff; `eval_aic_photographs_analysis` (E29)
  follows readiness.
- `data_commons_film_acquire`: 266/5,743; Wikimedia throttles this client (HTTP 429, Retry-After honoured); attempts
  continue with doubling backoff; deprioritised.
- Benchmark v1 test split spent for the session-2 candidate set; BlueNeg blue-intact untouched; no test half of any
  new dataset has been read.

## Known limitations
- Stock labels on the new real-film sets are uploader claims (CommonCatalog, Commons) or one photographer's records
  (ARTeFACT, BlueNeg). No source offers many photographers and owner-verified stocks at once.
- Curated group is historic museum photography (1840s-1920s); no modern editorial aesthetics are licensed for use.
- Students are not promoted; nothing is deployed; the EXIF authenticity baseline is unchanged.
- The research loop runs only entries that ship a script and a scalar falsification rule.

## Rejected approaches (do not repeat without a new hypothesis)
- E3, E2/E4, E9/E9b (superseded by E26), E11, E12, E16, E17, E18, E19, E10 (see earlier sessions).
- E23c SDXL outputs as distillation inputs; E25 ridge residualisation; E26 Gram statistics and late-layer targets;
  BlueNeg blue-corrupted as a selection metric; E28 generator-swap shows the L3 synthetic gain is projection-bound;
  E30 pooled-pipeline SupCon.

## Next recommended actions (in order)
1. Let the acquisitions finish (automatic). When `commoncatalog_film` is ready, read
   `reports/eval_pipeline/history.jsonl`: E27b decides the L3 teacher and `student_style_l3_real` /
   `student_style_fs_v2_real` on the analysis half (rules fixed in `research/backlog.yaml`, `docs/BENCHMARK.md`).
2. When `aic_photographs` is ready, E29 runs (`scripts/eval_curated_photos.py`); if not falsified, register the set
   as a distillation-input domain for an E23-style run; never assign photographer names to clusters.
3. Generator invariance: pooling pipelines failed (E30); the remaining levers are the training signal (treatment
   parameters as targets, E11-style, or augmentation-based invariance) and real-film supervision from uploader
   claims (contrastive on `commoncatalog_film` TRAIN would need a training-allowed snapshot: it is evaluation-only, so
   do not).
4. Commons: if the throttle persists for a day, widen spacing (`--spacing 5`) and cap files per uploader x stock
   before another attempt; never fetch originals, never change identities.
5. Product: deterministic attribute layer plus the shadow-mode record with the abstention threshold; no deployment.
