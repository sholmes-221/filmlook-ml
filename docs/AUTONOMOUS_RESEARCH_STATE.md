# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`,
`cat reports/orchestrator/loop_state.json`). Conversation context is not the record. Validated claims with their
evidence are in `docs/FINDINGS.md`; the research model card is `docs/MODEL_CARD.md`; the selection instrument is
`docs/BENCHMARK.md`; pipelines, commands and recovery are in `docs/PIPELINES.md`; data sources in
`docs/REAL_FILM_DATASETS.md` and `docs/CURATED_PHOTOGRAPHY_SOURCES.md`.

Last updated: 2026-11-08 02:50 (session 4; the 20-hour orchestrator runs until 2026-11-08 17:47:57).

## Current objective
A lightweight, scientifically defensible film-aesthetic representation that transfers to real photographs
(DINOv2-Small research teacher, MobileNetV3 students for deployment): explainable attributes, cross-photo
film-style similarity, badge attributes. Film authenticity is a separate task with the EXIF student as its unchanged
baseline. The standing question is whether the representation captures treatment rather than content, generator
fingerprints, photographer, scanner or dataset shortcuts. Session 4 answered part of it on real film: across
photographers, uploader-claimed film stock is not recoverable by any current embedding.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 4: 0.2 metered GPU-minutes plus
about 6 unmetered (feature passes for three real-film evaluations and one curated evaluation); all sessions together
85.2 metered GPU-minutes, 0.46 GB charged plus 1.6 GB of newly acquired evaluation datasets (`docs/RESOURCE_USAGE.md`).
Nothing paid.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo style similarity (research) | hybrid (explicit + DINOv2 L4+L8) + family SupCon | benchmark v1 test: B fs 0.44 [0.39, 0.48], rank-1 0.33, gen share 0.59; BlueNeg same-stock p@5 0.25 [0.16, 0.31] (chance 0.12); **commoncatalog_film same-stock p@5 0.051 [0.036, 0.069] vs chance 0.044 (at chance)** | `reports/style_benchmark_test.json`, `reports/eval_pipeline/history.jsonl` |
| abstention for the above | top-1 cosine similarity, answer the 50 % most confident | BlueNeg rank-1 0.22 -> 0.37 at 50 % coverage; AURC 0.59 vs 0.78 | `reports/abstention_test.json` |
| lightweight student, candidate | `student_style_fs_v2_real` (E23b) | val B fs 0.557 vs 0.520; real teacher agreement 0.62; **not promoted**: at chance on commoncatalog_film (0.058), falsified on museum prints (E29) | `reports/student_style_fs_v2_real.json` |
| lightweight student, shipped research baseline | `student_style_fs_v2` (4.4 MB, 1.2 ms) | best on synthetic; BlueNeg 0.18; commoncatalog_film 0.049 (chance); not promoted | `style_benchmark_test.json` |

## What session 4 established
- Real film, across photographers (`commoncatalog_film`, 905 CC-BY scans, 252 uploaders, uploader-claimed stocks,
  two snapshots): same-stock p@5 is at chance for every embedding (E27b); stock-line labels give 0.078 vs 0.066 for
  the teacher and up to 0.101 for a student; colour-type (negative vs reversal) is a weak global-colour cue the
  teacher does not carry cleanly (E31 rejected: 0.670 [0.597, 0.726] vs 0.642). Raw DINOv2-global and colour
  statistics leak the photographer (uploader share 0.14 / 0.10 vs prior 0.03); the SupCon hybrid and students do
  not. The real-film retrieval evidence therefore rests on within-photographer structure (BlueNeg).
- The layer-3 target is closed without adoption: its synthetic gain is projection-bound (E28), it brings nothing on
  real film (E27b twice), and its student is the weakest on the stock-line metric and falsified on museum prints.
- Pooling two synthetic pipelines does not give generator invariance (E30).
- Curated CC0 museum photographs (`aic_photographs`, 593 images, 207 artists): the hybrid teacher retrieves the
  physical print process above chance (0.532 [0.453, 0.596] vs 0.389) with low artist leakage; the three
  real-distilled students lose that signal (lower CIs below chance). Real-image distillation did not buy
  cross-source robustness (E29).
- Data: API-based multilingual discovery with a persistent registry (187 sources decided); three new datasets
  acquired and validated (`commoncatalog_film`, `artefact_film`, `aic_photographs`); PhotoVogue, DEHAZED and Iconic
  Artist are metadata-only (terms prohibit automated access and ML use); Commons stays throttled.
- Infrastructure: 20-hour orchestrator with persisted loop state and graceful shutdown; evaluation script takes
  `--dataset`; the evaluation pipeline covers `commoncatalog_film`; curated evaluation script; 36 new tests (209 total).

## Live state at the time of writing
- Orchestrator loop PID 335761 (started 2026-11-07 21:47:57, deadline 2026-11-08 17:47:57, 15-minute ticks, 14 ticks,
  0 tick errors). Restart after a crash: `python -m research.orchestrator loop --interval 900 --resume`.
- All data tasks done except `data_commons_film_acquire` (retryable, attempt 5 at 05:53, 12 attempts max; every
  attempt so far got HTTP 429 on every rendition). All queued evaluations done. No research task pending: the loop
  idles between ticks by design.
- Datasets ready: commoncatalog_film 905, artefact_film 88, aic_photographs 593, plus the earlier seven. Test halves
  of the new sets untouched; benchmark v1 test split spent for the session-2 candidate set.

## Known limitations
- Stock labels are uploader claims (CommonCatalog, Commons) or one photographer's records (ARTeFACT, BlueNeg); the
  null result on cross-photographer stock retrieval cannot separate "no signal in the embeddings" from "labels too
  noisy" without owner-recorded stocks from many photographers, which no licensed source provides.
- Curated group is historic museum photography; no modern editorial photography is licensed for use.
- Students are not promoted; nothing is deployed; the EXIF authenticity baseline is unchanged.

## Rejected approaches (do not repeat without a new hypothesis)
- E3, E2/E4, E9/E9b, E11, E12, E16, E17, E18, E19, E10 (earlier sessions); E23c SDXL distillation inputs; E25 ridge
  residualisation; E26 Gram statistics and late layers; BlueNeg blue-corrupted as a selection metric; E28/E27b the
  L3 target; E30 pooled-pipeline SupCon; E31 cross-photographer colour-type retrieval as a teacher claim.

## Next recommended actions (in order)
1. Reframe the real-film claim: cross-photo similarity is supported within a photographer's rolls (BlueNeg), not
   across photographers by stock. Update `docs/MODEL_CARD.md` and `REPORT.md` wording accordingly before any product
   use of style similarity; the abstention threshold (E21) becomes more important, not less.
2. Next falsifiable hypothesis (pre-register before running): within-photographer same-stock retrieval on
   `commoncatalog_film` (uploader not excluded, date-grouped rolls) is above chance for the teacher beyond what
   photographer identity alone gives, measured against a photographer-only control (shuffle stocks within uploader).
   The report already shows uploader-included same-stock 0.068 [0.055, 0.082] vs chance 0.042; the control decides
   whether any of it is stock.
3. Augmentation-based invariance for the teacher features (E27's rejection path): photometric augmentation
   consistency on real unlabeled scans as an additional SupCon term; judge on commoncatalog_film stock-line p@5 and
   the museum process metric, never on synthetic B alone.
4. ARTeFACT: run the EXIF authenticity student on the 44 originals as a sanity check (expected high film
   probability); no selection use.
5. Commons: if attempt 5 is throttled again, leave it to the backoff; next session may cap files per uploader x
   stock and widen spacing; never fetch originals, never change identities.
6. Product: deterministic attribute layer plus the shadow-mode record with the abstention threshold; no deployment.
