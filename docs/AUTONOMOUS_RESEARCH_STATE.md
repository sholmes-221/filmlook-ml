# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`). Conversation context
is not the record. Validated claims with their evidence are in `docs/FINDINGS.md`; the research model card is
`docs/MODEL_CARD.md`; the selection instrument is `docs/BENCHMARK.md`; the pipeline architecture, commands and
recovery procedures are in `docs/PIPELINES.md`.

Last updated: 2026-11-07, end of the third autonomous session (Commons acquisition, E22-E26, pipeline separation).

## Current objective
A lightweight, scientifically defensible film-aesthetic representation that transfers to real photographs
(DINOv2-Small research teacher, MobileNetV3 students for deployment): explainable attributes, cross-photo
film-style similarity, badge attributes. Film authenticity is a separate task with the EXIF student as its unchanged
baseline. The standing question is whether the representation captures treatment rather than content, generator
fingerprints, photographer, scanner or dataset shortcuts; benchmark v1 answers it per candidate on synthetic data,
and the evaluation pipeline will answer it on a second real-film source.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 3 used about 11 GPU-minutes
(3 metered) and 0.02 GB; all sessions together 84 metered GPU-minutes, 0.34 GB (`docs/RESOURCE_USAGE.md`). Nothing paid.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo style similarity (research) | hybrid (explicit + DINOv2 L4+L8) + family SupCon | benchmark v1 test: B fs 0.44 [0.39, 0.48], rank-1 0.33, gen share 0.59; BlueNeg same-stock p@5 0.25 [0.16, 0.31] (chance 0.12) | `reports/style_benchmark_test.json` |
| abstention for the above | top-1 cosine similarity, answer the 50 % most confident | BlueNeg rank-1 0.22 -> 0.37 at 50 % coverage; AURC 0.59 vs 0.78 | `reports/abstention_test.json` |
| lightweight student, candidate | `student_style_fs_v2_real` (E23b: real unlabeled distillation inputs) | val B fs 0.557 vs 0.520; held-out real teacher agreement Spearman 0.62 vs 0.53; **not promoted**: awaits the Commons evaluation | `reports/student_style_fs_v2_real.json`, `student_teacher_agreement_style_fs_v2_real.json` |
| lightweight student, shipped research baseline | `student_style_fs_v2` (4.4 MB, 1.2 ms) | best on synthetic but BlueNeg 0.18 = DINOv2 reference; not promoted | `style_benchmark_test.json` |

## What session 3 established
- The student's real-scan failure is a distillation input-domain shift (E22); unlabeled real inputs close part of
  it (E23b) and SDXL inputs do not (E23c). Scanner family is not what style embeddings retrieve (E24). Linear content
  residualisation destroys the treatment signal (E25; do not repeat unchanged).
- E26 (12 layers x {mean++std, Gram32}): nothing beats L4+L8 under the pre-registered rule because the real-film check
  used (BlueNeg blue-corrupted) is at chance (0.565, one stock = 73 %); that partition is retired for selection. On
  synthetic data, layer-3 mean/std is CI-separated better (val B 0.572 vs 0.467) with less content leakage; Gram
  statistics are worse everywhere; layers >= 9 are content-heavy. Open hypothesis E27, to be judged on real film.
- A second real-film source: Wikimedia Commons film-brand categories (5,632 eligible CC0/PD/CC-BY(-SA) files, 401
  uploaders, 112 uploader-asserted stocks, roll and scanner categories for some), registered evaluation-only. The
  owner may veto it by deleting `data/raw/commons_film` and the `commons_film` entry in `configs/datasets.yaml`.
- Three independent pipelines with a local orchestrator (`docs/PIPELINES.md`): resumable rate-limit-aware
  acquisition with persistent per-file and per-dataset state; a research loop that runs backlog entries with
  runnable specs under pre-registered falsification rules; an evaluation pipeline that scores every eligible student
  on the Commons analysis half automatically when the dataset is ready (test half once, `--final`). 28 new tests.

## Live state at the stopping point
- `data_commons_film_acquire`: retryable, 266 of 5,743 items stored, 5,476 pending; Wikimedia throttles every
  rendition for this client (`Retry-After: 10`) after the first run's 3 hours of requests. The orchestrator loop
  (`reports/orchestrator/loop.log`, 15-minute ticks) retries with a 30-minute doubling backoff, 12 attempts. If the
  throttle does not lift within a day, the fallback is `--spacing 5` and a smaller stratified target (cap files per
  uploader x stock) - a decision for the next session, not a reason to fetch originals.
- `eval_commons_film_analysis`: pending on `dataset:commons_film` (needs >= 1,000 rows and a complete acquisition,
  or `python -m research.data_pipeline validate commons_film --allow-partial` on a recorded snapshot).
- Benchmark v1 test split is spent for the session-2 candidate set; BlueNeg blue-intact untouched since.

## Known limitations
- Real-film retrieval evidence still rests on BlueNeg (one photographer, 27 rolls); blue-corrupted is uninformative
  for selection. Commons labels will be uploader claims, not records. No human aesthetic judgement anywhere; no claim
  about beauty, stock identity or photographer identity.
- Students are not promoted; nothing is deployed; the EXIF authenticity baseline is unchanged.
- The research loop runs only entries that ship a script and a scalar falsification rule.

## Rejected approaches (do not repeat without a new hypothesis)
- E3, E2/E4, E9/E9b (superseded by E26), E11, E12, E16, E17, E18, E19, E10 (see earlier sessions).
- E23c SDXL outputs as distillation inputs; E25 ridge residualisation on DINOv2-global; E26 Gram statistics and
  late-layer (>= 9) targets; BlueNeg blue-corrupted as a selection metric.

## Next recommended actions (in order)
1. Let the acquisition finish (automatic). When `commons_film` is ready the evaluation task runs; read
   `reports/eval_pipeline/history.jsonl` and `python -m research.eval_pipeline status`. Decide on
   `student_style_fs_v2_real` from the analysis half only.
2. E27: add `--teacher-layers` to `scripts/train_style_student.py` and a `research/candidates_e27.py` (L3 and
   L2-L4 hybrid SupCon), evaluate on the Commons analysis half under the eval-pipeline rules; only then consider the
   distillation target.
3. If a Commons snapshot is partial for long, validate `--allow-partial` once (snapshot recorded) rather than waiting
   indefinitely; never evaluate the test half until the candidate set is final.
4. Product: deterministic attribute layer plus the shadow-mode record with the abstention threshold; no deployment.
