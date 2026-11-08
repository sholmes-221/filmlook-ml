# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`,
`cat reports/orchestrator/loop_state.json`). Conversation context is not the record. Validated claims with their
evidence are in `docs/FINDINGS.md` (claims 1-40); the research model card is `docs/MODEL_CARD.md`; the MVP decision is
`docs/MVP_DECISION.md`; the selection instrument is `docs/BENCHMARK.md`; pipelines, commands and recovery are in
`docs/PIPELINES.md`; data sources in `docs/REAL_FILM_DATASETS.md` and `docs/CURATED_PHOTOGRAPHY_SOURCES.md`; the
per-experiment record with the session syntheses is `docs/EXPERIMENT_LOG.md`.

Last updated: 2026-11-08 18:40 (session 7). **Research is stopped** by the owner's instruction: finish E53-E56, select
an MVP, build the CPU service, write the decision. No session 8 is planned; do not start open-ended research or
training unless the owner asks.

## Where things stand
- The owner's MVP brief (2026-11-08, afternoon) was executed: fresh holdouts acquired, the candidate frozen before the
  read, E53 read once, E55 and E56 run, E54 falsified, the CPU service built and measured, `docs/MVP_DECISION.md` written.
  The 20-hour orchestrator loop (PID 335761) reached its deadline at 17:47:57 and stopped by itself; nothing is scheduled.
- **Decision: LIMITED GO** for an exploratory "similar look" embedding (frozen `style_fs_v2_real_f576_deg`, PCA-32 int8,
  32 bytes per image, with the 28 measured statistics as the explanation layer and the abstention gate), **NO-GO** for
  film-stock or photographer identification. Nothing promoted, deployed or pushed; the production FilmLook application
  and the Mini Tool are unchanged.

## Frozen candidate (do not retrain, do not re-freeze)
`checkpoints/frozen_session7/` (chmod 444): the three E52 seeds, the three E23b seeds, the untrained ImageNet control,
each with `pca.npz` fitted on the 601 CC-BY train originals; SHA-256s in `freeze.json` and `docs/FROZEN_CANDIDATE_S7.md`.
Service: `service/filmlook_style_service.py` (stdlib HTTP + ONNX Runtime; 65 / 104 / 135 MB RSS idle / loaded / peak,
21 ms per embedding, 113 ms with the statistics on one core here; `reports/service_bench.json`), `service/README.md`,
`service/filmlook-style.service`, `service/build_bank.py`.

## Fresh holdouts and what remains unread
- `met_photographs` (698 CC0 images, 267 artists; `data/raw/met_photographs`): the analysis half (383 rows, 5 strips
  under 256 px dropped) was read once in E53. **The test half (315 rows) is untouched.**
- `commoncatalog_film_hd` (474 scans, 133 uploaders new to the project, 51 stocks): read in full once in E53 (the 324
  rows with a stock line); nothing of it remains unread. A further snapshot would need the remaining 3,210 shards of the
  2048-4096 px range (about 7 h of metadata scanning at 8 workers for roughly 500 fresh rows).
- `artefact_film` originals (44): read once in E53 as a recorded reading.
- Spent earlier: `commoncatalog_film` both halves, `aic_photographs` both halves, filmsynth test split, BlueNeg (E44).
- `commons_film` stays `permanent_failure` (HTTP 429/503 after six attempts); never change identities or endpoints.

## Session 7 results in one paragraph (details: `docs/EXPERIMENT_LOG.md`, claims 35-40)
On fresh data the student's only statistically supported advantage over the untrained ImageNet backbone with the same
PCA is within-photographer stock separation (AUC +0.128 [+0.039, +0.232] over 93 queries); it ties the 28 measured
statistics there, trails them on museum process p@5 by 0.05 [0.02, 0.08], equals the untrained backbone on museum and
stock-line p@5, and is below both the untrained backbone and DINOv2 on ARTeFACT's owner-labelled emulsions. Its product
value is robustness (JPEG/rescale 0.95-0.99 where the E23b seeds lose a quarter and the statistics half) and content
invariance (uploader share 0.13 vs 0.18). The pre-registered E53 rule was falsified by the museum CI guard (chance 0.648
on an albumen-dominated half) and the 80 % crop guard (0.87 on a 324-image bank). E55: the abstention gate lifts museum
label precision marginally and unstably; stock claims stay unusable. E54: edits are encoded monotonically but not as
linear axes. E56: no linear stock / scanner / uploader signal in any representation; the stock signal does not need the
scanner. Costs: under 3 GPU-minutes this session; ledger 7,232 GPU-seconds across all sessions.

## If the owner reopens research (not before)
1. Gather human "similar / not similar" judgements from the product first; every metric so far is catalogue metadata.
2. Then a single confirmatory read on the untouched Met test half (315 rows), pre-registered with a chance-aware guard
   (the albumen prior must be matched or the guard dropped) and a crop bound set from the scan-bank size.
3. Corner-crop augmentation remains the obvious next recipe change, judged as E52 was.

## Standing constraints (unchanged)
No deployment, no push, no human annotation requests, no paid APIs or cloud GPU, no Claude outputs as training targets,
HF tags never override copyright, BlueNeg credit "Copyrighted by Tien-Tsin Wong", Unsplash Lite internal only, the
quarantined yearbook image stays quarantined, EXIF authenticity baseline `checkpoints/student_sup` preserved, datasets and
checkpoints never deleted, never claim film-stock or photographer identification.
