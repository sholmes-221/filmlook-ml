# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`,
`cat reports/orchestrator/loop_state.json`). Conversation context is not the record. Validated claims with their
evidence are in `docs/FINDINGS.md` (claims 1-28); the research model card is `docs/MODEL_CARD.md`; the selection
instrument is `docs/BENCHMARK.md`; pipelines, commands and recovery are in `docs/PIPELINES.md`; data sources in
`docs/REAL_FILM_DATASETS.md` and `docs/CURATED_PHOTOGRAPHY_SOURCES.md`; the per-experiment record with the session
syntheses is `docs/EXPERIMENT_LOG.md`.

Last updated: 2026-11-08 09:00 (session 5; the 20-hour orchestrator loop PID 335761 runs until 2026-11-08 17:47:57).

## Current objective
A lightweight, scientifically defensible film-aesthetic representation that transfers to real photographs
(DINOv2-Small research teacher, MobileNetV3-Small students for deployment): explainable attributes, cross-photo
"similar look" retrieval, badge attributes. Film authenticity is a separate task with the EXIF student as its unchanged
baseline. Session 5 answered the E29 question (what the teacher learned, why the students lost it) and found the first
student representation with a held-out gain; nothing is promoted or deployed.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 5: about 13 GPU-minutes (eleven
student retrains at ~1 minute each, two feature passes, the rest on cached features); ledger total 96 metered
GPU-minutes and 0.5 GB charged across all sessions (`docs/RESOURCE_USAGE.md`). Nothing paid.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo similar look, research teacher | hybrid (explicit + DINOv2 L4+L8) + family SupCon | BlueNeg same-stock p@5 0.25 [0.16, 0.31] (chance 0.12); museum process (test half, decade matched) 0.535 vs 0.448; cross-photographer stock at chance | `reports/style_benchmark_test.json`, `reports/final_holdout_e44.json` |
| abstention | top-1 cosine, answer the 50 % most confident | BlueNeg rank-1 0.22 -> 0.37 at 50 % coverage | `reports/abstention_test.json` |
| lightweight research candidate (session 5, **not promoted**) | PCA-64 of the pooled 576-d features of `student_style_fs_v2_real_f576` (PCA fitted on CC-BY train originals) | held-out: museum 0.593 [0.504, 0.683] vs 0.448; stock-line 0.103 vs 0.077; BlueNeg 0.191 [0.126, 0.263]; B 0.589; own-source rank-1 0.32; rank 9.0; within-uploader AUC lower CI 0.452 (rule failed) | `reports/final_holdout_e44.json`, `e43_penultimate_pca.json` |
| lightweight student, shipped research baseline | `student_style_fs_v2` 64-d head (4.4 MB, 1.2 ms) | best on synthetic; BlueNeg 0.18; collapsed (rank 3) | `style_benchmark_test.json` |

## What session 5 established (E32-E44; details in `docs/EXPERIMENT_LOG.md`, claims 23-28)
- E29's teacher signal is print rendering, inflated by format: with stereo cards removed and the decade matched the
  teacher stays above chance (0.483 [0.401, 0.560] vs 0.358); colour-tone and the raw L4+L8 statistics carry more of it
  than the family projection.
- The family SupCon projection is collapsed (effective rank ~4) and the students inherit it (rank 2-3, mean cosine
  0.6-0.8 on monochrome prints); whitening restores their museum signal. That is why the real-distilled students
  "lost" E29: a collapsed target reproduced faithfully.
- On frozen DINOv2-Small statistics, every head that keeps real-photograph information lets scene content in
  (own-source rank-1 0.7-1.0) and clusters by photographer; collapse controls, photographer-subspace removal,
  independent whitening and treatment-parameter supervision all trade the same way (E33-E36, E40, E41). The content-
  invariant embeddings are the collapsed ones.
- Within a photographer, stock lines are separable (AUC 0.640 [0.563, 0.734] vs a permutation null of 0.45-0.55),
  mostly through the monochrome-vs-colour cue; colour-only is underpowered (p = 0.06).
- A non-collapsed distillation target (SupCon + VICReg) gives the student a small, seed-consistent museum gain with no
  synthetic cost, but one seed misses chance by 0.004 and artist share doubles (E38/E39, falsified).
- The end-to-end student's 576-d pooled layer is the first representation with rank > 8, own-source rank-1 < 0.45 and
  museum lower CIs above chance on three seeds; a label-free PCA-64 keeps it (E42/E43 supported). Its one held-out
  reading confirms the museum gain and no cost on B or BlueNeg, but the within-uploader AUC lower CI failed the
  pre-registered rule, as it did for every comparator on that half (E44 falsified; not promoted).
- The pooled-layer signal is the same without the teacher term (E45: `--kd 0` pooled museum 0.494 vs 0.517, PCA-64
  rank 7.99 vs the 8 the rule asked, own-source 0.24): the teacher's contribution lands in the collapsed head only.
- Infrastructure: `scripts/student_real_matrix.py` (the real-photograph matrix with own-source rank-1 and effective
  rank), `scripts/features_for_manifest.py`, `--kd-target` in the trainer and the 576-d export, nine helper tests
  (208 tests pass), all experiments pre-registered and recorded through the research loop.

## Live state at the time of writing
- Orchestrator loop PID 335761 (started 2026-11-07 21:47:57, deadline 17:47:57, 15-minute ticks, 40 ticks, 0 errors).
  Restart after a crash: `python -m research.orchestrator loop --interval 900 --resume`.
- `data_commons_film_acquire` retryable (attempt 6 at 14:02, 12 max; every attempt so far HTTP 429 on every file).
  No other task pending. The loop idles between ticks by design.
- Datasets ready: commoncatalog_film 905, artefact_film 88, aic_photographs 593, plus the earlier seven. Test halves of
  commoncatalog_film and aic_photographs, the filmsynth test split and BlueNeg were read once (E44) for the session-5
  candidate and its three comparators; they are spent for those representations.
- Checkpoints added this session (all research, none deployed): `student_style_fs_v2_real_{s1,s2,vic3,vic3_s1,vic3_s2,pcaw,f576,f576_s1,f576_s2,nokd,vic3f}`.

## Known limitations
- Stock labels are uploader claims (CommonCatalog) or one photographer's records (BlueNeg, ARTeFACT); "process" on
  museum prints correlates with decade and subject even after decade matching.
- Content leakage is measured with 15 synthetic validation/test sources (own-source rank-1 is coarse).
- Photographer clustering (artist/uploader share) rises with every informative representation; it is recorded, not
  guarded, because it is inseparable from film look in these features (claim 25) and is itself a product feature.
- No human evaluation exists; nothing here says what a person would call a film look.

## Rejected approaches (do not repeat without a new hypothesis)
Earlier sessions: E3, E2/E4, E9/E9b, E11, E12, E16, E17, E18, E19, E10, E23c, E25, E26, E28/E27b (L3), E30, E31.
Session 5: the no-teacher pooled layer as a promotion candidate at rank 7.99 (E45, by 0.01; repeat only with seeds); label-free heads on frozen features as a student target (E33), VICReg/temperature rank controls on the
projection (E34), LDA photographer-subspace removal (E35), post-hoc whitening of the students (E36), the VICReg
distillation target as a promotion candidate (E38/E39), treatment-parameter heads with VICReg (E41).

## Next recommended actions (in order)
1. Student head that keeps the pooled layer's rank (pre-register before running): since E45 shows the teacher term adds
   nothing measurable to the pooled layer, train the recipe with `--kd 0` plus a VICReg term on the pooled layer (or a
   PCA-64 of a first-pass student's pooled features as the target) over three seeds; judge on the analysis-half matrix (rank >= 8, own-source <= 0.45,
   museum lower CI above chance, within-AUC, synthetic cost); the test halves are spent for the current candidate, so a
   new candidate needs a fresh holdout decision (a third snapshot of commoncatalog_film or a new museum source).
2. Within-uploader power: the test half carries too few multi-stock uploaders for the AUC rule; acquire more scans per
   uploader x stock (raise the 6/uploader x stock cap on a new snapshot) before using that rule as a gate again.
3. Explainable attributes: the ridge treatment-parameter estimate (E41 `params_ridge6`, own-source 0.39) is the
   cheapest interpretable layer; evaluate it against the deterministic attributes on real scans, no selection use.
4. ARTeFACT: EXIF authenticity student on the 44 originals as a sanity check (unchanged from session 4).
5. Commons: leave the backoff alone; next session may widen spacing and caps; never change identities.
6. Product: "similar look" (one signal for film look and photographer look) with the abstention rule; no stock or
   photographer identification; no deployment.
