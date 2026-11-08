# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`,
`research/ledger.json`, and the live pipeline state (`python -m research.orchestrator status`,
`cat reports/orchestrator/loop_state.json`). Conversation context is not the record. Validated claims with their
evidence are in `docs/FINDINGS.md` (claims 1-34); the research model card is `docs/MODEL_CARD.md`; the selection
instrument is `docs/BENCHMARK.md`; pipelines, commands and recovery are in `docs/PIPELINES.md`; data sources in
`docs/REAL_FILM_DATASETS.md` and `docs/CURATED_PHOTOGRAPHY_SOURCES.md`; the per-experiment record with the session
syntheses is `docs/EXPERIMENT_LOG.md`.

Last updated: 2026-11-08 14:25 (session 6; the 20-hour orchestrator loop PID 335761 runs until 2026-11-08 17:47:57).

## Current objective
A lightweight, reliable photography-style feature extractor for the Xiaohongshu Mini Tool: real-world generalisation,
compact size, fast inference, meaningful style representations, over synthetic benchmark scores (owner's session-6
brief). Film authenticity is a separate task with the EXIF student as its unchanged baseline. Session 6 judged the
session-5 candidate on what the product needs (reductions, no-training control, robustness, look-vs-content) and
closed the teacher-free rank-control line. Nothing is promoted or deployed.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 6: about 14 GPU-minutes (eleven
student retrains at ~40-80 s each, one DINOv2 pass over 1,582 images, the rest on cached features); ledger total 112
metered GPU-minutes and 0.5 GB charged across all sessions (`docs/RESOURCE_USAGE.md`). Nothing paid.

## Strongest lightweight representation (research; not promoted, not deployed)
PCA-64 (or PCA-32, float or int8) of the pooled 576-d features of the teacher-distilled students
`student_style_fs_v2_real_f576{,_s1,_s2}` (E23b recipe, seeds 0-2), PCA fitted on the 601 CC-BY train originals.
MobileNetV3-Small, 4.4 MB fp32 ONNX, 1.3 ms per image on one CPU thread at 224 px; the PCA is a 576x64 matrix.

| reading | value | where |
|---|---|---|
| museum process p@5, held-out half (one reading, E44) | 0.593 [0.504, 0.683] vs chance 0.448 | `reports/final_holdout_e44.json` |
| BlueNeg same-stock p@5, held-out (E44) | 0.191 [0.126, 0.263] vs 0.117 (teacher 0.248) | same |
| within-uploader stock AUC, held-out (E44) | 0.582 [0.452, 0.713] (rule failed; under-powered half) | same |
| analysis halves, three seeds (E43/E47) | museum 0.50-0.53 (lower CIs above 0.358), within-AUC 0.63-0.67, own-source rank-1 0.26-0.30, rank 8.6-8.9 | `e43_penultimate_pca.json`, `e47_pooled_reductions.json` |
| no-training control (E48) | ImageNet backbone + same PCA: museum 0.528, within-AUC 0.550, own-source 0.93 | `e48_frozen_vs_trained.json` |
| robustness, ccby validation (E49) | own-image rank-1: crop80 0.96-1.00, flip 0.99-1.00, JPEG q50 0.90-0.95, rescale 0.89-0.94, corner crop 0.74-0.82; edit-type p@5 0.46-0.48 vs 0.11 | `e49_robustness.json` |
| robust variant `student_style_fs_v2_real_f576_deg{,_s1,_s2}` (E52, `--aug-degrade 0.5`; analysis halves + validation only) | JPEG q50 0.98-1.00, rescale 0.99-1.00, corner crop 0.84-0.88; edit-type p@5 0.44-0.45; museum 0.49-0.51 (lower CIs above chance), within-AUC 0.62-0.64, own-source 0.30-0.31, rank 8.8-9.0 | `e52_degrade_augmentation.json`, `e52_robustness.json` |
| look vs content (E50) | neighbour colour-tone difference 0.65x random, texture 0.85-0.87x; content R^2 -0.02 | `e50_aesthetic_vs_content.json` |

Other models to quote: film authenticity `checkpoints/student_sup` (AUROC 0.78 [0.69, 0.88], ECE 0.20); attributes
v1 `checkpoints/best`; research teacher hybrid + family SupCon (BlueNeg 0.25 [0.16, 0.31]; 88 MB, 90 ms); abstention
top-1 cosine at 50 % coverage (BlueNeg rank-1 0.22 -> 0.37).

## What session 6 established (E46-E51; details in `docs/EXPERIMENT_LOG.md`, claims 29-33)
- Rank control on the pooled layer is closed: VICReg on the layer (E46) and VICReg on source-centred features (E51)
  raise rank to ~23 and the real metrics, and re-admit scene content (own-source 0.61 / 0.73 vs 0.28); the plain
  no-teacher recipe over three seeds confirms E45 (rank 8.0, museum 0.489, own-source 0.28).
- The content lives in the low-variance directions of the pooled layer: PCA 8-128 trace museum 0.45-0.52 against
  own-source 0.18-0.29; whitening gives 0.54 at 0.60; PCA-32 equals PCA-64 within 0.01 (falsified only on the rank-8
  bound written for 64 dims); int8 is free (E47).
- The training buys content invariance and within-photographer stock separation, not museum p@5, which every
  representation reaches (ImageNet-init 0.528, DINOv2 L4+L8 0.545, 28 statistics 0.594) (E48, supported).
- Robustness: crops and flips are safe; JPEG q50 and 0.5x rescale lose the nearest neighbour 5-11 % of the time (one
  seed 11.5 %, which falsified E49); the manifest's unseen edit pipeline is recognised at 0.47 vs 0.11 and moves the
  embedding 7-10x a crop for film/monochrome emulations.
- Neighbours share colour-tone statistics (0.65x random) more than texture (0.85-0.87x); the embedding is not a
  linear function of content, of the 28 statistics, or of provenance (E50; the R^2 sub-rule was trivial).
- Re-encoding half of the training crops (JPEG q40-70 / 0.5x rescale, `--aug-degrade 0.5`) closes the JPEG/rescale
  gap on all three seeds (0.98-1.00) at a bounded cost in edit sensitivity (edit-type p@5 0.45 vs 0.47, film-emulation
  displacement 6x vs 7.7x the crop) and no loss on the real matrix (E52, supported). This recipe is now the reference
  recipe for the next fresh-holdout reading; it has no held-out reading yet.
- Infrastructure: trainer `--kd 0` skips the DINOv2 pass, `--vic-pooled`, `--vic-mode source`, `--aug-degrade`;
  scripts E46-E52; `checkpoints/student_imagenet_init` (untrained control); six helper tests (`tests/test_session6.py`).

## Live state at the time of writing
- Orchestrator loop PID 335761 (started 2026-11-07 21:47:57, deadline 17:47:57, 15-minute ticks, 62 ticks, 0
  errors). Restart after a crash: `python -m research.orchestrator loop --interval 900 --resume`.
- `commons_film` is now `permanent_failure` (attempt 6 of 6 ended 14:18: HTTP 429/503 on every file, Retry-After
  honoured, 0 new files; 266 of 5,743 items stored from the first session). The pipeline will not retry on its own;
  a future session may `python -m research.data_pipeline reset commons_film` with wider spacing and lower caps,
  never with another identity or endpoint. No other task pending.
- Datasets ready: commoncatalog_film 905, artefact_film 88, aic_photographs 593, plus the earlier seven. Test halves
  of commoncatalog_film and aic_photographs, the filmsynth test split and BlueNeg were read once (E44) for the
  session-5 candidate and its comparators; session 6 read no test half.
- Checkpoints added this session (all research, none deployed): `student_style_fs_v2_real_{nokd_s1,nokd_s2,nvic1,
  nvic1_s1,nvic1_s2,svic1,svic1_s1,svic1_s2,f576_deg,f576_deg_s1,f576_deg_s2}`, `student_imagenet_init`.

## Known limitations
- Stock labels are uploader claims (CommonCatalog) or one photographer's records (BlueNeg, ARTeFACT); "process" on
  museum prints correlates with decade and subject even after decade matching, and is reached by untrained features.
- Content leakage is measured with 15 synthetic validation sources (own-source rank-1 is coarse) and, in E49, with 113
  real originals under eight edits.
- Photographer clustering (artist/uploader share) rises with every informative representation; recorded, not guarded
  (claim 25); the "similar look" signal includes the photographer's look.
- No human evaluation exists; neighbour attribute coherence (E50) is a proxy for "look".
- The within-uploader AUC rule is under-powered on both halves (15 multi-stock uploaders on the analysis half).

## Rejected approaches (do not repeat without a new hypothesis)
Earlier sessions: E3, E2/E4, E9/E9b, E11, E12, E16, E17, E18, E19, E10, E23c, E25, E26, E28/E27b (L3), E30, E31,
E33, E34, E35, E36, E38/E39, E41, E45.
Session 6: VICReg on the pooled layer, batch-wise (E46) and source-centred (E51); whitening of the pooled PCA (E47
row `pcaw64`); the ImageNet backbone without training as a candidate (E48 control). PCA-32 is not rejected on signal
(E47 fell on the rank bound); a 32-d rule needs its own pre-registered bound.

## Next recommended actions (in order)
1. Fresh holdout before any promotion: a third commoncatalog_film snapshot with higher per-uploader x stock caps
   (raises the within-uploader power) or a new museum source; the E44 halves are spent for the candidate. Until then
   every reading is analysis-half only.
2. The degrade-augmented recipe (E52) is the candidate for that holdout reading: PCA-64 (and PCA-32 int8 under its own
   pre-registered 32-d rule: rank >= 6, own-source <= 0.45, museum lower CI above chance, within-AUC lower CI above
   0.5) of `student_style_fs_v2_real_f576_deg{,_s1,_s2}`, read once, next to the E23b seeds and the teacher.
3. Corner-crop robustness (0.84-0.88 after E52) is the remaining geometric weakness; a random-crop-scale augmentation
   is the obvious next pre-registration, judged the same way as E52.
4. Explainable attributes: the deterministic 28 statistics already explain neighbour coherence (E50) and are a strong
   museum/within-uploader baseline (E48) at a scanner-probe cost (0.56); a product "similar look" can re-rank the
   embedding's neighbours by them without any model change.
5. ARTeFACT: EXIF authenticity student on the 44 originals as a sanity check (unchanged from session 4).
6. Commons: leave the backoff alone; never change identities.
7. Product: "similar look" with the abstention rule; stock and photographer matches are exploratory similarity
   estimates, never identifications; no deployment without the owner's approval.
