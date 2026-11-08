# Model card — FilmLook-ML research representations (2026-11-07)

This card covers the research artifacts produced by the autonomous sessions. The shipped product model
(`checkpoints/best`, measured-statistics attributes) is documented in `REPORT.md`; the Xiaohongshu Mini Tool is a
separate, unmodified project. Nothing in this card is deployed.

## Artifacts

| artifact | what it is | status |
|---|---|---|
| `checkpoints/student_sup` | MobileNetV3 student, EXIF capture-type (film vs digital) head | **authenticity baseline, keep** |
| hybrid + family SupCon projection | explicit 28 statistics ++ frozen DINOv2-small L4+L8 patch mean/std (1564-d) → 64-d, trained on filmsynth v2 train sources (`synth_experiment.supcon`) | research embedding for similarity; recomputed from the benchmark cache, not shipped |
| `checkpoints/student_style_fs_v2` | MobileNetV3 student distilled from the above (4.4 MB ONNX; outputs `treated_logit`, `family_logits`, `style_embedding`) | experimental; not promoted |
| `filmlook_infer` explicit statistics (28; colour/tone 16, texture 12) | deterministic, explainable attribute layer | shipped in the runtime package |
| `checkpoints/dinov2_baseline_v0`, `spatial_student_v0`, `student_style_fs`, `student_style_both` | earlier research checkpoints | kept for reproducibility |

## Intended use
- Research into whether photographic treatment can be represented separately from scene content.
- Shadow-mode logging of attribute candidates (`filmlook_infer/shadow.py`, hash-only) for a later privacy review.
- Authenticity (film vs digital) as a separate output with its own uncertainty.

Not intended for: identifying a film stock, a photographer, or a person; rating a photograph's quality; any
production decision without the abstention rule below.

## Training data
- Synthetic: 100 CommonCatalog CC-BY source photographs × filmsynth v2 presets (deterministic engine, grain model
  drawn independently of the label), SDXL base 1.0 img2img (research-only models; outputs used as labels, not
  shipped), Z-Image-Turbo img2img (Apache-2.0). Uploader-grouped split 70/15/15; no derivative crosses splits.
- Real: CommonCatalog CC-BY originals with EXIF capture type (authenticity), used for training the EXIF student
  and for the semantic-leakage probe only.
- Evaluation-only: BlueNeg (491 real 35 mm negatives, 27 rolls, owner-recorded stocks; credit "Copyrighted by
  Tien-Tsin Wong"; never trained on). Unsplash Lite is internal-use only and not used for training. One yearbook
  image is quarantined. No Claude output is a training target.

## Evaluation (benchmark v1, test split, one run; `docs/BENCHMARK.md`)
| metric | hybrid + family SupCon | student_fs_v2 64-d | colour/tone 16 | raw DINOv2 global |
|---|---|---|---|---|
| same treatment across scenes, filmsynth p@5 (chance 0.11) | 0.44 [0.39, 0.48] | 0.55 [0.49, 0.62] | 0.34 | 0.13 |
| own-source rank-1 (content leakage; 0 = none) | 0.33 | 0.16 | 0.52 | 0.99 |
| same-generator share of top-5 (prior 0.40) | 0.59 | 0.63 | 0.51 | 0.44 |
| BlueNeg same-stock p@5 (chance 0.12, roll CI) | 0.25 [0.16, 0.31] | 0.18 [0.13, 0.23] | 0.18 [0.09, 0.28] | 0.17 [0.13, 0.23] |
| authenticity, EXIF student within-source AUROC | — | — | — | 0.78 [0.69, 0.88], ECE 0.20 (`student_sup`) |

Synthetic and real numbers are reported separately on purpose; synthetic retrieval does not predict real-scan
retrieval (the student is best on synthetic and at the reference on real).

## Uncertainty and abstention
- Retrieval confidence: cosine similarity to the nearest neighbour from another image (`top1_sim`). For the
  hybrid SupCon embedding, answering only the 50 % most confident queries raises same-stock rank-1 precision
  from 0.22 to 0.37 on BlueNeg; below that threshold the system should abstain. Coverage is reported, not a
  probability (`scripts/abstention_analysis.py`).
- Authenticity probabilities from the EXIF student are not calibrated (ECE 0.20); treat them as scores.
- Every CI in this card is a bootstrap over sources (synthetic) or rolls (BlueNeg); BlueNeg intervals are wide
  because there are 27 rolls.

## Known failure modes
- Scene content: a treated image's nearest neighbour is its own source 16-33 % of the time.
- Generator fingerprint: neighbours over-concentrate on the query's generator (share 0.59-0.63 vs 0.40).
- Diffusion-generated treatments at strength 0.35 are indistinguishable from JPEG+rescale nuisance; labels from
  them are meaningless.
- One real-film source; results may not hold for other photographers, scanners or modern stocks.
- Scope of the similarity claim (updated 2026-11-08, sessions 4-5): "similar film look" is supported *within* a
  photographer's own scans (BlueNeg rolls; CommonCatalog within-uploader stock AUC 0.64 [0.56, 0.73] vs a permutation
  null of 0.45-0.55, largely the monochrome-vs-colour cue) and for the physical print process on museum prints, but
  *not* across photographers by stock: same-stock retrieval on 905 Flickr scans from 252 uploaders is at chance for
  every embedding. In the frozen DINOv2-Small statistics, film look and a photographer's look occupy the same
  directions (claims 23-26 in `docs/FINDINGS.md`), so photographer-style similarity and film-look similarity are one
  signal, to be shown as "similar look" with the abstention rule, never as a stock or photographer identification.
- The 64-d students reproduce a collapsed teacher projection (effective rank 2-3 of 64); on out-of-domain monochrome
  prints their neighbours are noise-dominated unless the target is changed (E38-E39) or the output is whitened.
- Research candidate after session 5 (not promoted, not deployed): the PCA-64 of the pooled 576-d features of
  `student_style_fs_v2_real_f576` (same network, 4.4 MB, 1.3 ms; the PCA is a 576x64 matrix fitted on CC-BY train
  originals). Held-out readings (one pass, E44): museum process p@5 0.593 [0.504, 0.683] vs chance 0.448, BlueNeg
  same-stock 0.191 [0.126, 0.263] vs 0.117 (teacher 0.248), benchmark B 0.589, own-source rank-1 0.32, effective rank
  9.0; its within-uploader stock AUC lower CI (0.452) missed the pre-registered bound, as every comparator's did on that
  half. `reports/final_holdout_e44.json`.
- What the training buys, and what it does not (session 6, E46-E50, analysis halves and the CC-BY validation split):
  the untrained ImageNet MobileNetV3-Small pooled layer with the same PCA-64 reaches the same museum process p@5
  (0.528 vs 0.515) but is content-dominated (own-source rank-1 0.93 vs 0.26-0.30) and separates stocks within a
  photographer only at 0.55 (trained 0.63-0.67); museum p@5 is reached by every representation down to 28
  deterministic statistics (0.594) and must not be read as evidence for the student on its own. The pooled PCA-64
  keeps its nearest neighbour under an 80 % centre crop and a flip (0.96-1.00) but loses it under JPEG q50 and 0.5x
  rescale 5-11 % of the time and under a 70 % corner crop 18-26 % of the time; an edit from an unseen edit pipeline
  moves the embedding 7-10 times as far as a crop for film and monochrome emulations and 0.3-0.5 times for
  over-sharpening and posterising. Neighbours share colour-tone statistics (0.65 of the random difference) more than
  texture statistics (0.85-0.87). A VICReg term on the pooled layer, or whitening its PCA components, raises every
  real metric and re-admits scene content (own-source 0.60-0.61); PCA truncation is the content filter, and 32
  components (float or int8) keep the signal within 0.01 of 64. Re-encoding half of the training crops (E52,
  `--aug-degrade 0.5`, `student_style_fs_v2_real_f576_deg{,_s1,_s2}`) lifts the JPEG q50 and 0.5x rescale readings
  to 0.98-1.00 and the corner crop to 0.84-0.88 with edit-type p@5 0.45 (vs 0.47) and the real matrix unchanged; that
  variant has analysis-half and validation readings only, no held-out reading. Claims 29-34 in `docs/FINDINGS.md`.

## Compute and cost
Local RTX 4090 only; no paid API or cloud GPU at any point. Session budgets in `docs/RESOURCE_USAGE.md`.
