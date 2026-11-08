# Evaluation of the statistics-first MVP (E57 methods, E58 calibration; 2026-11-08)

Scripts: `scripts/e57_retrieval_methods.py` → `reports/e57_retrieval_methods.json`;
`scripts/e58_filmlikeness_calibration.py` → `reports/e58_filmlikeness_calibration.json` and the shipped artifacts in
`checkpoints/retrieval_mvp/`. Both are post-hoc on sets that E53 read once under a pre-registered rule; they are
method comparisons and calibrations, not new pre-registered tests. Labels are catalogue records, EXIF capture
medium and owners' claims: weak but independent of every model.

## Film-likeness (E58)

Data: positives = 30 TEST-split film-scan originals (CommonCatalog, EXIF) + 324 fresh `commoncatalog_film_hd`
scans; negatives = 59 TEST-split digital-camera and smartphone originals. Groups (uploader / group id) split by
seed into a FIT half (Platt fitting) and a READ half (180 film, 30 digital, 86 groups). CIs: bootstrap over groups.

| signal | AUROC (READ) | 95 % CI | ECE after Platt | ECE raw | within-source rows only (test originals) |
|---|---|---|---|---|---|
| v1 `film_score` (checkpoints/best) | **0.49** | [0.35, 0.65] | 0.08 | 0.15 | 0.62 |
| `student_sup` P(film scan) | **0.86** | [0.79, 0.92] | 0.07 | 0.30 | 0.80 |
| both, logistic combination | 0.75 | [0.62, 0.88] | 0.06 | n/a | 0.77 |

Shipped: the student head alone with its Platt map (`calibration.json`). Robustness of the shipped estimate on
the READ half: JPEG q50 → AUROC 0.86, decision flips at 0.5: **0 %**, mean shift 0.8 points, p95 3.7 points;
0.5x rescale → AUROC 0.87, flips 0 %, mean shift 2.7, p95 11.9. The estimate is therefore shown with band wording
and a rounded percentage, and the 95th-percentile shift under heavy resampling is stated in the docs.

Caveats: 30 negatives in the READ half (the CI is wide); positives are dominated by one fresh snapshot, so part of
the separation may be source-specific (the within-source reading, 0.80, says most of it is not); provenance
labels are weak; the estimate is a *resemblance to real film scans*, never proof that film was used.

## Rendering-family prototypes (E58): rejected for the MVP

Centroids fitted on the AIC analysis half (5 families with ≥ 15 images) and read on the Met analysis half:
balanced accuracy **0.23** against chance 0.20 (accuracy 0.16; albumen 0.16, daguerreotype 0.33, photogravure
0.00). Within one collection (E57, artist excluded) the same statistics reach 0.85; across collections the
digitisation pipeline dominates. The artifact is kept as experimental and is not used by `RetrievalMVP`.

## Retrieval methods (E57, summary)

Rendering-family p@5 / balanced accuracy within the Met collection: statistics 0.746 / 0.847 (5-NN), hybrids
0.74 to 0.76 / 0.84 to 0.85, student 0.69 / 0.68, DINOv2 0.66 / 0.58. Stock line across uploaders: 0.11 to 0.15
p@5 against 0.08, best balanced accuracy 0.12 against 0.043. Within-uploader stock AUC: student 0.68,
statistics 0.69, DINOv2 0.49. Uploader share (leakage, lower is better): student 0.13, statistics 0.18, DINOv2
0.22. K: plain 5-NN best for the rendering task; class-balancing lowers accuracy on this skewed catalogue.
Calibration of vote shares: ECE 0.17 to 0.43 for every method (not shippable as probabilities).

## Characteristics (from `reports/student_sup.json`, 127 held-out originals, labels = known edits)

temperature 0.89, saturation 0.84, contrast 0.85, key 0.87, sharpness 0.81, grain 0.59 (grain is therefore read
from the measured statistics in the MVP). ONNX fp32 agrees with PyTorch on 100 % of top-1 labels; the int8 export
does not (0.11 to 0.47) and is not used.

## Subject bias versus style

Measured by artist / uploader share of the top five (E57): the learned student is the least subject-driven
representation; DINOv2 the most; adding statistics to an embedding raises leakage slightly (the statistics carry
scanner and processing habits). The MVP does not retrieve neighbours at all today, so subject bias enters only
through the attribute heads, which were trained on edits of the same photographs (content held constant).

## What remains unmeasured

A fresh digital negative set; coverage and sanity of the look rules on a held-out photo set with varied subjects;
the instance benchmark (`docs/ml/oracle-benchmark.md`); the side-by-side comparison with the provider output on the
same photographs (completeness and internal consistency only; no ground truth exists for the provider's text).
