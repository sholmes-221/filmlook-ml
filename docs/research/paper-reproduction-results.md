# Reproduction and comparison results (E57, 2026-11-08)

What was reproduced: the *method family* of Karayev et al. (linear/kNN heads on generic features plus colour and
texture statistics) and the CBIR pattern (embedding + cosine kNN), on FilmLook's own fresh evaluation sets.
What was not reproduced: the LAION/CLIP aesthetic heads (task mismatch and a 430 M-parameter backbone), VILA
and SAAN (size; the SAAN page was unreachable), pyiqa (non-commercial licence).

Command: `.venv/bin/python scripts/e57_retrieval_methods.py` → `reports/e57_retrieval_methods.json` (88 s on the
development host; DINOv2 on the GPU for the comparison row only). Data: the Met analysis half (378 CC0
images, 140 artists; label = catalogue process family, 4 classes; candidates decade-matched, non-stereo, artist
excluded) and the fresh CommonCatalog film-scan snapshot (324 CC-BY-2.0 scans, 91 never-seen uploaders; label =
uploader-claimed stock line, 23 classes; uploader excluded). Both sets were read once by E53 under a
pre-registered rule; **E57 is a post-hoc method comparison on the same images and is labelled exploratory**.

## Representations

A = frozen student MobileNetV3-Small, PCA-32 int8 (the production candidate); B = frozen DINOv2-Small global,
PCA-64; B' = untrained ImageNet MobileNetV3-Small, PCA-64; C = 28 standardised statistics; H(x, w) = concat of
the unit-normalised x with w × the unit-normalised C.

## Process-family task (Met; 4 classes; balanced chance 0.25; 345 queries)

| representation | p@5 (artist excl.) | artist share ↓ | 5-NN plain bal. acc | 10-NN weighted | 10-NN class-balanced | 20-NN balanced | prototype | acc (10-NN bal.) [CI over artists] | ECE (10-NN bal.) |
|---|---|---|---|---|---|---|---|---|---|
| C statistics | 0.746 | 0.130 | **0.847** | 0.841 | 0.744 | 0.739 | 0.707 | 0.643 [0.572, 0.714] | 0.186 |
| A student PCA-32 int8 | 0.694 | **0.080** | 0.677 | 0.602 | 0.415 | 0.273 | 0.550 | 0.525 [0.448, 0.606] | 0.280 |
| B' ImageNet init | 0.687 | 0.111 | 0.632 | 0.593 | 0.536 | 0.539 | 0.573 | 0.556 [0.489, 0.624] | 0.254 |
| B DINOv2 global | 0.660 | 0.113 | 0.576 | 0.596 | 0.595 | 0.556 | 0.450 | 0.588 [0.517, 0.656] | 0.216 |
| H(A, 2) | 0.746 | 0.145 | 0.836 | 0.828 | 0.743 | 0.722 | 0.718 | 0.623 [0.548, 0.701] | 0.199 |
| H(B', 2) | **0.757** | 0.156 | 0.846 | 0.817 | 0.773 | 0.738 | 0.734 | 0.632 [0.553, 0.705] | 0.193 |
| H(B, 1) | 0.740 | 0.159 | 0.841 | 0.787 | **0.786** | 0.753 | **0.745** | **0.661** [0.595, 0.727] | **0.168** |

Reading: on rendering-family recognition the 28 statistics are the strongest single representation; adding a
learned embedding to them changes balanced accuracy by at most +0.01 (5-NN) and raises artist leakage (artist
share 0.13 → 0.15). Plain 5-NN beats class-balanced 10/20-NN here because the classes are imbalanced in the
*same* direction as the catalogue (albumen-heavy): balancing trades accuracy for fairness across families.
Calibration is poor for every method (ECE 0.17 to 0.28; mean confidence 0.80 to 0.83 against accuracy 0.53 to
0.66): vote shares must not be shown as probabilities without a calibration step.

## Film-stock line task (scans; 23 classes; balanced chance 0.043; 311 queries)

| representation | p@5 (uploader excl.) | uploader share ↓ | within-uploader AUC | best classifier (bal. acc) | ECE |
|---|---|---|---|---|---|
| C statistics | 0.119 | 0.181 | 0.691 | 20-NN balanced 0.089 | 0.43 |
| A student | 0.141 | **0.131** | 0.678 | prototype 0.124 | 0.36 |
| B' ImageNet init | 0.147 | 0.156 | 0.560 | 10-NN balanced 0.079 | 0.39 |
| B DINOv2 global | 0.109 | 0.222 | 0.490 | 20-NN plain 0.037 | 0.41 |
| H(A, 2) | 0.127 | 0.202 | 0.689 | 20-NN balanced 0.046 | 0.43 |
| H(B', 2) | 0.141 | 0.204 | 0.682 | 20-NN balanced 0.097 | 0.41 |
| H(B, 1) | 0.151 | 0.251 | 0.672 | 10-NN balanced 0.089 | 0.40 |

Reading: no representation and no method identifies a stock line across photographers (best 0.12 against
0.043 chance, with a lower CI touching chance for most). Within one photographer, the student, the statistics
and the hybrids separate that photographer's stocks (AUC 0.67 to 0.69) while DINOv2 does not (0.49): the
learned embedding and the statistics carry a *rendering* signal, the semantic embedding carries *content*.
The student has the lowest uploader share (0.13), i.e. the least photographer leakage, which is what it was
trained for.

## Subject bias versus style

Uploader share and artist share are the direct measures of "retrieves the same photographer's pictures"
(content and habit) rather than style. Ordering from least to most leakage on scans: A 0.13 < B' 0.16 < C
0.18 < H(A,2) 0.20 ≈ H(B',2) 0.20 < B 0.22 < H(B,1) 0.25. The learned student is the least subject-driven
representation; the semantic embedding is the most. Adding statistics to any embedding raises leakage slightly
because the statistics themselves carry the scanner and the photographer's processing habits.

## Teacher offline, student online

The within-uploader result (A 0.68 vs B 0.49) and the robustness readings from E53 (own-image rank-1 under JPEG
q50 / 0.5x rescale 0.95 to 0.99 for A vs 0.45 for C under rescale) say that the distilled student is a valid
production encoder for *re-encoded user photographs*, while the teacher is not needed at inference. Reference
and query embeddings must come from the same frozen ONNX file and the same preprocessing (short side 384,
centre 224, ImageNet normalisation): the service already enforces this by hashing the model file.

## Storage and brute-force latency (NumPy, one core of the development host)

| representation | bytes/image | 10k | 50k | 100k | top-5 brute force (100k) |
|---|---|---|---|---|---|
| PCA-32 int8 | 32 | 0.3 MB | 1.6 MB | 3.2 MB | 4.5 ms |
| hybrid 60-d float16 | 120 | 1.2 MB | 6.0 MB | 12 MB | 1.7 ms |
| PCA-64 float32 | 256 | 2.6 MB | 12.8 MB | 25.6 MB | 2.9 ms |

Even 100k references fit in a few MB and a brute-force dot product answers in milliseconds; faiss is not needed
at this scale (the E2 Micro core is several times slower; still well under 50 ms).

## Limits of this reproduction

Post-hoc on sets already read once; 4-class process labels are museum catalogue records, not photographic
style in the Karayev sense; stock labels are uploader claims; no human judgement of "looks like" exists; ECE is
computed on vote shares, which no method calibrated. The numbers are method comparisons on fixed data, not a
new pre-registered test.
