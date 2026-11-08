# Recommendation: the simplest architecture that demonstrably works (2026-11-08)

## Recommendation

Ship a **statistics-first hybrid** for the Gemini-free backend:

```
photo → validation, 2048 px copy (existing)
      → 28 measured statistics (existing filmlook_infer explicit features; ~100 ms on a fast core)
      → MobileNetV3-Small student ONNX, PCA-32 int8 (frozen_session7; 1.5 ms) for robustness and similar-look retrieval
      → student_sup ONNX fp32 (4.3 MB): P(film scan) + temperature/saturation/contrast/key/sharpness heads
      → rendering-family prototype classifier on the statistics (4 centroids; 0.85 balanced accuracy)
      → rule-based style suggestion within the tier's library of defined looks (never a stock claim)
      → measured 17-axis photographer ranking (existing, unchanged)
      → optional similar-look neighbours from a licensed bank with the E55 abstention gate
      → Chinese and English templates filled from the above; film-likeness shown as an *experimental estimate*
```

Why this and not something cleverer:

- On fresh data the 28 statistics are the best single representation for every rendering task (E53, E57) and
  cost nothing to serve; every learned alternative tried is equal or worse on labels and adds leakage.
- The frozen student adds the one thing the statistics lack, robustness to re-encoded uploads (JPEG q50,
  0.5x rescale: 0.95 to 0.99 vs 0.45), and the lowest photographer leakage; it is 4.4 MB and 1.5 ms.
- The attribute heads are the only *validated* characteristic readings (0.81 to 0.89 against known edits).
- P(film scan) is the only validated score-like output (AUROC 0.78 within source); it is shown as an estimate
  with its limits, not as the Gemini "Film Look Score".
- Prototypes need 4 vectors, not a bank; retrieval stays optional and gated.
- Everything runs in NumPy + ONNX Runtime on one thread; expected under 150 MB RSS and under 1.5 s per photo on
  the E2 Micro core (to be measured there before cutover; the service benchmark script exists).

## What it will and will not say

Says: measured characteristics (warm/cool, saturation, contrast, key, texture, grain visibility); a rendering
family with its accuracy; a film-likeness estimate labelled experimental; the look in the library closest to
the measured character, worded as resemblance; the measured photographer ranking under the library rules;
"looks similar to" neighbours only when the gate passes. Does not say: which film was used, which photographer
took it, a mood, a subject, or an aesthetic judgement.

## Gate before production (unchanged from docs/ml-self-hosted-mvp.md in the web repository)

Fresh film-vs-digital AUROC ≥ 0.75 with lower CI ≥ 0.65 and calibration ECE ≤ 0.10 after Platt scaling on a
separate half; robustness flips ≤ 5 %; attribute heads re-checked; instance benchmark p95 < 3 s, RSS < 200 MB;
application tests green including "no external request" in that mode; a comparison with the provider output on
the same photos for completeness and consistency only.

## Alternatives considered and rejected

| alternative | why not |
|---|---|
| DINOv2-Small in production | 88 MB, ~400 MB RSS, content-dominated (uploader share 0.22, within-uploader AUC 0.49) |
| CLIP + LAION aesthetic head | wrong task (preference), 430 M-parameter backbone, weights unlicensed |
| a learned film-look score head | no ground truth exists; would be trained on a proxy and presented as more |
| stock classifier / stock retrieval | 0.12 balanced accuracy vs 0.043 chance on fresh data |
| faiss / vector database | 100k references are 3 to 12 MB; brute-force NumPy answers in milliseconds |
| another custom architecture | the fresh-data readings say representation is not the bottleneck; labels and licences are |

## Order of work

1. Export parity check and preprocessing alignment for `student_sup` (fp32 only; the int8 export is broken).
2. Prototype vectors for the rendering family from the Met fit half; store with the statistics' mean/std.
3. Platt calibration of P(film scan) on a fresh fit half; fresh read on the held half (the E53 scan snapshot's
   uploader-excluded split can serve as fit/read halves for the calibration only).
4. `filmlook_infer` analysis adapter producing the structured result; templates; `ANALYSIS_BACKEND=retrieval`
   in the web repository behind the existing `filmlook_ml` switch.
5. Instance benchmark, then the gate.
