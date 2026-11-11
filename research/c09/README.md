# C09: local prediction of the Claude style annotation

Question: can a small local head that predicts the structured style annotation (Haiku, style-annotation-v2, subject
fields removed: 211 targets) from local image features recover the +0.01 to +0.04 photographer-matching gain that the
real annotation gives, with no Claude call at inference?

    python -m research.c09.bank_inputs              # data/c09/bank_inputs.npz (F36 + six DINOv2-small parts, production path)
    python -m research.c09.local_predictors --seed 0   # reports/c09/local_predictors_seed0.json, dist/c09/style_head.npz
    python -m research.c09.local_predictors --seed 1
    python -m research.c09.local_predictors --pool     # reports/c09/local_predictors_pooled.json

Protocol. Same leakage-grouped split, 27 classes, top-3 scorer and cross-subject protocol as
`research/photographers/subject_audit.py`. The heads are trained on BANK-split works only (about 2,700). Hyper-parameters
and the predictor are chosen on VAL by validation loss, and the fusion weight w is chosen on VAL top-1 under the
cross-subject protocol. TEST is read once. Loss: softmax cross-entropy per single-choice field, plus BCE for the multi-label
flags and the 0..4 axes. The query's Claude subject label is used only to define the cross-subject protocol. It is never
an input. Training targets are Claude outputs, used under the owner's recorded permission (`training_authorization:
user_confirmed`; preflight PASS). No API calls were made.

Predictors: A = linear multinomial/logistic heads on L12 [CLS, mean] (768 inputs; weight decay grid 1e-3 to 1e-1, chosen
3e-2). B = MLP, 2 x 256, GELU, dropout, on all six DINOv2 parts (2304). C = the same MLP on 2304 + F36 + axes17 + film205.
**B had the lowest validation loss on both seeds and is the selected predictor.**

## Predictive quality on TEST works

AUROC is the mean over the binary or one-hot columns in each group. Spearman is the mean over the 7 remaining 0..4 axes.

| Predictor | seed | film_rendering | composition | motifs | style_traits | mood | palette_tone | light | camera_space | all binary | ordinal rho |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A linear L12 | 0 | 0.847 | 0.889 | 0.903 | 0.904 | 0.857 | 0.880 | 0.900 | 0.904 | 0.888 | 0.620 |
| A linear L12 | 1 | 0.829 | 0.883 | 0.909 | 0.906 | 0.863 | 0.868 | 0.899 | 0.916 | 0.887 | 0.647 |
| **B MLP layers** | 0 | 0.871 | 0.887 | 0.882 | 0.916 | 0.874 | 0.949 | 0.912 | 0.924 | 0.913 | 0.637 |
| **B MLP layers** | 1 | 0.866 | 0.873 | 0.884 | 0.922 | 0.867 | 0.946 | 0.911 | 0.918 | 0.910 | 0.657 |
| C MLP layers + desc | 0 | 0.858 | 0.880 | 0.877 | 0.911 | 0.869 | 0.944 | 0.915 | 0.914 | 0.907 | 0.623 |
| C MLP layers + desc | 1 | 0.881 | 0.877 | 0.889 | 0.920 | 0.869 | 0.940 | 0.911 | 0.920 | 0.911 | 0.652 |

Size and latency of the head only (the DINOv2 pass is already run in production). A: 154k parameters, 0.3 MB fp16, about
0.006 ms. B: 707k parameters, 1.4 MB fp16 (`dist/c09/style_head.npz`, 1.47 MB with normalisation stats and names), about
0.03 ms. C: 773k parameters, 1.5 MB, about 0.03 ms. Latency is a CPU numpy forward pass for one image, batch 1. One seed-0
reading for C, 2.8 ms, was a contention outlier.

## Matching on TEST queries (selected predictor B)

Variants: reference = prod hybrid [1.0, 0.5, 0.5] over prod, axes17, film205. Oracle = real annotations for bank and
query. Predicted = predicted annotations for both. Asymmetric = real annotations for the bank and predicted annotations
for the query. Each variant adds w x cosine of the annotation vectors, centred on the real-annotation bank mean. The
differences d are paired against the reference. CIs are 95% bootstrap intervals over leakage groups. Pooled =
both seeds, with clusters = (seed, group).

**All works**

| Variant | seed | w | top-1 | top-3 | MRR | d top-1 [95% CI] | d top-3 [95% CI] | d MRR [95% CI] |
|---|---|---|---|---|---|---|---|---|
| reference | 0 | - | 0.636 | 0.790 | 0.736 | | | |
| reference | 1 | - | 0.578 | 0.780 | 0.703 | | | |
| oracle | 0 | 1.0 | 0.629 | 0.816 | 0.739 | -0.008 [-0.037, +0.023] | +0.026 [+0.003, +0.051] | +0.003 [-0.015, +0.024] |
| oracle | 1 | 1.0 | 0.613 | 0.804 | 0.729 | +0.034 [+0.008, +0.061] | +0.024 [+0.001, +0.046] | +0.026 [+0.009, +0.044] |
| oracle | pooled | | | | | +0.013 [-0.006, +0.034] | +0.025 [+0.009, +0.042] | +0.015 [+0.002, +0.028] |
| predicted | 0 | 1.0 | 0.602 | 0.801 | 0.721 | -0.035 [-0.066, -0.002] | +0.010 [-0.013, +0.035] | -0.015 [-0.035, +0.006] |
| predicted | 1 | 1.0 | 0.582 | 0.792 | 0.706 | +0.004 [-0.024, +0.030] | +0.013 [-0.016, +0.040] | +0.003 [-0.016, +0.021] |
| predicted | pooled | | | | | -0.015 [-0.036, +0.005] | +0.011 [-0.007, +0.030] | -0.006 [-0.020, +0.007] |
| asymmetric | 0 | 1.0 | 0.603 | 0.778 | 0.717 | -0.033 [-0.062, -0.004] | -0.013 [-0.036, +0.013] | -0.019 [-0.038, +0.001] |
| asymmetric | 1 | 1.0 | 0.576 | 0.790 | 0.705 | -0.003 [-0.031, +0.025] | +0.010 [-0.016, +0.037] | +0.002 [-0.018, +0.020] |
| asymmetric | pooled | | | | | -0.018 [-0.038, +0.001] | -0.001 [-0.019, +0.017] | -0.008 [-0.022, +0.004] |

**Cross-subject** (bank works that share the query's subject are removed)

| Variant | seed | w | top-1 | top-3 | MRR | d top-1 [95% CI] | d top-3 [95% CI] | d MRR [95% CI] |
|---|---|---|---|---|---|---|---|---|
| reference | 0 | - | 0.528 | 0.730 | 0.654 | | | |
| reference | 1 | - | 0.468 | 0.699 | 0.618 | | | |
| oracle | 0 | 1.0 | 0.508 | 0.749 | 0.651 | -0.021 [-0.049, +0.007] | +0.019 [-0.005, +0.045] | -0.003 [-0.022, +0.015] |
| oracle | 1 | 1.0 | 0.503 | 0.732 | 0.646 | +0.036 [+0.010, +0.063] | +0.033 [+0.005, +0.063] | +0.028 [+0.010, +0.047] |
| oracle | pooled | | | | | +0.008 [-0.011, +0.027] | +0.026 [+0.007, +0.046] | +0.012 [-0.001, +0.025] |
| predicted | 0 | 1.0 | 0.514 | 0.728 | 0.652 | -0.014 [-0.043, +0.014] | -0.003 [-0.029, +0.024] | -0.003 [-0.022, +0.017] |
| predicted | 1 | 1.0 | 0.470 | 0.730 | 0.630 | +0.003 [-0.027, +0.032] | +0.031 [+0.003, +0.060] | +0.012 [-0.009, +0.032] |
| predicted | pooled | | | | | -0.006 [-0.025, +0.015] | +0.014 [-0.005, +0.033] | +0.004 [-0.009, +0.018] |
| asymmetric | 0 | 1.0 | 0.497 | 0.719 | 0.638 | -0.031 [-0.059, -0.004] | -0.012 [-0.038, +0.015] | -0.016 [-0.035, +0.002] |
| asymmetric | 1 | 1.0 | 0.478 | 0.727 | 0.630 | +0.010 [-0.017, +0.038] | +0.028 [+0.006, +0.051] | +0.012 [-0.006, +0.030] |
| asymmetric | pooled | | | | | -0.010 [-0.030, +0.011] | +0.008 [-0.010, +0.026] | -0.002 [-0.015, +0.011] |

On validation, every variant looked strong. Cross-subject top-1 gained +0.06 to +0.10 over the reference (seed 0: reference
0.467, predicted 0.563, asymmetric 0.541, oracle 0.548), and w = 1.0, the top of the grid, was chosen almost everywhere.
None of that carried over to test. Sanity check, not used for any selection: at w = 0.5 the oracle reproduces the C18
numbers exactly (seed 0: 0.638 / 0.536; seed 1: 0.610 / 0.507), so the pipeline is the same. The C18 oracle gain is
itself small and depends on the weight. At w = 1.0, seed 0 test is already below the reference.

## Verdict: the gate fails

The gate requires higher held-out top-1 on both seeds, cross-subject accuracy preserved or improved, and a pooled paired
CI that excludes zero.

- **Asymmetric** (real bank annotations, predicted queries): seed 0 is significantly worse on top-1 (-0.033 all works,
  -0.031 cross-subject; both CIs exclude zero on the wrong side). Seed 1 is flat or slightly positive. Pooled top-1:
  -0.018 [-0.038, +0.001] all works, -0.010 [-0.030, +0.011] cross-subject. **Fails.**
- **Predicted** (both sides predicted): seed 0 top-1 -0.035 (CI excludes zero, negative). Pooled -0.015 / -0.006.
  **Fails.**
- Even the **oracle** fails this gate as fused here: seed 0 top-1 is -0.008 / -0.021, and the pooled top-1 CIs include
  zero. Its only robust gain is top-3 (+0.025 pooled, CI excludes zero). The "+0.01 to +0.04" annotation gain is therefore
  not a stable top-1 effect to recover in the first place. It depends on the seed and the weight, and a validation-chosen
  weight did not transfer.

The head predicts the annotation reasonably well: AUROC 0.91 overall, 0.95 on palette and tone, 0.87 on film rendering and
mood, and Spearman 0.64 on the narrative axes. The palette and tone information is largely what the hybrid's measured
descriptors already carry, though, and the predicted vector adds nothing reliable to the top-1 match. Recommendation: do
not integrate. `dist/c09/style_head.npz` (B, seed-0 bank split, float16, 1.47 MB, with input mean and std, target bank
mean, feature names and softmax groups) is kept as a research artefact only.
