# C19: a stronger image encoder in the photographer-style hybrid?

Question: can photographer-style matching improve by swapping or adding an image encoder, within the production
limits (Oracle E2.1.Micro, 2 weak vCPUs, 1 GB RAM, app container 640 MB that already holds DINOv2-small and the
other models)?

## Pre-registration (written and committed before any TEST number of this study was read)

Protocol: identical to `subject_audit.py` / `research/c09/local_predictors.py`: same rows, leakage groups, 60/20/20
split by group (seeds 0 and 1), 27 photographers, works with a subject label, scorer = mean of the 3 closest bank
works per photographer over a weighted sum of cosine similarities (`run_match`), cross-subject protocol (bank works
with the query's subject removed). Reference = the shipped hybrid: prod 1.0 + axes17 0.5 + film205 0.5.
Encoder vectors reused from `data/embeddings/photographers/` (`clip_b32`, `siglip_base`, `dinov2_base`; nothing
recomputed for the accuracy study).

Variants (weights chosen on VAL cross-subject top-1, ties to the weight closest to the shipped value, TEST read once
per seed):

| variant | parts | grid |
|---|---|---|
| `retuned_ref` (control) | prod 1, axes m, film m | m in {0.25, 0.5, 0.75, 1, 1.5} |
| `swap_<E>` | E 1, axes m, film m | m in {0.25, 0.5, 0.75, 1, 1.5} |
| `add_<E>` | prod 1, E w, axes 0.5, film 0.5 | w in {0.25, 0.5, 0.75, 1, 1.5, 2} |

E in {clip_b32, siglip_base, dinov2_base}. Readings: top-1, top-3, MRR, both protocols (all works, cross-subject),
paired difference against the reference with a 95 % bootstrap interval over leakage groups, pooled over the two seeds
(clusters = (seed, group)), as in `local_predictors.pool`.

Primary candidate: the variant (excluding the control) with the highest mean VAL cross-subject top-1 over the two
seeds. The decision is about the primary candidate; other variants are exploratory (seven comparisons, no
multiplicity correction). Cost is measured for the primary candidate and the best variant of a different encoder.

### Deployment gate (all four must hold for the primary candidate)

1. Pooled TEST top-1 gain over the shipped hybrid with the 95 % interval above 0, in BOTH protocols (all works and
   cross-subject). If the int8 model is the one that would ship, the gain must hold with the int8 vectors.
2. Extra memory <= 150 MB (peak RSS increase of a fresh process loading the ONNX model and embedding one image,
   CPU, 1 thread; the shipped model's file size is reported beside it).
3. Extra latency <= 0.5 s per image on this development machine, onnxruntime intra_op_num_threads=1, production
   input size. (This machine is much faster than the E2.1.Micro; the raw number is reported as is.)
4. The weights' licence allows commercial use (model card cited).
