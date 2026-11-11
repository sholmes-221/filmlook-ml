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

## Commands

```bash
python scripts/preflight_check.py                                        # PASS
python -m research.photographers.encoder_hybrid                          # reports/photographers/encoder_hybrid.json
python -m research.photographers.encoder_cost export --model clip_b32    # also siglip_base; parity json
python -m research.photographers.encoder_cost embed --model clip_b32 --variant int8
python -m research.photographers.encoder_hybrid --encoders clip_b32_onnx_int8 siglip_base_onnx_int8 \
    --out reports/photographers/encoder_hybrid_int8.json
python -m research.photographers.encoder_cost ext --model clip_b32 --variant int8   # mmap-able external weights
OMP_NUM_THREADS=1 python -m research.photographers.encoder_cost measure --model clip_b32 --variant int8 --format ext --lean
```

## Results (all after the pre-registration commit)

Validation (cross-subject top-1, mean of the two seeds; reference 0.462): add_clip_b32 0.576, swap_clip_b32 0.536,
add_siglip_base 0.534, add_dinov2_base 0.522, swap_dinov2_base 0.513, swap_siglip_base 0.503, retuned control 0.493.
**Primary candidate: `add_clip_b32`.** Chosen weights were identical on both seeds: add_* put weight 2.0 on the new
encoder (clip, siglip; 1.5 for dinov2_base), i.e. the edge of the grid, so a larger weight might do slightly better;
swap_* and the control chose measured weight 0.25.

TEST, pooled over seeds 0 and 1 (n = 1,556 query readings), paired difference against the shipped hybrid, 95 %
bootstrap interval over (seed, leakage group). Reference: full top-1 0.607 / top-3 0.785 / MRR 0.719; cross-subject
0.498 / 0.715 / 0.636.

| variant | full top-1 | Δ top-1 [95 % CI] | Δ top-3 | Δ MRR | cross top-1 | Δ top-1 [95 % CI] | Δ top-3 | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| retuned control | 0.608 | +0.001 [-0.014, 0.016] | +0.012 | +0.003 | 0.479 | -0.019 [-0.035, -0.003] | +0.010 | -0.006 |
| **add CLIP B/32** (primary) | 0.647 | +0.040 [0.022, 0.060] | +0.040 | +0.038 | 0.549 | +0.051 [0.033, 0.071] | +0.056 | +0.049 |
| swap CLIP B/32 | 0.647 | +0.040 [0.021, 0.058] | +0.027 | +0.032 | 0.551 | +0.054 [0.033, 0.075] | +0.044 | +0.043 |
| add SigLIP-B/16 | 0.636 | +0.029 [0.013, 0.046] | +0.028 | +0.026 | 0.544 | +0.046 [0.028, 0.064] | +0.041 | +0.039 |
| swap SigLIP-B/16 | 0.637 | +0.029 [0.012, 0.047] | +0.018 | +0.023 | 0.543 | +0.045 [0.027, 0.065] | +0.036 | +0.036 |
| add DINOv2-base | 0.623 | +0.016 [-0.005, 0.038] | +0.028 | +0.017 | 0.500 | +0.002 [-0.019, 0.022] | +0.016 | +0.008 |
| swap DINOv2-base | 0.628 | +0.021 [0.001, 0.040] | +0.022 | +0.021 | 0.518 | +0.020 [0.001, 0.039] | +0.030 | +0.021 |

Same benchmark with the vectors of the int8 ONNX models (website preprocessing re-implemented in numpy/PIL; weights
re-chosen on VAL, same choices):

| variant (int8) | full Δ top-1 [95 % CI] | Δ top-3 | Δ MRR | cross Δ top-1 [95 % CI] | Δ top-3 | Δ MRR |
|---|---|---|---|---|---|---|
| add CLIP B/32 | +0.036 [0.019, 0.055] | +0.035 | +0.035 | +0.047 [0.030, 0.065] | +0.057 | +0.046 |
| swap CLIP B/32 | +0.035 [0.016, 0.054] | +0.031 | +0.029 | +0.045 [0.026, 0.065] | +0.042 | +0.038 |
| add SigLIP-B/16 | +0.026 [0.011, 0.043] | +0.028 | +0.025 | +0.043 [0.026, 0.062] | +0.043 | +0.038 |
| swap SigLIP-B/16 | +0.029 [0.011, 0.048] | +0.013 | +0.022 | +0.049 [0.029, 0.068] | +0.031 | +0.036 |

The gain is the encoder, not re-tuning: re-tuning the shipped three-part hybrid on the same validation data gives
nothing (full) or loses (cross-subject).

### Export and parity (300 bank works, cosine with the stored PyTorch vectors, end to end incl. preprocessing)

| model | fp32 ONNX | int8 ONNX (dynamic, MatMul only, per-channel) |
|---|---|---|
| CLIP B/32 | mean 0.99999, min 0.9998 | mean 0.989, p01 0.972, min 0.947 |
| SigLIP-B/16 | mean 0.99999, min 0.9999 | mean 0.990, p01 0.968, min 0.958 |

Quantising the patch-embedding Conv as well (the onnxruntime default) gave mean 0.971 / 0.975, so it stays fp32.

### CPU cost (this machine: Intel i9-14900KF; 1 thread; 224 x 224; the E2.1.Micro is much slower, roughly several
times; these are raw development-machine numbers)

Peak extra RSS = peak resident memory of a fresh process (numpy, PIL, onnxruntime already imported) after loading
the model and embedding 31 images, minus the resident memory before loading. Latency = JPEG decode +
preprocessing + inference, median of 28 images (after 3 warm-ups). "ext" = the same graph with the weights in a
separate 64 KB-aligned file, which onnxruntime memory-maps (no protobuf copy); "lean" = no CPU arena, no memory
pattern, no weight pre-packing. Outputs are bit-identical between onnx and ext.

| model | file MB | load format | peak extra RSS MB | steady extra RSS MB | latency median s | p90 s |
|---|---|---|---|---|---|---|
| DINOv2-small (shipped, for scale) | 84 | onnx | 189-195 | 180-187 | 0.114 | 0.14-0.17 |
| DINOv2-small (shipped) | 84 | onnx lean | 158 | 149 | 0.117 | 0.126 |
| CLIP B/32 fp32 | 335 | onnx | 582 | 583 | 0.080 | 0.105 |
| CLIP B/32 int8 | 92 | onnx | 185-189 | 176-180 | 0.034-0.036 | 0.040 |
| CLIP B/32 int8 | 92 | onnx lean | 165 | 159 | 0.036-0.040 | 0.042-0.046 |
| **CLIP B/32 int8** | 92 | **ext** / ext lean | **137 / 134** | 128 / 126 | **0.035-0.043** | 0.041-0.043 |
| SigLIP-B/16 fp32 | 355 | onnx | 597-613 | 590-606 | 0.268 | 0.28-0.30 |
| SigLIP-B/16 int8 | 97 | onnx | 220-221 | 211-213 | 0.107-0.118 | 0.11-0.17 |
| SigLIP-B/16 int8 | 97 | onnx lean | 175 | 167 | 0.111-0.142 | 0.12-0.40 |
| **SigLIP-B/16 int8** | 97 | ext / **ext lean** | 158 / **145-146** | 149 / 137-140 | **0.119-0.131** | 0.16 |

(Ranges: repeated runs; another project was loading this machine's CPUs.) Memory-mapped weights are file-backed
pages: they count in RSS and in the container's cgroup, but are reclaimable under pressure, so the kernel can evict
and re-read them (slower, not an OOM). The CLIP int8 latency is lower than the shipped DINOv2-small (49 tokens vs
256); SigLIP-B/16 (196 tokens, base width) is about the same as the shipped encoder on this machine.

### Licences

- CLIP ViT-B/32 (`openai/clip-vit-base-patch32`): the code repository is MIT (github.com/openai/CLIP/LICENSE,
  "Copyright (c) 2021 OpenAI"); the Hugging Face card has no licence tag and states under Out-of-Scope Use Cases:
  "**Any** deployed use case of the model - whether commercial or not - is currently out of scope." and "The model
  is intended as a research output for research communities." A website feature is a deployed, commercial use.
- SigLIP base patch16-224 (`google/siglip-base-patch16-224`): Apache-2.0 (model card tag). Commercial use allowed.
- DINOv2-base (`facebook/dinov2-base`): Apache-2.0 (model card tag).

## Gate and decision

| criterion | add CLIP B/32 (primary) | add SigLIP-B/16 (exploratory, best other encoder) |
|---|---|---|
| 1 top-1 gain, CI > 0 both protocols (int8) | PASS: +0.036 [0.019, 0.055], +0.047 [0.030, 0.065] | pass: +0.026 [0.011, 0.043], +0.043 [0.026, 0.062] |
| 2 extra memory <= 150 MB | PASS only with mmapped external weights (134-137 MB); 165-189 MB as a plain ONNX | pass only with external weights + lean session (145-146 MB, 4 MB headroom); 158-221 MB otherwise |
| 3 extra latency <= 0.5 s (this machine) | PASS: 0.04 s | pass: 0.12-0.13 s |
| 4 licence allows commercial use | **FAIL**: weights released for research; the publisher puts every deployed use, commercial or not, out of scope (the MIT grant on the code does not remove that statement) | pass: Apache-2.0 |

**Decision: the pre-registered gate fails (primary candidate, criterion 4). Do not integrate.** A stronger encoder
does improve style matching (about +3 to +5 points top-1, both protocols, int8 included), and the cost is within the
limits once the weights are memory-mapped; the blocker for CLIP is the licence, which this study should have applied
when choosing the primary candidate. SigLIP-B/16 met all four criteria as an exploratory reading, with almost no
memory headroom; under the pre-registered rule it is not a deployment decision. The next step, if wanted, is a
confirmatory pre-registration with SigLIP as the only candidate (fresh seeds 2 and 3, the same gate, and a memory
measurement inside the 640 MB production container on the E2.1.Micro, where 146 MB extra on top of DINOv2-small and
the other models may not fit).

Limits: two seeds of one bank (27 photographers, archive-era public-domain work); the cross-subject protocol uses
Haiku subject labels; weights on the add_* grid edge; latency and memory measured on a fast desktop CPU, not the
production VM; seven variants without multiplicity correction (all CLIP and SigLIP intervals are well clear of 0).
