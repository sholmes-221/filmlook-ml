# C19b: confirmatory test of SigLIP-B/16 int8 in the photographer-style hybrid

Follows C19 (`encoder_hybrid.md`), where SigLIP met every criterion only as an exploratory reading.

## Pre-registration (committed before any number of this study was read)

**Candidate (the only one):** `add_siglip_base_onnx_int8`, i.e. shipped hybrid (prod 1.0 + axes17 0.5 + film205
0.5) plus the SigLIP-B/16 (`google/siglip-base-patch16-224`, Apache-2.0) vectors produced by the **int8 ONNX model**
(dynamic, MatMul only, per-channel; `checkpoints/c19/siglip_base_int8.onnx`, the same graph as the memory-mapped
`siglip_base_int8_ext.onnx`, outputs bit-identical), with the website-style numpy/PIL preprocessing
(`data/embeddings/photographers/siglip_base_onnx_int8.npz`, built in C19 before this study).

**Weight:** w in {0.5, 1, 1.5, 2, 2.5, 3, 4} on SigLIP, chosen per seed on VAL cross-subject top-1 (ties -> smaller w).

**Seeds:** 2 and 3 (not used by subject_audit / C09 / C19; seed 2 was used once by the prod-only `evaluate.py`
benchmark). `split_groups` assignments differ: between any two of seeds 0-3, 44-45 % of rows keep their split and
test sets overlap at the chance level (162-206 of ~860), checked before this commit.

**Protocol:** unchanged from C19 (`encoder_hybrid.py`; `run_match`, `paired_ci` from C09; 27 photographers;
top-3 scorer; all-works and cross-subject protocols; TEST read once per seed; pooled over seeds with bootstrap
clusters (seed, leakage group)). Top-3 and MRR are reported too.

```bash
python -m research.photographers.encoder_hybrid --encoders siglip_base_onnx_int8 --variants add_siglip_base_onnx_int8 \
    --seeds 2 3 --w-grid 0.5 1 1.5 2 2.5 3 4 --out reports/photographers/encoder_confirm.json
```

**Production memory:** `prod_memory.py`, run with the website's own virtual environment, imports the website's
analysis path read-only (prepare_image, photo gate judge, local vision signals, `DnaService.analyze` with measured
matching + reference retrieval, `LocalFilmLookAnalyzer.analyze` with `local_report` and the film head), one DNA and
one film analysis of `tests/fixtures/cc0_portrait_curly_man.jpg`, ORT and OMP single-threaded, in a fresh process;
peak RSS = `ru_maxrss`. Three fresh processes without and three with the SigLIP int8 mmap session loaded and run on
the same prepared JPEG. The reading used for the gate is the **largest** of the three peaks with SigLIP.

### Gate (both must hold)

1. Pooled TEST top-1 gain over the shipped hybrid, 95 % interval above 0, in BOTH protocols, with the int8 vectors.
2. Peak RSS of the production path with SigLIP <= 520 MiB (container limit 640 MiB; ~120 MiB margin).

Licence (Apache-2.0) was settled in C19. Latency is reported, not gated (C19: 0.12-0.13 s single-thread here).
Also reported, not gated: what building the production reference index with SigLIP needs (works, time here).
If both hold: write the integration steps (website files) without doing them. If either fails: do not integrate.

## Results (after the pre-registration commit 65b1a4c)

Accuracy (`reports/photographers/encoder_confirm.json`). VAL chose w = 4.0 on both seeds, the top of the
grid again (seed 2 VAL cross top-1 0.453 -> 0.521; seed 3 0.482 -> 0.554). TEST, pooled over seeds 2 and 3
(735 + 777 queries), paired against the shipped hybrid, 95 % bootstrap over (seed, leakage group):

| protocol | reference top-1 / top-3 / MRR | +SigLIP int8 top-1 | Δ top-1 [95 % CI] | Δ top-3 [95 % CI] | Δ MRR [95 % CI] |
|---|---|---|---|---|---|
| all works | 0.612 / 0.801 / 0.725 | 0.663 | +0.050 [0.028, 0.073] | +0.028 [0.012, 0.044] | +0.039 [0.025, 0.054] |
| cross-subject | 0.515 / 0.740 / 0.651 | 0.568 | +0.053 [0.031, 0.075] | +0.020 [0.000, 0.039] | +0.040 [0.025, 0.054] |

Production-path memory (`prod_memory.py`; website venv, onnxruntime 1.31.1, OMP/ORT 1 thread; three fresh
processes each; `reports/photographers/prod_memory_{base,siglip}_{1,2,3}.json`). Every run: photo gate accept,
DNA ok with reference reading `match`, film report score 95.

| | RSS after imports MiB | peak RSS MiB (3 runs) | RSS at end MiB |
|---|---|---|---|
| current path | 75 | 368.7, 367.6, 371.2 | 350-353 |
| + SigLIP int8 mmap session, run | 75 | 447.1, 453.9, 453.6 | 438-444 |
| difference | | +78 to +86 (max vs max: +82.7) | +88 to +91 |

The gate reading is the largest peak with SigLIP: **453.9 MiB <= 520 MiB**. Not in this measurement, and to be added
by anyone sizing the container: (a) the web server's own baseline (production idle 110 MiB vs 75 MiB after these
imports, about +35 MiB); (b) the SigLIP reference vectors held as float32 (4,249 x 768 x 4 B = 12.5 MiB; 6.2 MiB if
kept float16). With both, about 500 MiB, still under 520 but with 20 MiB of slack, not 66. Memory-mapped weights
are file-backed and reclaimable: under pressure they are evicted and re-read (slower), not an OOM.

Reference index with SigLIP: the shipped index holds 4,249 works (4,543 exported, 294 dropped by the vendoring
screen); the bank vectors here (`siglip_base_onnx_int8.npz`, 4,548 works) already cover it. Re-embedding from scratch
with the int8 model on this machine: 0.126 s/work at 1 thread (about 9.5 min for 4,543 works), 0.063 s/work at
4 threads (about 4.8 min). Output adds `siglip.f16.npy` (4,249 x 768, 6.2 MiB).

## Gate and decision

| criterion | reading | |
|---|---|---|
| 1 pooled top-1 gain CI > 0, both protocols, int8 vectors | +0.050 [0.028, 0.073]; +0.053 [0.031, 0.075] | PASS |
| 2 peak RSS with SigLIP <= 520 MiB | 453.9 MiB (max of 3) | PASS |

**Decision: the gate passes.** Integration is recommended, not done. Caveats: w = 4.0 is the top of the grid (the
encoder weight is not settled, and the shipped fusion is z-scored, so the weight must be re-chosen in its own
scale, see below); the memory slack after the two additions above is about 20 MiB; latency on the E2.1.Micro is not
measured (0.12-0.13 s here, roughly as costly as the existing DINOv2-small pass).

## Integration steps for the website (not done)

1. Model file: ship `siglip_base_int8_ext.onnx` + `siglip_base_int8_ext.weights` (97 MiB, 64 KB-aligned external
   weights; the `.weights` file must sit beside the `.onnx`, the read-only container is fine for mmap) in
   `models/photo_gate/` or a new `models/style_encoder/`, with its sha256 and the Apache-2.0 notice.
2. `filmlook/photo_gate.py`: a second lazily created session (intra/inter 1 thread, `enable_cpu_mem_arena=False`,
   `enable_mem_pattern=False`, `session.disable_prepacking=1`), a `siglip_pixels` function (RGB, resize 224 x 224
   bicubic, no crop, (x/255 - 0.5)/0.5; must match `research/photographers/encoder_cost.pixels`, pin it with a test
   like the DINOv2 one) and `style_embedding(jpeg)` cached by content like `_outputs`, so one SigLIP pass per upload.
   Load it only when the reference index declares a SigLIP space, and under the same lock as the DINOv2 pass.
3. `filmlook/photographer_dna/reference_match.py`: `Index` gains `siglip` vectors (load `siglip.f16.npy`, check
   length, L2-normalise); `for_jpeg` computes `gate.style_embedding(jpeg)` when present; in `match`, add a `siglip`
   term to the fused score (top-m mean per photographer, z-scaled by `calibration.fusion.scale.siglip`, weight
   `fusion.weights.siglip`). Retrieval/abstention thresholds stay on the DINOv2 score unless re-validated. Missing
   SigLIP file or failed pass -> today's fusion, never a guess.
4. Reference index build (filmlook-ml): `research/photographers/export.py` writes `siglip.f16.npy` from
   `siglip_base_onnx_int8.npz` (the int8 ONNX vectors, not the PyTorch ones), records the encoder sha256 and the
   preprocessing in `manifest.json`, and chooses the siglip fusion weight on VAL in the export's own z-scored fusion
   (grid extending above 4); then film-evaluation `tools/vendor_refmatch.py` copies and filters it with the others,
   and `refresh.sh` re-embeds new works with `encoder_cost embed --model siglip_base --variant int8`.
5. Tests: index-load and missing-file tests, a pixel-parity test, and an end-to-end check that `match` with the
   shipped index reproduces the research ranking on a few bank works. Measure the container's real peak on the
   E2.1.Micro after deploy (`docker stats`) before turning it on for everyone.
