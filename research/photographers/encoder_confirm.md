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
