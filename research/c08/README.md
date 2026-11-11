# C08: film-look matching research (2026-11-11)

Results and interpretation: website repository `docs/FILM_RECOGNITION_BENCHMARK.md` §9. Reports land in
`reports/c08/` and data in `data/c08/` (both git-ignored).

| Step | Command | Output |
|---|---|---|
| Measure every LUT on standard patches | `python -m research.c08.lutfeat` | `data/c08/lut_features.json`, `lut_pairs.json`, `lut_distance.npy` |
| Datasheets: extract (Sonnet, budget-guarded, cached) and verify quotes | `python -m research.c08.claude_research datasheets --no-dry-run --confirm-paid`; `python -m research.c08.datasheet_kb` | `data/c08/film_datasheets.json` |
| Controlled pairs (features only) | `OMP_NUM_THREADS=1 python -m research.c08.pairs build`; `... pairs extra` | `data/c08/pairs/` |
| Difference embeddings / single-image presets / stocks + real scans | `python -m research.c08.experiments delta|presets|stocks` | `reports/c08/{delta,presets,stocks}.json` |
| Live engine on the same renders | `python -m research.c08.engine_baseline` | `reports/c08/engine_baseline.json` |
| Encoders for film vs digital, controls, probes | `python -m research.c08.encoders embed|evaluate|controls|probes` | `reports/c08/encoders*.json` |
| Production-faithful inputs, head v2, ONNX re-export | `python -m research.c08.prod_inputs`; `python -m research.c08.film_head_v2`; `python -m research.c08.export_layers_onnx` | `dist/film_head/film_head_v2layers.npz`, `checkpoints/c08/*.onnx` |
| Claude hypotheses on measured signatures | `python -m research.c08.claude_research lutsig --no-dry-run --confirm-paid` | `data/c08/claude/lutsig/`, `reports/c08/lutsig_tests.json` |

Pitfalls found: DINOv2 in fp16 overflows (cosine 0.95 to production); the production ONNX L2-normalises CLS and mean
separately; `split_of` by group put all of iNaturalist into one split; numpy threads must be pinned to 1 per worker.
