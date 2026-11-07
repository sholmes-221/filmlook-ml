# Content-invariance benchmark for style embeddings (benchmark v1)

Purpose: evidence that an embedding captures photographic *treatment* rather than scene content, generator
fingerprints or dataset shortcuts. Implemented by `scripts/build_bench_features.py` (one GPU pass that caches every
feature) and `scripts/style_benchmark.py --split val|test` (CPU). Results: `reports/style_benchmark_<split>.json`.

## Protocol
- **Sources:** the 100 shared source photographs (CommonCatalog CC-BY / PD12M), uploader-grouped split 70/15/15.
  Every synthetic derivative inherits its source's split; nothing crosses.
- **Selection** uses the **val** split only (15 sources × all generators, plus CommonCatalog val originals for the
  semantic probe). **Test** (15 sources + BlueNeg) is run once per retained candidate for the final table and never
  for tuning. Earlier sessions used BlueNeg for selection; from benchmark v1 onward it is final-only.
- **Images** are standardised identically (384 px short side, seeded JPEG q80-95, no metadata) before any feature.
- **Uncertainty:** source-level bootstrap (val/test sources) for A, B, C, E; roll-level bootstrap for BlueNeg.

## Tests
| test | question | construction | metric (chance) |
|---|---|---|---|
| A. same image, different treatments | are treatment changes larger than nuisance changes? | for each source and each pair of its filmsynth v2 variants: d(variant i, variant j) vs d(variant i, nuisance copy of variant i) where the nuisance copy is JPEG q55 + 0.5× rescale | P(d_treat > d_nuisance) (0.5); median ratio |
| A2. style-gap ratio | does treatment move the embedding more than scene change? | d(same source, different family) / d(different source, same family) | ratio (>1 = style-dominated) |
| B. different images, same treatment | does the embedding retrieve the same family across scenes? | query each val variant; bank = variants of the other val sources; precision@5 and mAP of same family; also same-treatment within SDXL and within Z-Image | p@5 (empirical prior ≈ 0.12-0.33) |
| C. generator fingerprint | is the generator recoverable? | linear probe filmsynth / SDXL / Z-Image / controls, trained on train split; and "same-generator share" of the top-5 when querying treated images against a bank of all generators' treated images (other sources) | accuracy (0.44); share (prior) |
| D. unseen real photographs (test only) | does similarity hold on real scans? | BlueNeg blue-intact previews, same-stock p@5 with the query's roll excluded (owner-recorded stock labels) | p@5 (0.12), roll CI |
| E. semantic leakage | how much content survives? | (i) own-source rank-1 of treated variants against raw sources (0 = none); (ii) linear probe of the CommonCatalog `subjects` tag from the embedding vs DINOv2-global (content ceiling) and majority; (iii) overlap of top-10 neighbours with DINOv2-global neighbours on real originals | rank-1; accuracy; Jaccard |

## What the benchmark cannot show
- No human judged any aesthetic. Families are engine presets; stock labels are one photographer's records.
- "Same treatment across scenes" is defined by the generator's parameters, not perception; two variants of one
  family can differ visibly, and two families can overlap.
- Real-film generalisation (D) rests on BlueNeg alone (27 rolls); CI widths of about ±0.08 are the floor.
- Diffusion treatments carry no parameters, so B for SDXL/Z-Image uses treatment ids (4 each).

## Candidate embeddings in v1
raw DINOv2 global (768), raw hybrid (explicit 28 ++ DINOv2 L4+L8, 1564), hybrid + family SupCon projection (64,
trained on train sources), `student_style_fs_v2` 576-d and 64-d (ONNX), colour/tone statistics (16), texture
statistics (12). Features are cached in `data/embeddings/bench_features_v1.npz` with the image list in
`data/embeddings/bench_features_v1.json`.
