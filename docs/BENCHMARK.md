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

## Results (benchmark v1, 2026-11-07)

Val (selection; `reports/style_benchmark_val.json`, `_val_e19.json`) and the one-time test run
(`reports/style_benchmark_test.json`, registry `B1_bench_test_final`). B/C/E chance rates: B filmsynth 0.11, B
sdxl/zimage 0.33, C accuracy 0.44, C share prior 0.40, E subject majority 0.20 (val) / 0.22 (test), E content
ceiling (DINOv2 global) 0.53 (val) / 0.48 (test). D chance 0.12.

| candidate (test split) | A p | A2 | B fs p@5 [CI] | B sdxl | B zim | C acc | C share | E rank-1 | E subj | E jac | D p@5 [roll CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw DINOv2 global (reference) | 0.82 | 0.06 | 0.13 [0.12, 0.14] | 0.36 | 0.37 | 0.47 | 0.44 | 0.99 | 0.48 | 1.00 | 0.17 [0.13, 0.23] |
| raw hybrid | 0.49 | 0.19 | 0.21 [0.17, 0.25] | 0.34 | 0.34 | 0.74 | 0.40 | 0.37 | 0.49 | 0.09 | 0.15 [0.08, 0.19] |
| hybrid + family SupCon | 0.93 | 1.51 | 0.44 [0.39, 0.48] | 0.32 | 0.35 | 0.56 | 0.59 | 0.33 | 0.34 | 0.09 | **0.25 [0.16, 0.31]** |
| student_fs_v2 64-d | 0.90 | 3.33 | **0.55 [0.49, 0.62]** | 0.35 | 0.35 | 0.55 | 0.63 | **0.16** | 0.33 | 0.09 | 0.18 [0.13, 0.23] |
| colour/tone 16 stats | 1.00 | 1.91 | 0.34 [0.30, 0.38] | 0.35 | 0.35 | 0.48 | 0.51 | 0.52 | **0.25** | 0.08 | 0.18 [0.09, 0.28] |

Val numbers for the same candidates are within ±0.05 of these on every column (hybrid SupCon val: A 0.91, A2 1.71,
B fs 0.47, C share 0.54, rank-1 0.37, subj 0.24), so nothing was selected on noise. Readings:
- Raw DINOv2 global is a content embedding (A2 0.06, rank-1 0.99, Jaccard 1.0, subject probe at the ceiling).
  Every retained candidate is style-dominated on filmsynth (A2 > 1, B fs 3-5× chance, Jaccard < 0.1).
- The learned candidates still carry content and generator signal: own-source rank-1 0.16-0.37 (student lowest),
  subject probe 0.33-0.34 against a 0.22 majority, same-generator share 0.59-0.63 against a 0.40 prior. The
  16 colour/tone statistics carry the least semantic content (0.25) but the most own-source identity (0.52).
- B for SDXL and Z-Image is at chance for every embedding, including the ones trained on those labels (E19). The
  diagnostic `reports/diffusion_treatment_distinctness.json` explains why: two SDXL treatments of the same source
  differ by 0.19× the JPEG+rescale nuisance under raw hybrid features (Z-Image 0.35×; filmsynth 1.6×), so at strength
  0.35 the diffusion treatment ids are not distinct labels. E10 re-generates SDXL at strength 0.5 to test that.
- D (real scans): only hybrid + family SupCon is clearly above chance, 0.25 [0.16, 0.31]; the student (0.18) and the
  colour/tone statistics (0.18) are not separable from the DINOv2 reference (0.17) at 27 rolls.

## Confidence and abstention (`scripts/abstention_analysis.py`, `reports/abstention_test.json`)
Every retrieval query gets a label-free confidence: `top1_sim` (cosine similarity to the nearest neighbour from
another source/roll) or `knn_agree` (share of the 5 nearest index items that agree with the top-1's label).
Rank-1 precision when answering only the most confident fraction of queries (test split; chance 0.125 for B,
0.10 for D):

| candidate | task | 100 % | 75 % | 50 % | 25 % | AURC (unordered) |
|---|---|---|---|---|---|---|
| hybrid + family SupCon | B filmsynth, top1_sim | 0.60 | 0.73 | 0.83 | 1.00 | 0.18 (0.40) |
| hybrid + family SupCon | D BlueNeg, top1_sim | 0.22 | 0.26 | 0.37 | 0.50 | 0.59 (0.78) |
| student_fs_v2 64-d | B filmsynth, knn_agree | 0.59 | 0.72 | 0.75 | 0.92 | 0.19 (0.41) |
| student_fs_v2 64-d | D BlueNeg, top1_sim | 0.15 | 0.16 | 0.20 | 0.27 | 0.81 (0.85) |
| colour/tone 16 | D BlueNeg, knn_agree | 0.20 | 0.22 | 0.27 | 0.40 | 0.72 (0.80) |

`top1_sim` orders the hybrid SupCon errors well on both synthetic and real data: at 50 % coverage same-stock rank-1
precision rises from 0.22 to 0.37 (0.50 at 25 %, 48 queries, so ±0.14). The student's confidence is informative on
synthetic data only; its real-scan ordering is close to random. Deployment rule for the research embedding:
abstain from "similar film look" suggestions below the top1_sim value that gives 50 % coverage on the benchmark,
and report the coverage, not a probability. No score here is a calibrated probability of a correct match.

## Real-film evaluation on a second source (session 3): `scripts/eval_commons_film.py`
Commons film set (`docs/FILM_DATASET_RESEARCH.md`, session 3): 5.6k stock-labelled scans from 401 uploaders, plus
111 digital film-emulation controls. Uploaders are hashed into an analysis half (selection, diagnostics) and a test
half (final only). Metrics, each with the query's uploader excluded from the bank and an uploader bootstrap:
same-stock p@5 (chance = bank prior), same-film-type p@5, same-uploader share when not excluded (photographer bias),
uploader-grouped scanner probe (scanner effect), and real-film-vs-emulation AUROC for the authenticity head and for
grouped linear probes on texture / colour-tone / hybrid features (which axis separates real film from a software
film look). Labels are uploader-asserted; the numbers measure consistency with those claims, not ground truth.

## Pre-registered rules for the evaluation pipeline (session 3)
`research/eval_pipeline.py` applies these to every candidate against `hybrid_supcon_family` on a ready real-film
dataset (Commons analysis half; the test half once, with `--final`):
- primary: same-stock precision@5 with the query's uploader excluded, bootstrap CI over uploaders; `improved` needs
  delta >= 0.02 and non-overlapping CIs, `regressed` symmetrically, otherwise `inconclusive`;
- leakage guards: uploader-share excess over its prior may not exceed the baseline's by more than 0.02; the scanner
  probe accuracy (where scanner categories exist) may not exceed the baseline's by more than 0.05; a guard failure is
  `reject: leakage` whatever the primary metric;
- a candidate is never promoted on synthetic metrics alone, and test-half results are never used for selection.

### Note added 2026-11-08 (after the first commoncatalog_film evaluation)
The decision metric stays same-stock p@5 on the uploader-claimed label as recorded (speed-level when the uploader gave one). On the first snapshot every candidate, baseline included, was at chance on it (54 stocks, 20 with >= 2 uploaders). `same_stock_line` (brand + line, speed dropped) is now reported as a *secondary* reading and `same_type` as a sanity check; neither replaces the decision rule, and no threshold was changed. Larger snapshots are acquired to raise uploaders per stock, not to change the rule.

### Note added 2026-11-08 (session 5, after E32-E37): the real-photograph matrix
The decision rule above is unchanged. Session 5 showed that the pre-registered leakage guards (uploader share, artist
share) move together with every informative embedding on the frozen DINOv2-Small statistics (claims 23-26 in
`docs/FINDINGS.md`), so a candidate is now also *described* by the real-photograph matrix of
`scripts/student_real_matrix.py` (analysis halves only, group bootstrap CIs):
museum process p@5 with stereo cards removed and the decade matched (format control), artist share, stock-line p@5 with
the uploader excluded, uploader share, scanner probe, content- and aspect-controlled same-uploader retrieval, within-
uploader different-day stock AUC (and its colour-only reading) against a within-uploader permutation null, synthetic
family p@5 on the filmsynth validation split (the cost of a real-data gain), and the effective rank of the embedding
(collapse diagnostic; the current students are at 2-3 of 64). A candidate is judged on pre-registered thresholds over
this matrix (see E38 in `research/backlog.yaml`); the test halves and BlueNeg stay reserved for one final evaluation.
