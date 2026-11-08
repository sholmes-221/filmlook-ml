# Model evaluation: baselines, methodology, results, limits, recommendation (2026-11-08)

## 1. The candidates

| | what it is | size, cost |
|---|---|---|
| **A. the trained student** | MobileNetV3-Small distilled from frozen DINOv2 (E52 recipe), pooled 576-d → PCA-32 int8; frozen hashes in `checkpoints/frozen_session7` | 4.4 MB ONNX, 32 bytes per image, 1.3-1.6 ms per image on one CPU thread here |
| **B. frozen DINOv2-Small** | CLS++mean (768-d) and L4+L8 statistics, PCA-64 fitted on the same 601 CC-BY originals; nearest-neighbour retrieval | 88 MB, 90 ms per image on one CPU thread, about 400 MB RSS: not deployable on the 1 GB instance |
| **B'. untrained ImageNet MobileNetV3-Small** | the same backbone without any FilmLook training, same PCA | identical cost to A: the control for "what did training buy" |
| **C. 28 handcrafted statistics** | tone, colour and texture measurements from `filmlook_infer`, standardised on the fit set | no model, about 100 ms per image on one core |
| **D. Gemini-based FilmLook analysis** | the production LLM pipeline (scores, matches, prose) | evaluated separately in section 5; no ground truth exists for it |

## 2. Methodology

- **Evaluation sets** (fresh, legally usable, never used for training or selection, read once under a
  pre-registered rule, E53 on 2026-11-08): The Met public-domain photographs (378 images of the analysis half,
  140 artists, CC0; process = museum catalogue record), a second CommonCatalog film-scan snapshot (324 scans from 91
  uploaders never seen before, CC-BY-2.0; stock = uploader's own tag), ARTeFACT (44 originals, one photographer,
  owner-recorded emulsions, CC BY 4.0). Spent sets (both halves of `aic_photographs` and `commoncatalog_film`,
  BlueNeg, the synthetic test split) are reported only where marked "spent".
- **Metrics**: precision@5 with the query's artist or uploader excluded and decade matched (museum), p@5 with the
  uploader excluded (stock line), within-uploader different-day AUC (does the embedding separate a photographer's
  stocks from each other), artist/uploader share of the top five (photographer leakage; lower is better), own-image
  rank-1 under controlled transformations (crop 80 %, flip, JPEG q50, 0.5x rescale, 70 % corner), ARTeFACT
  same-emulsion AUC and p@3. CIs are bootstraps over artists or uploaders; effect sizes are paired per query.
- **Ground truth**: catalogue records and uploader claims only. No Claude or Gemini output and no human label
  made by me is used anywhere as truth. Controlled transformations (E49, E52, E54) are stated as such.
- **Seeds**: three training seeds for A; one deterministic value for B, B', C.

## 3. Results on the fresh sets (E53; `reports/e53_fresh_holdout.json`)

| reading | A (3 seeds) | B' untrained | C statistics | B DINOv2 L4+L8 |
|---|---|---|---|---|
| Met process p@5, chance 0.648 | 0.694-0.700 (lower CIs 0.642-0.648) | 0.678 | **0.746** [0.697, 0.794] | 0.678 |
| Met artist share (lower is better) | **0.078-0.081** | 0.105 | 0.130 | 0.136 |
| stock-line p@5 across uploaders, chance 0.080 | 0.141-0.156 | 0.142 | 0.119 | 0.140 |
| uploader share (lower is better) | **0.128-0.144** | 0.138 | 0.181 | 0.170 |
| within-uploader stock AUC, 93 queries | **0.672-0.715** | 0.560 | 0.691 | 0.556 |
| ARTeFACT same-emulsion AUC (40 queries) | 0.62 | 0.73 | 0.61 | 0.75 |
| crop 80 % / flip / JPEG q50 / rescale on 324 scans | 0.87-0.88 / 1.00 / 0.98-0.99 / 0.95-0.97 | 0.99 / 1.00 / 1.00 / 0.99 | 0.45 under rescale (spent validation set, E49) | not run |

Paired differences (A seed-mean minus baseline): within-uploader AUC vs B' **+0.128 [+0.039, +0.232]**, vs B
+0.187 [+0.029, +0.360], vs C −0.013 [−0.083, +0.044]; museum p@5 vs C −0.052 [−0.084, −0.022], vs B' +0.007
[−0.027, +0.044]; stock-line p@5 vs C +0.022 [−0.008, +0.048].

Per stock (A, seed 0): Portra 0.32 and Tri-X 0.29 above chance; Superia 0.08, T-Max 0.13, HP5 0.14; Astia, Provia,
Kodachrome, Ultramax and five others 0.00; 187 of 311 stock queries had no relevant neighbour in the top five.

## 4. The aesthetic dimensions the product talks about

| dimension | what can be measured | result |
|---|---|---|
| highlight roll-off, tonal transitions | single-parameter gamma and lift edits on 113 validation originals, scene-grouped decoding (E54) | not linearly decodable from A (R² 0.17 / 0.37); monotone response yes (Spearman 0.77-0.92); C decodes lift (0.63) |
| colour response | temperature and saturation edits (E54); neighbour colour-statistic coherence (E50) | A: temperature R² 0.48, saturation 0.32; neighbours share colour statistics at 0.65 of the random difference |
| microcontrast / contrast | contrast edit (E54) | A decodes contrast R² 0.73 (C 0.89) |
| texture | blur edit (E54), texture-statistic coherence (E50) | A blur R² 0.13 (C 0.72); texture coherence 0.85 of random |
| grain naturalness | additive-grain edit (E54); fake-grain displacement (E49) | A grain R² 0.57; a fake-grain edit moves the embedding 8-10x a crop. "Naturalness" has no label anywhere |
| overall film-like aesthetics | no ground truth. The only proxy is film-vs-digital capture type from EXIF (`student_sup`, AUROC 0.78 [0.69, 0.88], ECE 0.20; spent BlueNeg) | subjective dimension, unmeasured |
| film-stock style similarity | sections 3 | within one photographer: supported (A ties C, beats B'); across photographers: at chance for everyone |
| photographer-style similarity | artist share and photographer p@5 | A is the *least* photographer-dominated representation; no validation of matching a photographer's style exists and none is claimed |

Objective measurements above; "film-like" and "beautiful" remain subjective and unmeasured.

## 5. Gemini-based analysis, separately

No independent ground truth exists for the Film Look Score or the photographer match. Available evidence: the
production rating form (5 DNA ratings, mean fit 4.0 / 5; 7 feedback rows; all from the launch week and probably the
owner's own tests), the earlier Lab evaluation (`film-evaluation/docs/evaluation-2026-11-02.md`, consistency and
specificity checks on a small set), and content-check behaviour (today a synthetic noise image was correctly refused
as "not a photograph"). Cost per successful analysis in production so far: $0.0034 at Gemini Flash-Lite list price
(5 successes, `llm_api_requests`). The student embedding and the Gemini analysis are not comparable on the same
metric; the student does not produce scores or explanations.

## 6. Limitations

- Labels are museum process records and uploader stock claims; neither is a judgement of look.
- The Met half is albumen-dominated (chance 0.65), so museum p@5 barely discriminates representations.
- ARTeFACT is 44 damaged scans of one photographer; CIs are over queries.
- Robustness on mounted or bordered digitisations is weak for every learned representation (crop 80 % 0.77-0.80
  on museum images).
- Everything learned here comes from 601 real photographs and 1,960 synthetic ones; the selection halves were
  tuned on for two sessions before the one fresh read.
- The service's memory and latency were measured on the development host, not on the E2 Micro.

## 7. Recommendation

- **Ready for production?** As an *exploratory similar-look embedding only*: the frozen student (A) is the cheapest
  content-invariant, re-encoding-robust representation we have, 32 bytes per image, and the CPU service exists.
  **Not** ready for any identification, score, or claim about film-stock or photographer; **not** better than the
  28 statistics on label accuracy.
- **Further training?** Not justified: every recipe change since session 5 was rejected or neutral on fresh data,
  the one supported gain is small, and the thing the product sells (aesthetic judgement with explanation) has no
  label to train on. The next evidence that would change this is human similarity judgements from real users, not
  another GPU run.
- Baseline comparison the owner asked for: if a label-accuracy feature were wanted, C (the 28 statistics) wins and
  costs nothing; if a robust "find photos that look like this one" feature is wanted, A wins on robustness and
  content invariance; B (frozen DINOv2) cannot run on the 1 GB instance and adds nothing on these sets.
