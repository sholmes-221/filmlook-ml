# MVP decision: the photography-style feature for FilmLook (2026-11-08, session 7)

Research is stopped here by the owner's instruction. This page is the deliverable: the model to ship, the
independent evidence, the comparison with the simple baseline, the limits, the serving cost and a recommendation.

## 1. Selected model and exact checkpoint

**Representation:** the pooled 576-d layer of the teacher-distilled MobileNetV3-Small student trained with the
re-encoding augmentation (E52 recipe), reduced by a label-free PCA to 32 components and quantised to int8.

| artifact | value |
|---|---|
| checkpoint | `checkpoints/frozen_session7/style_fs_v2_real_f576_deg/model.onnx` (seed 0 of three; seeds `_s1`, `_s2` are equivalent) |
| SHA-256 (model.onnx) | `46b9b4f1e36440ff451f8687926963e831eb02badf5b65bb147efaeca680b3c5` |
| reduction | `checkpoints/frozen_session7/style_fs_v2_real_f576_deg/pca.npz` (mean, 64 nested components, int8 scales), PCA fitted on the 601 CC-BY train originals; use the first 32 components |
| embedding | 32 bytes per image (int8) |
| size | 4.4 MB fp32 ONNX + 150 KB PCA |
| frozen | 2026-11-08 15:29, before the holdout was read; hashes in `freeze.json`, `docs/FROZEN_CANDIDATE_S7.md` |

The explanation layer is the deterministic 28 measured statistics (`filmlook_infer` explicit features), which
ship beside the embedding (see section 3 for why both).

## 2. Independent evaluation (E53, one read, 2026-11-08)

Fresh data never used for training, selection or any earlier reading: The Met public-domain photographs
(378 analysis-half images, 140 artists, CC0), a second CommonCatalog film-scan snapshot from 91 uploaders
never seen before (324 scans with an uploader-claimed stock), and the 44 ARTeFACT originals (one photographer,
owner-recorded emulsions). Three seeds, CIs by bootstrap over artists / uploaders, paired effect sizes.
`reports/e53_fresh_holdout.json`, `docs/EXPERIMENT_LOG.md` (E53).

| reading (PCA-32 int8, three seeds) | candidate | untrained ImageNet backbone, same PCA | 28 statistics | DINOv2-Small L4+L8 PCA-64 |
|---|---|---|---|---|
| museum process p@5, artist excluded, decade matched (chance 0.648) | 0.694-0.700, lower CIs 0.642-0.648 | 0.678 | **0.746** [0.697, 0.794] | 0.678 |
| artist share of top-5 (lower is better) | 0.078-0.081 | 0.105 | 0.130 | 0.136 |
| stock-line p@5, uploader excluded (chance 0.080) | 0.141-0.156 [0.10, 0.20] | 0.142 | 0.119 | 0.140 |
| uploader share of top-5 (lower is better) | 0.128-0.144 | 0.138 | 0.181 | 0.170 |
| within-uploader different-day stock AUC (n = 93 queries) | **0.672-0.715** [0.51, 0.85] | 0.560 [0.46, 0.63] | 0.691 [0.55, 0.85] | 0.556 |
| ARTeFACT same-emulsion AUC, 40 queries (CI over queries) | 0.62 [0.54, 0.70] | 0.73 [0.65, 0.79] | 0.61 [0.55, 0.68] | 0.75 [0.69, 0.81] |
| ARTeFACT same-emulsion p@3 (chance 0.21) | 0.39-0.43 | 0.46 | 0.49 | 0.58 |
| own-image rank-1 under crop 80 % / flip / JPEG q50 / 0.5x rescale on the 324 scans | 0.87-0.88 / 1.00 / 0.98-0.99 / 0.95-0.97 | 0.99 / 1.00 / 1.00 / 0.99 | not run here; 0.45 under rescale on the validation set (E49) | n/a |
| 70 % corner crop on scans | 0.63 | 0.71 | n/a | n/a |

Paired effect sizes (seed-mean of the candidate minus baseline, bootstrap over groups):

| difference | mean [95 % CI] | d |
|---|---|---|
| within-uploader AUC vs ImageNet-init | **+0.128 [+0.039, +0.232]** | 0.39 |
| within-uploader AUC vs DINOv2-global | +0.187 [+0.029, +0.360] | 0.46 |
| within-uploader AUC vs 28 statistics | −0.013 [−0.083, +0.044] | −0.05 |
| museum p@5 vs 28 statistics | −0.052 [−0.084, −0.022] | −0.20 |
| museum p@5 vs ImageNet-init | +0.007 [−0.027, +0.044] | 0.02 |
| stock-line p@5 vs 28 statistics | +0.022 [−0.008, +0.048] | 0.10 |

Pre-registered rule: **falsified** (two of six guards failed on every seed: the museum lower CI did not clear the
albumen-heavy chance level of 0.648, and the 80 % centre crop on the larger scan bank read 0.87 against the
0.9 bound). The decisive claim (e) held: the within-photographer stock separation exceeds the untrained
backbone with a lower CI above zero, the first time on fresh data.

Abstention (E55, supported, `reports/e55_abstention.json`): with the gate "3 of 5 neighbours agree and top-1
cosine above a threshold calibrated on the spent halves", museum process claims reach 0.754 [0.686, 0.824]
precision at 39 % coverage (chance 0.468, answer-everything 0.747); stock-line claims reach 0.24-0.27 at
31-36 % coverage (chance 0.08, answer-everything 0.16). The calibration is unstable across seeds (seed 0
found no threshold at the 3/5 gate); stock claims stay far too weak for the product.

## 3. Comparison with the simple baseline, and the choice

The 28 handcrafted statistics are better or equal on every *label* metric: museum process (+0.05 over the
student, significant), within-photographer stock (tie), ARTeFACT (tie). The learned model is not chosen for
label accuracy. It is chosen for what a similar-look feature needs and the statistics lack:

- **robustness to re-encoding and crops**: the statistics lose their nearest neighbour under a 0.5x rescale
  55 % of the time and under JPEG q50 often (E49); the student keeps it 95-99 % of the time on fresh scans;
- **content and photographer invariance**: uploader share 0.13 vs 0.18, artist share 0.08 vs 0.13, and no
  linear content signal (4 % of variance vs 13 % for the untrained backbone, E56);
- **a fixed, tiny vector**: 32 bytes with cosine similarity, versus 28 heterogeneous statistics that need
  standardisation and weighting.

Against the untrained ImageNet backbone with the same PCA, the student buys exactly one thing with evidence:
within-photographer stock separation (+0.13, CI above zero). The untrained backbone is more robust to crops
(0.99 vs 0.88) and better on ARTeFACT (0.73 vs 0.62); it is content-dominated (own-source rank-1 0.93, E48).

**Decision:** ship the student PCA-32 int8 as the *similar-look* embedding, paired with the 28 statistics as the
explanation and re-ranking layer, and nothing as a label. This is the configuration the CPU service implements.

## 4. Known limitations and failure cases

- No metric shows film-stock identification: cross-photographer stock p@5 is 0.14-0.16 against chance 0.08
  for every representation, and the untrained backbone equals the student. Per stock (seed 0): Portra 0.32,
  Tri-X 0.29 (the two largest classes), everything else at or below 0.13, nine stocks at 0.00. 187 of 311
  stock queries had no relevant neighbour in the top five.
- Museum process retrieval is reached by everything (chance 0.65 on the Met half because albumen is two
  thirds of it); the student's failure cases are salted-paper and daguerreotype queries retrieving albumen
  prints at cosines 0.91-0.98, i.e. the embedding sees paper prints of the 1850s-60s as one look.
- Mounted or bordered digitisations shift the embedding under a 10 % crop (museum crop80 0.77-0.80); user
  photographs do not (0.87-0.88 on scans, 0.96-1.00 on the validation originals).
- Corner crops (70 %) remain the weakest geometric case (0.63 on scans).
- Within one photographer with owner-verified emulsions (ARTeFACT), the student is below the untrained
  backbone and DINOv2; the 44 damaged scans are a small, hard set, but the direction is clear.
- Edits of single parameters are encoded monotonically but not as linear, scene-independent axes (E54,
  falsified): the embedding is a holistic look signature; use the statistics for "why similar".
- Everything above rests on catalogue metadata and uploader claims; no human judgement of "look" exists.
- The analysis-half rule tuning of session 6 (E43-E52) and the single fresh read mean one more honest read
  needs new data again (the Met test half is still untouched).

## 5. Estimated Oracle Always Free resource requirements

Measured on the development host pinned to one core (`reports/service_bench.json`, `service/README.md`):
RSS 65 MB idle, 104 MB with the model loaded, 135 MB peak under six concurrent clients; 21 ms per embedding,
113 ms with the 28 statistics; with two in-flight requests allowed, surplus requests get 503 immediately.
An E2 Micro core is several times slower: plan for **150-300 ms per embedding, 0.6-1.5 s with statistics,
under 200 MB RSS, `MemoryMax=400M`**, one process, two concurrent requests. No GPU, no new paid resource.
These are estimates until measured on the instance itself.

## 6. Minimal integration

1. Copy `src/filmlook_infer/`, `service/` and `checkpoints/frozen_session7/style_fs_v2_real_f576_deg/` to the
   host; `python -m venv .venv && pip install -r service/requirements.txt` (onnxruntime, numpy, pillow).
2. Run `service/filmlook-style.service` (systemd) or `FLS_MODEL_DIR=... FLS_PORT=8765 python service/filmlook_style_service.py`;
   bind to 127.0.0.1 behind the existing Caddy.
3. `POST /embed?explicit=0` with the image bytes returns the 32 int8 values and their scales; store them with the
   application's own consented images to build a bank (`service/build_bank.py` does it from a manifest).
4. `POST /similar?k=5` returns the nearest bank entries with attribution and a `decision` that is
   `similar_look_only` or `abstain`; never surface a neighbour's stock or photographer as a fact.
5. The Mini Tool can consume the same endpoint later; nothing in `js/filmlook-style/` changes now.

## 7. Recommendation: LIMITED GO

- **GO** for an experimental "similar look" feature: cosine neighbours on the 32-byte embedding, re-ranked or
  explained by the 28 statistics, labelled exploratory, behind the abstention rule, on the CPU service.
- **NO-GO** for any film-stock or photographer identification, any accuracy claim, and any replacement of the
  current Film Look scoring or the LLM analysis.
- Conditions: measure memory and latency on the E2 Micro before exposing it; keep the embedding version
  pinned to the frozen hashes; collect user "similar / not similar" feedback before any further model work.

No model was promoted, deployed or pushed. Research stops here; the next experiment, if any, would be a
second fresh museum read (the untouched Met test half) only after the product has gathered human similarity
judgements.
