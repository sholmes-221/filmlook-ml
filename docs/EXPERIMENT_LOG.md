# Experiment log and leaderboard (main FilmLook model)

All runs on the development workstation (RTX 4090, shared; i9-14900KF). Seeds fixed (0). Splits are uploader-grouped
and shared by every synthetic derivative of a source. **Synthetic scores are never real-film scores**; the real-film
columns use (a) CommonCatalog film scans identified by scanner EXIF and (b) BlueNeg previews with owner-verified
stock labels (one photographer, external set, never trained on).

Label kinds used: `EXIF` (metadata), `measured` (deterministic statistics), `synthetic-param` (applied
transformation parameters), `owner-record` (BlueNeg stock), `none` (self-supervised). No human labels, no Claude labels.

## Leaderboard: real-film transfer (higher is better; chance stated)

| run | report | training labels | BlueNeg film vs digital AUROC (all / intact) | BlueNeg same-stock p@5 (chance 0.12) | unseen-generator (SDXL) treated AUROC | own-source rank-1 (lower = style-dominated) | size / CPU latency |
|---|---|---|---|---|---|---|---|
| frozen DINOv2 intermediate + explicit, linear probe trained on filmsynth | `synth_experiment_v2.json` §3, §6 | synthetic-param | **0.80 / 0.75** | 0.17 (SupCon projection: **0.23**) | **0.35 (inverted)** | 1.00 (raw), 0.44 (hybrid) | 88 MB backbone |
| MobileNetV3 style student, filmsynth only (`style_fs`) | `student_style_fs.json` | synthetic-param + SupCon distil | **0.81 / 0.75** | **0.24** (576-d), 0.20 (64-d) | 0.48 (chance) | **0.28** / 0.13 | 4.4 MB, 1.3 ms |
| MobileNetV3 style student, filmsynth + SDXL (`style_both`) | `student_style_both.json` | synthetic-param + SupCon distil | 0.67 / 0.64 (**worse**) | 0.22 (576-d), 0.18 (64-d) | 0.73 (seen in training) | 0.30 / 0.11 | 4.4 MB, 1.2 ms |
| MobileNetV3 style student, filmsynth **v2** (grain randomised, `style_fs_v2`) | `student_style_fs_v2.json` | synthetic-param + SupCon distil | 0.77 / 0.76 | 0.21 (576-d) | 0.54 (chance-ish, no inversion) | 0.34 / 0.13 | 4.4 MB, 1.2 ms |
| hybrid SupCon, filmsynth v2 + Z-Image families (E8b) | `synth_experiment_v4b.json` | synthetic-param | 0.65 / 0.63 (held-out Z-Image probe) | **0.26 [0.16, 0.34]** | held-out Z-Image 0.63 [0.52, 0.75] | 0.44 (hybrid) | 88 MB backbone |
| 16 coarse colour/tone statistics, linear (E15) | `colour_vs_texture.json` | synthetic-param | 0.53 (none: aesthetics axis) | 0.19 [0.11, 0.25] | cross-type mean 0.635 (SDXL->fs 0.74) | 0.84 | 16 numbers |
| `sat_clip` alone (E18) | `minimal_statistics.json` | synthetic-param | — | 0.09 | cross-type mean 0.66 | 0.04 | 1 number |
| existing supervised student (`student_sup`), P(film_scan) | `student_sup.json`, `film_authenticity_eval.json` | **EXIF capture type (real)** | 0.85 / 0.83 cross-source; **0.78 [0.69, 0.88] within CommonCatalog**, ECE 0.20 | not available (no embedding output) | — | — | 4.3 MB, 1.5 ms |
| frozen hybrid probe, filmsynth v2 (C3), linear | `synth_experiment_v3.json` | synthetic-param | 0.70 / 0.66 | SupCon 0.25 | **0.58 [0.52, 0.67]** / 0.48 | 0.29 (hybrid) | 88 MB backbone |

Reading: synthetic training produces a real-film signal (0.80) without any real labels, but a small model trained on
real EXIF capture labels is still better at film-vs-digital (0.86). Synthetic training adds something the EXIF
student lacks: a style embedding that retrieves same-stock scans at twice chance. Neither transfers across
generators.

## Runs

### E1 filmsynth v1 study (2026-11-07) — `reports/synth_experiment.json`, `synth_probes.json`
- Hypothesis: film-inspired parametric transformations with recorded parameters give a learnable, transferable
  style signal. Data: 100 colour sources × 8 variants (`film-synth-v1`). Representations frozen; heads linear.
- Result: family macro-F1 0.73 (hybrid_style) on unseen sources; pairwise attribute rankers 0.77-0.84 on independent
  edits of real photos; DINOv2 global memorises sources; composition not decodable from pooled features.
- Limitation: grain of another pipeline not recognised (generator fingerprint suspected).

### E2 cross-generator study v2 (2026-11-07) — `reports/synth_experiment_v2.json`
- Hypothesis: a second, independent generator (SDXL img2img, 3 treatments, strength 0.35) reveals whether E1's
  signal is film-likeness or a generator signature.
- Generation cost: 294 images, 1.1 s/image, 9.9 GB peak VRAM (fits a 12 GB 4070).
- Content preservation: SDXL own-source rank-1 0.98, DINOv2 cosine 0.80 (subject and composition kept).
- Generator identity leakage: a linear probe tells filmsynth / SDXL / untreated apart with 93% accuracy (chance 64%)
  on intermediate features; 72% on explicit statistics.
- **Leave-one-generator-out failed:** treated-vs-untreated AUROC 0.35 (train filmsynth → test SDXL) and 0.25-0.30 in
  reverse with intermediate features. Below chance = consistent inversion.
- Cause (`reports/generator_signature_diagnostic.json`): filmsynth treatments raise measured noise (0.0156 vs source
  0.0148) and cut saturation (0.15 vs 0.27); SDXL's VAE lowers noise (0.0114) and sharpness. The probe learned
  "noisier and duller than the source", which the two pipelines move in opposite directions.
- Real film: the same probe scores BlueNeg vs digital at AUROC 0.80 (intact 0.75); the SupCon projection retrieves
  same-stock scans at 0.23 vs 0.12 chance. Real scans sit on the "noisier" side like filmsynth, so the shortcut
  happens to align with one true film property (grain), but it is one axis, not film-likeness.
- Cross-generator retrieval: SDXL queries retrieve filmsynth-treated images below chance (0.27 vs 0.78).
- Robustness: the probe's SDXL predictions are stable under JPEG/rescale/exposure only because they are already
  "untreated".

### E3 style student (2026-11-07) — `reports/student_style_fs.json`, `student_style_both.json`
- Hypothesis: a MobileNetV3-Small trained end-to-end with treated/family/parameter heads and SupCon distillation
  keeps the real-film signal at E2 quality while being deployable, and is less generator-bound than frozen probes.
- `style_fs` (630 train images, 25 epochs, 4.5 s, 0.7 GB GPU): synthetic family F1 0.67, treated AUROC 0.87;
  unseen SDXL 0.48 (chance; no inversion); BlueNeg 0.81 / 0.75; same-stock p@5 0.24; own-source rank-1 0.28; ONNX
  4.4 MB, 1.3 ms/image on one CPU thread.
- Comparison: the existing EXIF-trained student reaches 0.86 on BlueNeg film-vs-digital. Real metadata supervision
  beats synthetic supervision for that task.
- `style_both` (840 train images, 6.7 s): adding SDXL outputs as "treated" raised the synthetic treated AUROC to 0.94
  and made SDXL test outputs detectable (0.73), but **real-film transfer dropped to 0.67 / 0.64** and the untreated
  sources themselves now score P(treated) = 0.78. With 70 pristine sources against 770 re-encoded images, "treated"
  collapsed into "passed through any pipeline". Robustness looks perfect (1.0) for the same reason: everything is
  called treated. Rejected per the Phase 6 rule (synthetic up, real down).

### E4 leakage benchmark v3 (2026-11-07) — `reports/synth_experiment_v3.json`, `generator_signature_diagnostic.json`
- Hypothesis: the inverted leave-one-generator-out AUROC comes from a nuisance (resolution, JPEG, VAE) or from the
  grain model being tied to the label. Cells add one fix at a time (hybrid features; AUROC fs→SDXL / SDXL→fs):
  C0 baseline 0.35 / 0.30 (reproduced); C1 + size/JPEG standardisation 0.39 / 0.36; C2 + SDXL VAE round-trip
  control 0.40 / 0.34; **C3 + grain randomised independently of the label 0.58 [0.52, 0.67] / 0.48**.
- Generator-identity accuracy (3-way, chance 0.58): 0.86 → 0.85 → 0.84 → **0.79**. Cross-generator treated
  retrieval (SupCon, chance 0.78): 0.27 → 0.29 → 0.25 → 0.53. Content leakage unchanged (own-source rank-1 1.00 raw,
  0.29 hybrid).
- Real film under C3: BlueNeg vs digital **0.80 → 0.70**; same-stock retrieval 0.23 → 0.25 (unchanged within noise).
  So the real-film separation of C0-C2 rode on the label-tied grain/texture structure. Single statistics do not
  reproduce it (`single_statistic_authenticity.json`: noise_std 0.52, noise_autocorr1 0.68 on BlueNeg), so it is
  texture *structure* in intermediate features, not grain amplitude alone.
- Reading: grain/texture ≈ authenticity cue; colour/tone treatment ≈ aesthetics cue, weaker but the only one that
  crosses generators. Keep them in separate heads (section 6 of the brief).
- filmsynth v1 reproducibility re-verified after the engine change (full-dataset sha unchanged; render pinned in tests).

### E5 film-authenticity student audit (2026-11-07) — `reports/film_authenticity_eval.json`, `blueneg_datestamp_check.json`
- Within CommonCatalog test (89 originals, 30 film): AUROC **0.78 [0.69, 0.88]**, Brier 0.21, ECE 0.20 (not
  calibrated). Per scanner: Noritsu 0.85 (n=16), other 0.75; too few Epson/Canon to say.
- Cross-source: BlueNeg vs CommonCatalog digital 0.85 (intact 0.83); **BlueNeg vs CommonCatalog film 0.63** (both
  film: the model prefers the BlueNeg source). A measured-statistics probe separates the two film sources at 0.89
  (chance 0.85). The earlier "0.86" therefore mixes film-vs-digital with source identity.
- Burned-in date stamps (53/491 BlueNeg frames) are not the shortcut: stamped frames score *lower* (0.56 vs 0.75)
  and the centre crop drops most of the corner.
- Robustness within source: JPEG q55 0.78, 0.5× rescale 0.78, 80 % crop 0.80; flip rates 2-8 %.
- Decision: stays the production authenticity baseline, labelled "experimental, uncalibrated"; its number to quote
  is 0.78 within-source, not 0.86.

### E6 spatial composition probe (2026-11-07) — `reports/spatial_probe.json`
- 240 real sources × 4 controlled crops (zoom 1-2×, signed x/y offsets), 38 test sources, ridge probes.
- Pooled DINOv2 global: signed offsets R² < 0, mirror ratio 0.05 (flip-blind). **2×2 patch grid: x 0.24, y 0.23,
  mirror 0.18**; 4×4 grid: 0.12 / 0.20, mirror 0.25; MobileNetV3 2×2 feature-map grid: zoom 0.22, mirror 0.29.
  Intermediate-layer statistics recover zoom best (0.49) through texture scale.
- Reading: framing becomes partly decodable with a spatial grid, but R² ≈ 0.2 is weak. Composition badges stay off.

### E7 style student on filmsynth v2 (2026-11-07) — `reports/student_style_fs_v2.json`
- Same recipe as E3 on grain-randomised data: unseen SDXL 0.54 (E3: 0.48), BlueNeg 0.77 / 0.76 (E3: 0.81 / 0.75),
  same-stock 0.21 (E3: 0.24), content rank-1 0.34. No credible real-film difference at these sizes; cross-generator
  behaviour no longer inverted. Kept as a research variant, not promoted.

### E9_layer_ablation — transferable style lives in specific DINOv2 layers; the 4/8 pair was untested
- Status: done; wall 179.6 s; peak GPU 20714.0 MB (total on card).
- best_layer_fs_to_sdxl [synthetic]: 0.6444 CI [0.545, 0.77]
- best_layer_same_stock [real]: 0.2471
- Decision rule: neutral: no credible real-film change
- Learned: Cross-generator transfer rises with depth (SDXL->filmsynth 0.31 at L2 to 0.53 at L12; filmsynth->SDXL best at L9, 0.64 [0.55, 0.77]); real-film signal peaks at mid layers (BlueNeg 0.71 and same-stock 0.25 at L4-L5) and collapses at depth (0.14). Shallow layers carry generator texture signatures, deep layers are generator-invariant but blind to real film texture. No single layer beats L4+L8 beyond the CI width.
- Next: E9b: mid+deep pairs (L4+L9, L5+L9, L5+L12) to test whether one feature set can hold both signals.

### E9b_layer_combos — a mid+deep layer pair captures both real-film texture (mid) and generator-invariant treatment (deep)
- Status: done; wall 188.3 s; peak GPU 20734.0 MB (total on card).
- best_layer_fs_to_sdxl [synthetic]: 0.6281 CI [0.494, 0.778] (baseline 0.6444, neutral)
- best_layer_same_stock [real]: 0.2318 (baseline 0.2471, neutral)
- Decision rule: neutral: no credible real-film change
- Learned: L4+L9: filmsynth->SDXL 0.62 [0.56, 0.73] vs 0.59 for L4+L8, same BlueNeg (0.70) and same-stock (0.23); three-layer mixes add nothing. No static layer mix resolves the mid-vs-deep trade-off beyond CI width. Keep L4+L8; L4+L9 is equivalent.
- Next: Stop layer search. Test a different objective instead (E11 parameter-similarity embedding).

### E11_parameter_similarity_supcon — continuous parameter-similarity positives (across sources) give a better cross-photo style embedding than family labels
- Status: done; wall 165.3 s; peak GPU 20734.0 MB (total on card).
- best_param_same_stock [real]: 0.2059
- family_same_stock [real]: 0.2482
- Decision rule: neutral: no credible real-film change
- Learned: Soft parameter-similarity positives fit the synthetic parameter geometry (embedding-distance vs parameter-distance Spearman 0.76-0.79 vs 0.28 for family SupCon) but real-scan same-stock retrieval fell to 0.21/0.19/0.18 (tau 0.2/0.35/0.7) from 0.25 with family labels; cross-generator retrieval also no better. Coarse family grouping transfers to real film better than the engine's fine parameterisation. Hypothesis falsified by its own criterion (needed > 0.28).
- Next: Rejected. Do not refine tau; the objective is wrong for the real target. Keep family SupCon as the embedding objective; test spatial supervision (E12) and the third generator (E8).

### E12_spatial_student — a MobileNetV3 student with a 2x2 feature-map grid head learns framing (zoom, signed offsets) from exact crop geometry at least as well as the frozen DINOv2 grid probe
- Status: done; wall 39.2 s; peak GPU 21326.0 MB (total on card).
- grid_x_offset_r2 [proxy]: -0.1502
- grid_y_offset_r2 [proxy]: 0.0351
- grid_zoom_r2 [proxy]: 0.0678
- global_x_offset_r2 [proxy]: -0.2463
- grid_mirror [proxy]: 0.4747
- Decision rule: neutral: no credible real-film change
- Learned: End-to-end MobileNetV3 with a 2x2 grid head trained on exact crop geometry (2,744 crops, 343 sources) does not recover framing on 57 held-out sources: x-offset R^2 -0.15, y 0.04, zoom 0.07 (offset MAE ~0.5 on a +/-1 target), below the frozen DINOv2 grid probe (0.24 / 0.23). Grid embedding is mirror-sensitive (0.47 vs 0.13 global), so left/right is encoded but not mapped to offset. Falsified (criterion R^2 < 0.1).
- Next: Drop direct geometry regression for the student. If composition is revisited, distil the DINOv2 grid features (which do carry offset) rather than regress geometry.

### E8_zimage_third_generator — grain-randomised synthetic training generalises to an unseen diffusion family (Z-Image-Turbo)
- Status: done; wall 223.4 s; peak GPU 7369.0 MB (total on card).
- heldout_zimage_auroc [synthetic]: 0.6281 CI [0.521, 0.748]
- heldout_sdxl_auroc [synthetic]: 0.5807 CI [0.527, 0.678]
- heldout_filmsynth_auroc [synthetic]: 0.4654 CI [0.358, 0.597]
- gen_id_4way_acc [proxy]: 0.7667
- blueneg_auroc_heldout_zimage [real]: 0.6462
- same_stock_supcon_heldout_zimage [real]: 0.1318 CI [0.082, 0.169]
- Decision rule: neutral: no credible real-film change
- Learned: Held-out Z-Image (train filmsynth+SDXL) 0.63 [0.52, 0.75]: CI excludes chance, so the criterion is met, but the gain sits in diffusion<->diffusion pairs (SDXL->Z-Image 0.65 [0.57, 0.73], Z-Image->SDXL 0.65 [0.52, 0.77]) while filmsynth<->diffusion stays at 0.47-0.58 and a probe trained on the two diffusion families scores real film INVERTED (BlueNeg 0.39). Generator identity stays high (4-way 0.77 vs chance 0.44). Content preserved for Z-Image (own-source rank-1 0.96). Real-film transfer did not improve (0.65 vs 0.70 for filmsynth alone).
- Next: Scaling (E13) not justified: generator identifiability did not fall and real-film did not improve. Test the artifact explanation directly (E14).

### E8b_v4_fine_supcon_labels — E8 retrieval re-run with fine SupCon labels (filmsynth families + diffusion treatments); the E8 collapse to chance was an evaluation bug
- Status: done; wall 220.2 s; peak GPU 7371.0 MB (total on card).
- same_stock_supcon_heldout_zimage [real]: 0.2176 CI [0.107, 0.311] (baseline 0.1318, improved?)
- same_stock_supcon_heldout_sdxl [real]: 0.2576 CI [0.162, 0.337]
- same_stock_hybrid_raw [real]: 0.1459 CI [0.081, 0.193]
- heldout_zimage_auroc [synthetic]: 0.6281 CI [0.521, 0.748] (baseline 0.6281, neutral)
- Decision rule: accept: real improvement
- Learned: With family/treatment labels the SupCon same-stock retrieval is 0.22 [0.11, 0.31] (filmsynth+SDXL) and 0.26 [0.16, 0.34] (filmsynth+Z-Image) vs raw 0.15 [0.08, 0.19]; the E8 collapse to 0.13 was an evaluation bug (coarse labels). The roll-bootstrap CIs (27 rolls) are wide: embedding variants within 0.03-0.04 of each other are indistinguishable on BlueNeg.
- Next: Treat same-stock differences below ~0.08 as noise; a second real-film source is needed for a sharper retrieval benchmark.

### E14_diffusion_artifact_probe — diffusion<->diffusion 'treated' transfer is a shared VAE/diffusion artifact: an SDXL-trained probe will call Z-Image's VAE control 'treated'
- Status: done; wall 152.3 s; peak GPU 7359.0 MB (total on card).
- A_treatment_transfer [synthetic]: 0.6474 CI [0.567, 0.734]
- B_artifact_on_vae_control [proxy]: 0.5244 CI [0.447, 0.629]
- E_vae_vs_raw [proxy]: 0.5289 CI [0.471, 0.596]
- Decision rule: neutral: no credible real-film change
- Learned: Not a VAE artifact: the SDXL-trained treated probe scores Z-Image's untreated VAE control vs raw at 0.52 [0.45, 0.63] (B), VAE controls vs raw are barely separable in hybrid features (0.53 [0.47, 0.60]), while true treatment transfer SDXL->Z-Image is 0.65 [0.57, 0.73] (A). The diffusion<->diffusion direction is a shared treatment signal, not a VAE round-trip signature; a shared denoising signature remains possible and real-film transfer is still unimproved.
- Next: Characterise the two signal types (programmatic vs diffusion) by which measured statistics they move; test whether colour/tone alone carries stock identity on real scans (E15).

### E15_colour_vs_texture — film-stock identity on real scans is carried by colour/tone statistics (aesthetics), not by texture (authenticity)
- Status: done; wall 210.7 s; peak GPU 7369.0 MB (total on card).
- colour_tone_same_stock_supcon [real]: 0.1906 CI [0.11, 0.253]
- texture_same_stock_supcon [real]: 0.1624 CI [0.124, 0.201]
- hybrid_same_stock_supcon [real]: 0.2471 CI [0.153, 0.314]
- colour_tone_blueneg_auroc [real]: 0.5295
- texture_blueneg_auroc [real]: 0.2886
- Decision rule: neutral: no credible real-film change
- Learned: Colour/tone statistics alone (16-d) transfer across generator TYPES in the previously failing direction: SDXL->filmsynth 0.74, Z-Image->filmsynth 0.71 (texture-only 0.47/0.55; DINOv2 L4+L8 0.44/0.44). Colour/tone alone does not separate real film from digital (BlueNeg 0.53) while texture does (DINOv2 0.70; explicit texture inverted 0.29). Same-stock retrieval: colour/tone 0.19 [0.11, 0.25], texture 0.16 [0.12, 0.20], hybrid SupCon 0.25 [0.15, 0.31]; CIs overlap, so stock identity is not shown to be colour-only. Authenticity = texture axis; generator-agnostic aesthetics transfer = colour/tone axis.
- Next: E16: texture-neutralised DINOv2 features (downsample before encoding) to get a colour/tone/layout representation that transfers across generator types without texture signatures.

### E16_texture_neutral_dino — DINOv2 features of texture-neutralised images transfer across generator types in both directions and keep same-stock retrieval
- Status: done; wall 290.7 s; peak GPU 7383.0 MB (total on card).
- best_cross_type_mean [synthetic]: 0.5048
- best_same_stock_supcon [real]: 0.2247 CI [0.12, 0.292]
- control_hybrid_same_stock [real]: 0.2471 CI [0.153, 0.314]
- Decision rule: neutral: no credible real-film change
- Learned: Texture-neutralised DINOv2 (downsample to 64/96/128 px before encoding) transfers WORSE across generator types (cross-type mean 0.45-0.51 vs 0.635 for the 16 colour/tone statistics; Z-Image->filmsynth inverts to 0.32-0.38), is fully content-dominated (own-source rank-1 1.00) and does not improve same-stock retrieval (0.20-0.24 vs 0.25 hybrid, CIs overlap). Deep features encode content-tied colour shifts; global low-dimensional statistics are what transfers.
- Next: Rejected. Build the aesthetics representation from richer GLOBAL colour-rendition statistics instead (E17); keep DINOv2 texture for authenticity.

### E17_colour_rendition_descriptor — a richer global explainable colour-rendition descriptor transfers across generator types at least as well as the 16-d set and lifts real-scan same-stock retrieval above 0.19
- Status: done; wall 37.9 s; peak GPU 6271.0 MB (total on card).
- cr80_cross_type_mean [synthetic]: 0.4279
- cr80_same_stock_supcon [real]: 0.1776 CI [0.113, 0.237]
- cr80_same_stock_raw [real]: 0.16 CI [0.066, 0.258]
- Decision rule: neutral: no credible real-film change
- Learned: The 80-d colour-rendition descriptor transfers WORSE than the 16 coarse statistics (cross-type mean 0.43 vs 0.635; SDXL->filmsynth 0.34, Z-Image->filmsynth 0.33, inverted), is fully scene-identifying (own-source rank-1 1.00) and does not lift same-stock retrieval (0.18 [0.11, 0.24]). Concatenating it with the 16-d set drags the 16-d set down to 0.46. Per-channel percentiles and histograms are scene fingerprints; probes on them learn content-tied hue shifts. Richer is not more transferable: the generator-agnostic aesthetics signal lives in a few coarse treatment-level statistics.
- Next: Ablate the 16 statistics to the minimal transferable set (E18, CPU); seek a second real-film source for the retrieval benchmark rather than more descriptors.

### E18_minimal_transferable_statistics — a subset of the 16 coarse colour/tone statistics carries all cross-type transfer; the rest add scene identity
- Status: done; wall 23.9 s; peak GPU 5740.0 MB (total on card).
- chosen_cross_type_mean [synthetic]: 0.6599
- full16_cross_type_mean [synthetic]: 0.6352
- chosen_same_stock_raw [real]: 0.0929 CI [0.004, 0.22]
- Decision rule: neutral: no credible real-film change
- Learned: Greedy selection stops after ONE statistic: sat_clip alone gives cross-type mean 0.66 (full 16-d 0.635): SDXL->filmsynth and Z-Image->filmsynth 0.76 [0.69, 0.84], filmsynth->SDXL 0.60, filmsynth->Z-Image 0.52. It carries no scene identity (own-source rank-1 0.04) and no stock identity (same-stock 0.09, below chance). The generator-agnostic 'treated vs untreated' signal is one-dimensional: treatments change saturation clipping. Treated-vs-untreated is not an aesthetics representation; it was a proxy that reduced to a single clipping statistic.
- Next: Retire treated-vs-untreated as a target. Keep cross-photo same-stock retrieval on real scans as the aesthetics benchmark (hybrid SupCon 0.25 [0.15, 0.31]); it needs a second real-film source before further model work is decisive.

### B0 content-invariance benchmark v1, val split (2026-11-07) — `docs/BENCHMARK.md`, `reports/style_benchmark_val.json`
- Infrastructure: 3,716 standardised images featurised once (`data/embeddings/bench_features_v1.*`, 317 s GPU, 37 MB). Selection on val (15 sources); test split + BlueNeg reserved for one final run per candidate.
- Baselines (A p / A2 ratio / B filmsynth p@5 (chance 0.11) / C same-generator share (prior 0.40) / E rank-1 / E subject probe (majority 0.20, content ceiling 0.53)):
  - raw_dino_global: 0.81 / 0.08 / 0.16 / 0.45 / 0.96 / 0.53
  - raw_hybrid: 0.53 / 0.24 / 0.17 / 0.42 / 0.39 / 0.56
  - hybrid_supcon_family: 0.91 / 1.71 / 0.47 / 0.54 / 0.37 / 0.24
  - student_fs_v2_64: 0.88 / 5.09 / 0.53 / 0.60 / 0.10 / 0.26
  - colour_tone16: 0.99 / 3.62 / 0.31 / 0.51 / 0.52 / 0.23
  - texture12: 0.44 / 0.18 / 0.13 / 0.41 / 0.29 / 0.19
  - raw_dino_style_L4L8: 0.83 / 0.12 / 0.19 / 0.52 / 0.98 / 0.57
- Learned: raw DINOv2 global is a content embedding (ratio 0.08, rank-1 0.96, subject probe at ceiling). Family SupCon on hybrid features and the v2 student are style-dominated (ratios 1.7 / 5.1, subject probe near majority, neighbour overlap with content 0.06-0.08). Two failures: diffusion treatment ids are not retrievable across scenes by any candidate (B sdxl/zimage = chance 0.33), and learned embeddings cluster by generator (share 0.54-0.60 vs prior 0.40).
- Next: E19 projection trained on all generators' labels with same-source hard negatives and a content-decorrelation penalty.

### E19_content_invariant_projection — a projection trained on all generators' treatment labels with same-source hard negatives and a content-decorrelation penalty raises cross-scene same-treatment retrieval (incl. diffusion) and lowers generator share and semantic leakage, on the val split
- Status: done; wall 4.4 s; peak GPU 6353.0 MB (total on card).
- B_fs_decorr [synthetic]: 0.4533 CI [0.417, 0.49]
- B_sdxl_decorr [synthetic]: 0.3778 CI [0.333, 0.427]
- Decision rule: neutral: no credible real-film change
- Learned: All-generator labels raise diffusion same-treatment retrieval only slightly (0.39-0.43 vs chance 0.33) while the generator fingerprint rises (same-generator share 0.70 vs 0.54, probe 0.64 vs 0.51) and semantic leakage grows (subject probe 0.35 vs 0.24); hard negatives and the content-decorrelation penalty change almost nothing. Diagnostic (reports/diffusion_treatment_distinctness.json): two SDXL treatments of the same source differ by 0.19x the JPEG+rescale nuisance under raw features (Z-Image 0.35x; filmsynth 1.6x; under SupCon 1.0x vs 13.7x). The diffusion treatment ids at strength 0.35 are not distinct beyond nuisance: B(sdxl/zimage) at chance is a label problem.
- Next: Rejected. Regenerate diffusion treatments at a strength where they are distinct (E10, strength 0.5) and re-measure distinctness, cross-scene retrieval and content preservation before using them as labels again.

### B1_bench_test_final — Final one-time test-split evaluation of the retained candidates on benchmark v1 (A-E plus D BlueNeg same-stock); no selection on this run
- Status: done; wall 15.5 s; peak GPU 20194.0 MB (total on card).
- B_fs [proxy]: 0.435
- D_p5 [real]: 0.2482
- Decision rule: neutral: no credible real-film change
- Learned: One-time test run of benchmark v1 (docs/BENCHMARK.md, Results). Val and test agree within 0.05 on every column. hybrid + family SupCon: B filmsynth 0.44 [0.39, 0.48], own-source rank-1 0.33, generator share 0.59, BlueNeg same-stock 0.25 [0.16, 0.31] (only candidate clearly above the 0.17 DINOv2 reference). student_fs_v2: best on synthetic (B 0.55, rank-1 0.16) but 0.18 on real scans; colour/tone 16: least semantic content (subject probe 0.25) but 0.18 on real scans. Abstention (reports/abstention_test.json): top-1 cosine similarity orders the hybrid SupCon errors on real scans (rank-1 0.22 -> 0.37 at 50% coverage, 0.50 at 25%, AURC 0.59 vs 0.78 unordered); the student's confidence is uninformative on real data.
- Next: Record in MODEL_CARD/FINDINGS; keep hybrid + family SupCon as the research embedding with abstention below the 50%-coverage threshold. Test split is now spent for this candidate set.

## Decisions taken from these runs (autonomous session 2026-11-07, E8-E18)

4. "Treated vs untreated" is retired as a research target: across generator types it reduces to one statistic
   (saturation clipping, E18); within a generator type it is a generator signature (E2-E4). It is neither aesthetics
   nor authenticity.
5. The aesthetics benchmark is cross-photo same-stock retrieval on real scans (BlueNeg, owner labels). Best: hybrid
   (explicit + DINOv2 L4+L8) with family SupCon, 0.25 [0.15, 0.31] vs chance 0.12. CIs are wide (27 rolls): a second
   real-film source is worth more than another model variant.
6. Authenticity = texture axis (DINOv2 mid layers; explicit texture statistics inverted on real film). The EXIF
   student stays the baseline (0.78 within source).
7. Deep or rich descriptors re-encode scene identity (E16, E17); coarse global statistics are the explainable
   attribute layer. Composition: frozen DINOv2 grid probe only (E12 student failed).
8. Scaling synthesis (E13) is not justified by E8; a third generator improved nothing real.

## Decisions taken from these runs (earlier)

1. Grain randomisation (film-synth-v2) is the fix that worked for cross-generator leakage; standardisation and the
   VAE control helped little but stay on as hygiene. Scale beyond 100 sources only with v2-style presets, and add a
   third generator (Z-Image-Turbo) before trusting any cross-generator number.
1b. Authenticity and aesthetics are separate heads: the grain/texture axis predicts film capture; colour/tone
   treatment is the transferable aesthetics signal. Never report one as the other.
2. Keep the EXIF capture-type head as the film-vs-digital signal in production; add the synthetic-trained style
   embedding for retrieval only, labelled experimental.
3. Report BlueNeg numbers as "film scans by one photographer, 1990s consumer stocks"; they are not a general film
   benchmark.
