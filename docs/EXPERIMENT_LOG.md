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

### E10_sdxl_strength_050 — SDXL treatments at strength 0.5 are distinct beyond nuisance (ratio > 1 under raw features) while keeping content (own-source rank-1 > 0.9), making diffusion treatment ids usable
- Status: done; wall 688.4 s; peak GPU 20218.0 MB (total on card).
- new_variants [proxy]: 500.0
- seconds_per_image [proxy]: 1.36
- Decision rule: neutral: no credible real-film change
- Learned: 500 SDXL variants at strength 0.5 (reports/sdxl_strength_sdxl_i2i_s050.json vs _sdxl_i2i_v1.json, same script). Per-image change grows (treatment-vs-treatment distance / nuisance: raw hybrid 0.60 -> 0.75, family SupCon 0.69 -> 2.15, colour/tone 3.4 -> 12.5) but it is not a consistent treatment: same-treatment retrieval across other sources stays at chance under every feature (0.26-0.31 vs chance 0.25; at 0.35: 0.34-0.36 vs 0.33), the treatment probe falls (raw hybrid 0.55 at 4-way vs 0.67 at 3-way), and content preservation drops (own-source rank-1 under DINOv2 global 0.97 -> 0.82). Higher strength makes diffusion rewrite each image more, differently for each image; it does not create a shared style signature.
- Next: Falsified. Diffusion img2img treatment ids are retired as style labels at any strength tried; diffusion sets stay as generator-fingerprint and nuisance controls only. Do not scale diffusion generation (E13 stays deferred).

### E22_student_teacher_agreement — H-domain: the student reproduces its teacher on filmsynth but not on real photographs (distillation input-domain shift), so distilling on unlabeled real images should close part of the synthetic-to-real gap
- Status: done; wall 2.8 s; peak GPU 6257.0 MB (total on card).
- jac_fs_val [proxy]: 0.3491
- jac_blueneg_corr [real]: 0.0489
- jac_cc [real]: 0.0418
- Decision rule: neutral: no credible real-film change
- Learned: The student reproduces its teacher on synthetic images (top-10 neighbour Jaccard 0.35 on filmsynth val/test, 0.32-0.38 on SDXL/Z-Image; similarity Spearman 0.59-0.68; the teacher's own nuisance self-agreement is 0.49 / 0.77) but not on real photographs: Jaccard 0.04-0.05 and Spearman 0.48 on CommonCatalog originals, 0.35 on BlueNeg blue-corrupted. Neither embedding follows DINOv2-global content on real images (Jaccard < 0.06), so the student's real-scan geometry is neither the teacher's nor content: it is undefined off the synthetic manifold. H-domain supported (reports/student_teacher_agreement.json).
- Next: E23: re-distil the student with unlabeled real images (CommonCatalog train originals, filmsynth sources) in the distillation input set, teacher targets only; evaluate agreement on real sets and benchmark v1 val, then Commons analysis half.

### E23b_student_real_distil — H-domain: adding unlabeled real CommonCatalog train originals as distillation-only inputs makes the student reproduce the teacher on real photographs (agreement up) without losing synthetic style retrieval
- Status: done; wall 54.4 s; peak GPU 7647.0 MB (total on card).
- rank1_kd64 [proxy]: 0.181
- fam_f1 [proxy]: 0.6501
- Decision rule: neutral: no credible real-film change
- Learned: Adding 534 unlabeled CommonCatalog train originals as distillation-only inputs (teacher targets, treated/family/parameter losses masked; 9 s of training) raises the student's agreement with its teacher on held-out real photographs: CommonCatalog val/test originals Spearman 0.53 -> 0.62 (Jaccard 0.10 -> 0.13), BlueNeg blue-corrupted 0.38 -> 0.47 (0.047 -> 0.066), with synthetic agreement unchanged or better (SDXL 0.71 -> 0.76, Z-Image 0.57 -> 0.71). Benchmark v1 val vs the identically trained control (student_style_fs_v2_sel): B filmsynth 0.557 vs 0.520, generator share 0.586 vs 0.595, subject probe 0.192 vs 0.263 (majority 0.202), own-source rank-1 0.174 vs 0.097. The gap to the teacher on real images is still large (Spearman 0.62 vs 0.72 nuisance ceiling), so domain shift of the distillation input explains part, not all, of the synthetic-to-real gap.
- Next: Keep the real-distil student as the candidate (not promoted); evaluate on the Commons analysis half when the set is in; E23c tests whether input diversity alone (SDXL outputs as distillation-only inputs) gives the same gain.

### E23c_student_sdxl_kd_only — Input diversity alone: SDXL train outputs as distillation-only inputs (no labels) give the same real-agreement gain as real photographs (if so, the gain is about input diversity, not real images)
- Status: done; wall 36.2 s; peak GPU 7655.0 MB (total on card).
- rank1_kd64 [proxy]: 0.1429
- fam_f1 [proxy]: 0.6577
- Decision rule: neutral: no credible real-film change
- Learned: SDXL train outputs as distillation-only inputs (same recipe as E23b but synthetic instead of real) do not reproduce the gain: held-out CommonCatalog agreement Spearman 0.55 (control 0.53, real-distil 0.62) and BlueNeg blue-corrupted 0.32 (control 0.38, real-distil 0.47). The E23b gain is specific to real photographs in the distillation input, not to input diversity.
- Next: Falsified as an alternative explanation. Real unlabeled photographs are the cheap lever for the student; more of them (CC originals are 534) is the next scaling question, not more synthesis.

### E24_scanner_effects — Scanner family (Noritsu / Frontier / flatbed) is recoverable from style embeddings of real film scans and drives same-look neighbours more than the film does (scanner confound)
- Status: done; wall 3.4 s; peak GPU 6257.0 MB (total on card).
- scanner_acc_hybrid_supcon [real]: 0.4566
- scanner_share_hybrid_supcon [real]: 0.3992
- Decision rule: neutral: no credible real-film change
- Learned: On 265 CommonCatalog film scans (138 uploaders; Noritsu 134, Fuji Frontier 93, flatbed/dedicated 26) the scanner family is barely recoverable: uploader-grouped probe accuracy 0.46 (hybrid SupCon), 0.47-0.48 (students), 0.44 (texture 12) against a 0.51 majority; only the 16 colour/tone statistics (0.59 [0.51, 0.67]) and raw hybrid (0.57) exceed it. Same-scanner share of the top-5 with the uploader excluded is at the prior for every embedding (0.38-0.42 vs 0.38). Photographer bias is small (same-uploader share 0.02-0.03 vs prior 0.007; DINOv2 content 0.055). The scanner is not the confound behind the real-scan gap in this population; the weak scanner signal sits in colour/tone, where the deterministic badges live.
- Next: No scanner correction needed for the learned embeddings; note the colour/tone scanner sensitivity in the badge documentation. The remaining real-film question is photographer/roll confounds, which the Commons set (multi-uploader stocks) can measure.

### E25_content_residualisation — Residualising hybrid features against DINOv2-global content (ridge fit on unlabeled train images) before the family SupCon lowers own-source rank-1 and subject leakage on val without lowering B filmsynth; a random-projection control does not
- Status: done; wall 5.7 s; peak GPU 6305.0 MB (total on card).
- B_fs_a10 [proxy]: 0.1317
- rank1_a10 [proxy]: 0.6872
- subj_a10 [proxy]: 0.1616
- C_share_a10 [proxy]: 0.4349
- Decision rule: neutral: no credible real-film change
- Learned: Regressing the hybrid features on DINOv2-global 'content' (ridge on unlabeled train images) and projecting the residual removes the treatment signal together with the content: B filmsynth 0.12-0.26 (vs 0.47), A2 0.2-0.4 (vs 1.7), and own-source rank-1 rises to 0.57-0.69 (vs 0.37); the random-projection control keeps most of the style (B 0.40, A2 1.05). DINOv2-global is not a content-only vector: it linearly predicts the style part of the hybrid as well, so linear content removal is the wrong tool.
- Next: Rejected. Content/appearance separation needs a content representation that is itself style-blind (e.g. same-source invariance across treatments), not the raw global token; parked.

### E26_layer_statistic_ablation — a (DINOv2 layer, patch statistic) choice exists that keeps L4+L8 mean/std treatment sensitivity, retrieves real same-stock scans at least as well and leaks less content; second-order (Gram) statistics of early layers are the classic style carrier
- Status: done; wall 32.6 s; peak GPU 7305.0 MB (total on card).
- beats [proxy]: 0.0
- best_real_same_stock [real]: 0.6085 CI [0.463, 0.723]
- baseline_real_same_stock [real]: 0.5759 CI [0.428, 0.698]
- best_real_val_B [synthetic]: 0.3367 CI [0.29, 0.383]
- best_real_rank1 [proxy]: 0.3231
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: beats=0.0000 < 0.5
- Reading the table (`reports/layer_stat_ablation.json`, 28 candidates, analysis data only): (1) the real-film
  condition could not be met by anything: BlueNeg blue-corrupted same-stock p@5 has chance 0.565 (218 of 297 frames
  are one stock) and every candidate sits at 0.55-0.61 with a roll CI of about +-0.13, so this partition is not a
  discriminating analysis metric and is retired for selection; (2) on the synthetic side, layer-3 mean/std is
  CI-separated above the L4+L8 baseline for cross-scene treatment retrieval (val B 0.572 [0.538, 0.603] vs 0.467
  [0.418, 0.514]) with less content leakage (own-source rank-1 0.262 vs 0.369, subject probe 0.212 vs 0.242); layers
  2-4 mean/std are all at or above the baseline, layers >= 9 are content-heavy (rank-1 up to 0.477 at L12);
  (3) Gram/covariance statistics (32-d projection) are worse than mean/std at every layer (B 0.25-0.41): the classic
  "style = second-order statistics" carrier does not transfer here, at least at this projection size. The decision
  stays `rejected` under the pre-registered rule; the L3 observation is a new hypothesis (E27), not a result.
- Next: if supported: retrain the real-distil student (E23b recipe) on the winning target and evaluate on CommonCatalog film scans; if rejected: keep L4+L8 mean/std and move to augmentation-based invariance (E27)

### D1_commons_film_download — Data pipeline incident and recovery (Wikimedia Commons acquisition)
- First run (18:16-21:26): 244 files in 3.1 h, no stdout (the runner captures it), manifest only written at exit.
  Diagnosis from the runner log: all 168 HTTP 429s were on original-file URLs (`Retry-After: 600`); 16 ten-minute
  sleeps account for 2.7 h. Thumbnails of large files were fine. Process was alive (`hrtimer_nanosleep`), not deadlocked.
- Second cause found while fixing: files whose requested width rounds up to the original get the `thumbnail_unscaled`
  alias (= the original, throttled), and the thumbnail service rejects every non-standard width with HTTP 400
  (standard: 20...330, 500, 960, 1280, 1920, 3840 px).
- Fix: `research/downloader.py` (resumable per-file state, atomic writes, sha256 verification, bounded backoff,
  Retry-After capped at 120 s, exit after 20 consecutive rate-limit waits, no-response stall timeout, wall-clock cap,
  heartbeat + status file, manifest checkpoints) and standard-width renditions only. Existing 244 files adopted after
  re-validation; 22 more fetched in a 2.5-minute smoke run; 34 bogus failures from the wrong URL scheme reset.
- Current blocker: after the incident the client is throttled on every rendition (`Retry-After: 10`, Varnish
  front). The acquisition task retries with a 30-minute doubling backoff (orchestrator); 266 of 5,743 items stored,
  5,477 pending, 7 without a usable rendition (<= 250 px). Dataset state `retryable_failure` (266 < 1,000 rows);
  evaluation waits on `dataset:commons_film` and runs automatically when it is ready.

### E27a_student_l3_teacher_interim — a real-distil student (E23b recipe) distilled from a layer-3 mean/std hybrid teacher keeps the synthetic family signal and content invariance of the L4+L8 real-distil student (interim, synthetic/proxy only; the real-film decision is E27 on Commons)
- Status: done; wall 54.7 s; peak GPU 7649.0 MB (total on card).
- rank1_kd64 [proxy]: 0.2095 (baseline 0.181, regressed)
- fam_f1 [proxy]: 0.6163 (baseline 0.6501, regressed)
- treated_auroc [synthetic]: 0.8292
- Decision rule: neutral: no credible real-film change
- Learned: decision inconclusive: neutral: no credible real-film change; rank1_kd64=0.2095 <= 0.231
- Teacher-side reading (`reports/style_benchmark_val_e27.json`, val only): the L3 mean/std hybrid SupCon teacher
  has B fs 0.572 vs 0.467 for L4+L8, A 0.934 vs 0.907, own-source rank-1 0.262 vs 0.369, subject probe 0.212 vs
  0.242 - but a higher same-generator share (0.594 vs 0.541; prior 0.396) and no gain on the held-out generators
  (B sdxl 0.324, B zimage 0.364, both at chance). L2-4 concatenated sits between the two. So early-layer statistics
  buy within-generator treatment sensitivity and content invariance at the price of more generator fingerprint; the
  student distilled from it loses a little family F1 (0.616 vs 0.650) and content invariance (rank-1 0.21 vs 0.18).
  Nothing here is a real-film result; `student_style_l3_real` is an eligible candidate of the evaluation pipeline and
  E27 decides on the Commons analysis half.
- Next: checkpoint student_style_l3_real becomes an eligible candidate of research.eval_pipeline; E27 decides on the Commons analysis half

### E28_l3_shortcut_audit — the layer-3 teacher's cross-scene treatment gain (E26/E27) is not a filmsynth-pipeline shortcut and the L3 features do not recover scanner family or uploader on real film scans more than L4+L8: under a generator-swapped SupCon projection (trained on sdxl families) the L3 advantage on held-out filmsynth persists, and on CommonCatalog film-scan originals (train+val) the L3 teacher's scanner-probe accuracy is within +0.05 and its same-uploader top-5 share within +0.02 of the L4+L8 teacher
- Status: done; wall 5.8 s; peak GPU 6284.0 MB (total on card).
- passes [proxy]: 0.0
- swap_gain [synthetic]: 0.0116
- scanner_excess [proxy]: -0.034
- uploader_excess [proxy]: -0.0046
- Decision rule: neutral: no credible real-film change
- Readings (reports/l3_shortcut_audit.json, analysis data only). C1 generator swap: with the family SupCon projection trained on the 4 sdxl families (280 rows) instead of filmsynth, both teachers collapse on held-out filmsynth val (L3 0.203 [0.183, 0.221] vs L4+L8 0.192 [0.166, 0.223]; chance 0.11) and are equal on zimage (0.413 both): `swap_status = vanishes`. The L3 advantage seen with filmsynth-trained projections (0.572 vs 0.467) is therefore tied to training the projection on the filmsynth pipeline, and the projection itself is generator-specific for every layer choice. C2/C3 on 265 CommonCatalog film-scan originals (138 uploaders): L3 scanner-probe accuracy 0.423 [0.345, 0.501] vs 0.457 (majority 0.506) and same-uploader share 0.014 vs 0.019, so L3 is not a scanner or photographer shortcut; explicit28 alone recovers the scanner best (0.581), consistent with E24. Students: style_l3_real scanner 0.426 / uploader 0.017 vs style_fs_v2_real 0.468 / 0.024. Decision: rejected under the pre-registered rule (passes = 0 because the swap gain vanished); the teacher target stays L4+L8; L3 remains only an E27 candidate on real film. No promotion. Cost 5.8 s GPU.
- Learned: decision rejected: falsified: passes=0.0000 < 1
- Next: not falsified: L3 stays a candidate for E27 on Commons (no promotion from this run); falsified: drop the L3 teacher target, keep L4+L8, record which shortcut

### E30_joint_generator_supcon — a family SupCon projection trained on the union of the filmsynth and sdxl TRAIN families (13 generator-namespaced labels) transfers to the held-out generator (zimage) better than the filmsynth-only projection, without more content leakage or generator fingerprint
- Status: done; wall 4.6 s; peak GPU 6287.0 MB (total on card).
- supported [proxy]: 0.0
- zimage_gain [synthetic]: 0.0089
- filmsynth_change [synthetic]: -0.01
- Decision rule: neutral: no credible real-film change
- Readings (reports/joint_generator_supcon.json, val split only). Hybrid L4+L8 projections: filmsynth-trained B fs 0.467 / sdxl 0.347 / zimage 0.342; sdxl-trained 0.192 / 0.436 / 0.413; joint 0.457 / 0.396 / 0.351 [0.324, 0.378], generator share 0.541 -> 0.635, rank-1 0.369 -> 0.344. L3 features behave the same (joint 0.530 / 0.444 / 0.356, share 0.653). The joint projection gains on the second training pipeline and nothing on the held-out generator while the generator fingerprint grows: rejected (zimage CI overlaps the baseline; share excess 0.094 > 0.02). Cost 4.6 s GPU.
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: add hybrid_joint_supcon as a fixed candidate in the real-film evaluation (E27/E27b decide on real film); rejected: keep the single-generator projection and record that cross-pipeline families do not merge linearly

### E27b_early_layer_teacher_commoncatalog — the L3 teacher judged on real film (CommonCatalog film scans with uploader stock tags)
- Run: research.eval_pipeline on the analysis half of `commoncatalog_film` (487 scans, 175 uploaders, 71 uploader-claimed stocks; analysis half 247 scans / 87 uploaders; snapshot 28d45d1d; 2026-11-08 00:12). Candidates: fixed benchmark set, six eligible students, and the pre-registered `hybrid_l3_supcon_family`.
- Primary metric (pre-registered: same-stock p@5, uploader excluded, speed-level labels, 185 queries from 77 uploaders): baseline 0.146 [0.119, 0.184], L3 teacher 0.137 [0.105, 0.176], chance 0.125; students 0.124-0.160; raw DINOv2-global 0.157 but rejected for photographer leakage (uploader share 0.134 vs prior 0.030); colour_tone16 and student_style_both also rejected for leakage. No candidate is CI-separated from chance.
- Decision: **inconclusive** for E27b and for every student (no promotion). The instrument is the problem on this snapshot, not a specific candidate: with 54 stocks on the analysis half and only 20 of them held by two or more uploaders, uploader-excluded same-stock retrieval has too few positives.
- Post-hoc diagnostics (recorded as observations, not decisions): with stock *line* labels (brand + line, speed dropped; 217 queries) the baseline reaches 0.166 [0.139, 0.193] vs chance 0.129 and the students 0.166-0.195 (student_style_fs_v2_kdsdxl 0.195 [0.165, 0.231]); film-type retrieval (colour negative / B&W / reversal) is clearly above chance for every learned embedding (0.34-0.41 vs 0.264) and at chance for raw DINOv2-global and texture12, though B&W-vs-colour is a trivial part of that signal. Scanner probes sit at the majority rate (0.38-0.51 vs 0.518).
- Next: a larger snapshot (700 more shards queued: `data_commoncatalog_film_scan2` -> acquire -> re-evaluation) to raise uploaders-per-stock; the pre-registered rule is unchanged; the line-level metric is pre-registered now as a *secondary* reading for the next snapshot, never as a replacement.

### E29_curated_cross_source_robustness — style embeddings on CC0 museum photographs (aic_photographs)
- Run: scripts/eval_curated_photos.py on the analysis half (306 prints, 111 artists; 593 images, 207 artists in the pilot; 1840s-1890s albumen, daguerreotype, salted paper, photogravure ...), 2026-11-08 00:50; features 64 s GPU.
- Process p@5, artist excluded (chance 0.389): hybrid_supcon_family 0.532 [0.453, 0.596]; raw_hybrid 0.519; student_fs_v2 0.505; student_style_fs 0.506; student_style_fs_v2_sel 0.504; student_style_both 0.482; student_style_fs_v2_kdsdxl 0.468 [0.386, 0.541]; student_style_fs_v2_real 0.446 [0.365, 0.517]; student_style_l3_real 0.445 [0.366, 0.509]; raw_dino_global 0.580 and colour_tone16 0.617 are the highest but leak the artist (top-5 artist share 0.219 / 0.212 vs prior 0.017; baseline 0.103; students 0.04-0.07).
- Teacher agreement (Spearman of pairwise cosines vs hybrid_supcon_family): fs_v2 0.52, both 0.55, fs_v2_real 0.55, kdsdxl 0.51, sel 0.53, fs 0.45, l3_real 0.41.
- Decision under the pre-registered rule: raw_dino_global and colour_tone16 falsified (leakage); the three real-distilled students (fs_v2_real, l3_real, kdsdxl) falsified (process p@5 lower CI below chance); the hybrid teacher and the synthetic-only students pass. Overall **inconclusive** for the hypothesis (mixed), and a warning: real-image distillation (E23b) did not buy cross-source robustness on this third domain; the process signal here is largely colour/tone and era, which the treatment-family projection partly discards. No promotion; aic_photographs stays evaluation-only for now.

### E27b, second snapshot (905 scans) — same-stock retrieval from uploader tags stays at chance for every embedding
- Run: research.eval_pipeline on the analysis half of the 905-row `commoncatalog_film` snapshot (c2b088c3; 445 scans, 124 uploaders, 389 stock-labelled rows, 68 stocks, 34 with >= 2 uploaders; 356 queries), 2026-11-08 02:30.
- Chance: same-stock 0.044, stock-line 0.066, same-type 0.264 (first snapshot); uploader-share prior 0.03.
  raw_dino_global              stock 0.071 [0.047, 0.1]  line 0.085 [0.065, 0.109]  type 0.343  uploader share 0.140
  raw_hybrid                   stock 0.043 [0.031, 0.057]  line 0.065 [0.054, 0.077]  type 0.392  uploader share 0.041
  hybrid_supcon_family         stock 0.051 [0.036, 0.069]  line 0.078 [0.065, 0.091]  type 0.467  uploader share 0.052
  student_fs_v2_64             stock 0.049 [0.037, 0.063]  line 0.087 [0.073, 0.105]  type 0.467  uploader share 0.058
  colour_tone16                stock 0.061 [0.047, 0.079]  line 0.086 [0.069, 0.104]  type 0.458  uploader share 0.095
  texture12                    stock 0.046 [0.033, 0.063]  line 0.062 [0.051, 0.076]  type 0.349  uploader share 0.028
  hybrid_l3_supcon_family      stock 0.055 [0.042, 0.071]  line 0.089 [0.074, 0.104]  type 0.441  uploader share 0.061
  student_style_both_64        stock 0.053 [0.041, 0.07]  line 0.086 [0.069, 0.104]  type 0.458  uploader share 0.058
  student_style_fs_64          stock 0.064 [0.048, 0.085]  line 0.101 [0.087, 0.118]  type 0.492  uploader share 0.058
  student_style_fs_v2_kdsdxl_64 stock 0.056 [0.043, 0.071]  line 0.091 [0.078, 0.104]  type 0.470  uploader share 0.069
  student_style_fs_v2_real_64  stock 0.058 [0.045, 0.073]  line 0.092 [0.077, 0.107]  type 0.461  uploader share 0.053
  student_style_fs_v2_sel_64   stock 0.054 [0.04, 0.07]  line 0.084 [0.072, 0.099]  type 0.473  uploader share 0.060
  student_style_l3_real_64     stock 0.045 [0.033, 0.061]  line 0.066 [0.056, 0.081]  type 0.441  uploader share 0.058
- Decisions: raw_dino_global and colour_tone16 rejected for photographer leakage (0.140 / 0.095 vs prior 0.03); everything else inconclusive; E27b inconclusive again (L3 0.055 vs 0.051). Nothing is promoted. The only readings CI-separated from chance are the stock-line metric for the SupCon hybrid (0.078 vs 0.066, barely) and several students (0.084-0.101), and film-type retrieval (0.44-0.49), which the monochrome cue inflates.
- Reading: across photographers, uploader-claimed film stock is not recoverable from any of these embeddings on Flickr scans at a useful level; BlueNeg's 0.25 (one photographer, 27 rolls) was within-photographer structure. Either the labels are too noisy (tags, speeds, expired/pushed film, lab scanning dominates) or the embeddings carry no stock signal that survives the photographer/scanner change. The next test (E31) removes the trivial monochrome cue and asks whether colour negative vs colour reversal is separable across uploaders at all.

### E31_colour_type_retrieval — on real Flickr film scans a film-process signal survives the change of photographer once the monochrome cue is removed: with only colour scans (uploader-claimed colour negative vs colour reversal), the hybrid teacher retrieves the colour type above its class prior with the query's uploader excluded and without photographer leakage
- Status: done; wall 3.1 s; peak GPU 6261.0 MB (total on card).
- supported [proxy]: 0.0
- baseline_p5 [real]: 0.6705
- chance [real]: 0.6417
- Decision rule: neutral: no credible real-film change
- Readings (reports/colour_type_retrieval.json; 295 colour scans, 93 uploaders, 228 negative / 67 reversal, chance 0.642, uploader prior 0.025): raw_dino_global 0.648 [0.577, 0.708] (uploader share 0.155); hybrid_supcon_family 0.670 [0.597, 0.726] (uploader share 0.063); student_fs_v2_64 0.719 [0.662, 0.763] (uploader share 0.057); colour_tone16 0.715 [0.653, 0.765] (uploader share 0.098); texture12 0.647 [0.561, 0.714] (uploader share 0.035); student_style_both_64 0.719 [0.664, 0.766] (uploader share 0.054); student_style_fs_64 0.738 [0.687, 0.782] (uploader share 0.056); student_style_fs_v2_kdsdxl_64 0.734 [0.675, 0.786] (uploader share 0.073); student_style_fs_v2_real_64 0.713 [0.643, 0.766] (uploader share 0.052); student_style_fs_v2_sel_64 0.729 [0.673, 0.776] (uploader share 0.067); student_style_l3_real_64 0.686 [0.625, 0.742] (uploader share 0.060). The hybrid teacher's lower CI (0.597) is below chance and its uploader share exceeds the prior by 0.038: rejected under the pre-registered rule. Observation only: the synthetic-only students (0.72-0.74, lower CIs 0.66-0.69) and colour_tone16 (0.715) retrieve the colour type above chance, so a weak cross-photographer reversal-vs-negative signal exists and is mostly global colour/tone; it comes with photographer leakage of 0.03-0.07 over the prior. Cost 1.8 s CPU.
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: film-process (not stock) becomes the real-film retrieval target for the next benchmark version; falsified: record that these embeddings carry no cross-photographer film-process signal on Flickr scans and move the real-film claim to within-photographer / within-roll structure only

### E32_e29_audit — the hybrid teacher's museum-print process signal (E29: 0.532 vs 0.389) is print rendering rather than format: it survives the removal of stereo cards and the matching of the candidate bank to the query's decade (lower CI above the within-decade prior); and the real-distilled students' loss of it is anisotropy on this out-of-domain set (label-free whitening restores a lower CI above chance) rather than information loss (whitening does not help)
- Status: done; wall 6.2 s; peak GPU 6257.0 MB (total on card).
- teacher_survives [proxy]: 1.0
- teacher_decade_matched [real]: 0.4828
- teacher_decade_matched_chance [real]: 0.358
- students_whitened_recover [proxy]: 6.0
- Decision rule: neutral: no credible real-film change
- Learned: decision supported: not falsified: teacher_survives=1.0000 >= 1
- Next: teacher survives: the process signal is rendering, not format; next transfer experiment targets the component that carries it (ablation) with a projection head on frozen DINOv2-Small features, judged on commoncatalog_film and the museum set with artist/uploader exclusion; falsified: record E29 as a format/era cue, retire process p@5 as a selection metric and keep aic_photographs for authenticity/robustness checks only

### E33_frozen_head_choice — a label-free whitened PCA-64 of the frozen DINOv2-Small L4+L8 statistics, fitted on independent unlabeled real originals, carries at least as much real-photograph signal as the synthetic-family SupCon projection: museum process p@5 (stereo removed, decade matched) lower CI above chance, commoncatalog_film stock-line p@5 not below the projection's, uploader- and artist-share excess <= 0.02
- Status: done; wall 8.6 s; peak GPU 6257.0 MB (total on card).
- supported [proxy]: 0.0
- candidate_museum_p5 [real]: 0.5646
- candidate_stock_line_p5 [real]: 0.1029
- candidate_synth_family_p5 [proxy]: 0.2686
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: the whitened head becomes the distillation target candidate for a student retrained with the E23b recipe (E34), judged on the same real metrics plus the synthetic validation cost; falsified: keep the SupCon projection, test a projection head trained with real-photo structure instead (E34 alternative)

### E34_head_anticollapse — the SupCon projection's rank collapse (effective rank ~4) is what loses real-photograph information, and its low photographer leakage is not merely a by-product of that collapse: a SupCon head trained on the same synthetic family labels with VICReg variance/covariance regularisation (lambda 1) reaches effective rank >= 16, museum process p@5 (stereo removed, decade matched) and commoncatalog_film stock-line p@5 both with lower CI above chance, while its uploader share, artist share (<= +0.02) and scanner-probe accuracy (<= +0.05) stay within the guards relative to the current projection
- Status: done; wall 7.3 s; peak GPU 6277.0 MB (total on card).
- supported [proxy]: 0.0
- primary_eff_rank [proxy]: 9.95
- primary_stock_line_p5 [real]: 0.0799
- primary_museum_p5 [real]: 0.4869
- primary_uploader_share [real]: 0.072
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: retrain the real-distil student (E23b recipe) against the anti-collapse head as its distillation target (E35) and judge it on the same real metrics; falsified with leakage: information and photographer invariance trade off in the frozen features, so the next head needs an explicit photographer-invariance term on real data with several images per uploader (feature pass over commoncatalog_ccby); falsified on rank/information: collapse is not the mechanism, revisit the distillation loss

### E35_head_photographer_invariant — photographer identity and real-photograph film/process information occupy separable directions of the frozen DINOv2-Small L4+L8 statistics: removing the top-32 between-uploader LDA directions (estimated on independent training-allowed CC-BY originals with several images per uploader) before a label-free whitened PCA-64 head keeps the stock-line and museum-process lower CIs above chance while bringing uploader and artist share within +0.02 of the current projection and the scanner probe within +0.05
- Status: done; wall 15.1 s; peak GPU 6259.0 MB (total on card).
- supported [proxy]: 0.0
- primary_stock_line_p5 [real]: 0.1
- primary_museum_p5 [real]: 0.5727
- primary_uploader_share [real]: 0.1553
- primary_artist_share [real]: 0.2529
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: the removed+whitened head is the new distillation target candidate (student retrain, E36); falsified because leakage stays: photographer identity is entangled with the film signal in frozen features, so the remaining routes are (a) an uploader-adversarial head trained on more multi-image uploaders or (b) accepting photographer sensitivity as a product feature and measuring it content-controlled; falsified because information drops: the stock and process cues live in the same directions as photographer identity on these sources

### E36_student_whitening — the real-distilled student's museum-print failure (E29) is geometry, not missing information: a 64x64 PCA-whitening fitted on its embeddings of independent training-allowed real photographs (commoncatalog_ccby train originals) and applied unchanged restores a museum process p@5 lower CI above chance, does not lower its stock-line p@5, and keeps uploader share, artist share (<= +0.02) and scanner probe (<= +0.05) within the guards relative to the raw student
- Status: done; wall 8.3 s; peak GPU 5740.0 MB (total on card).
- supported [proxy]: 0.0
- primary_whitened_museum_p5 [real]: 0.5273
- primary_raw_museum_p5 [real]: 0.4192
- primary_whitened_stock_line_p5 [real]: 0.0787
- students_recovered_indep [real]: 7.0
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: ship-candidate post-processing (a fixed 64x64 matrix) recorded for the student; verify once on the held-out halves at the end of the session (--final) together with the synthetic benchmark; falsified: the collapse must be fixed at training time (non-collapsed target or anti-collapse term in the student loss, E37)

### E37_within_photographer_stock — with the photographer held fixed (same uploader, scans from other days only), the hybrid teacher ranks a scan's same-stock-line scans above its other-line scans: within-uploader pairwise AUC lower CI above 0.5 and a within-uploader permutation p-value below 0.05 on the commoncatalog_film analysis half
- Status: done; wall 81.2 s; peak GPU 6257.0 MB (total on card).
- supported [proxy]: 1.0
- teacher_auc_different_day [real]: 0.6405
- teacher_perm_p [real]: 0.0
- n_significant_different_day [real]: 9.0
- Decision rule: neutral: no credible real-film change
- Learned: decision supported: not falsified: supported=1.0000 >= 1
- Next: supported: the within-library stock-similarity claim is kept (BlueNeg + CommonCatalog), the cross-photographer claim is dropped from the model card, and the student is judged on the within-uploader AUC too; falsified: no embedding separates stocks even within a photographer on Flickr scans, so stock similarity leaves the product roadmap until owner-recorded stocks from many photographers exist

### Session 5 synthesis (2026-11-08, E32-E37): what the E29 teacher learned and why the students lost it

All six runs use cached features of the analysis halves only (no test half, no BlueNeg); E32-E37 cost under a GPU-minute
in total. Reports: `reports/e29_audit.json`, `frozen_head_choice.json`, `head_anticollapse.json`,
`head_photographer_invariant.json`, `student_whitening.json`, `within_photographer_stock.json`.

1. **Format inflates E29 but does not explain it (E32, supported).** 76 of the 306 analysis prints are stereo cards
   (aspect ratio ~2:1, all albumen); a 2-d format baseline (log aspect ratio, stereo flag) alone retrieves the process
   at 0.479 vs chance 0.389. With stereo cards removed the hybrid teacher is 0.410 [0.343, 0.479] vs 0.266; with the
   candidate bank also restricted to the query's decade it is 0.483 [0.401, 0.560] vs 0.358. The rendering signal is
   real. Its carriers are the colour-tone statistics (0.520 no-stereo) and the DINOv2 L4+L8 statistics (0.489); the
   family SupCon projection keeps less of it (0.410) than its own inputs.
2. **The distillation target is collapsed, and the students inherit the collapse (E32).** The SupCon projection has an
   effective rank of 3.9 on Flickr film scans and 3.3 on museum prints; the six students' 64-d outputs have 3.1-3.2 and
   2.1-2.6, with a mean pairwise cosine of 0.63-0.79 on the monochrome prints (0.14-0.28 on Flickr). Label-free PCA
   whitening fitted on the evaluated half restores every student's museum p@5 to 0.50-0.54 with lower CIs above chance,
   and artist-grouped ridge probes recover the teacher projection from the students at R^2 0.37-0.43. The information is
   present; the metric geometry is degenerate. This answers E29's teacher-student gap: not representation collapse
   in the student alone, but a collapsed target reproduced faithfully.
3. **Every head that keeps the information also clusters by photographer (E33-E36, all falsified on the leakage
   guards).** Label-free whitened PCA heads on the frozen L4+L8 statistics (fit on independent CC-BY originals) reach
   stock-line p@5 0.103 [0.080, 0.128] vs chance 0.066 with the uploader excluded and museum 0.565 vs 0.358, but their
   uploader share is 0.14-0.17 (projection 0.06, prior 0.03) and artist share 0.21-0.28 (projection 0.10). Raising the
   projection's rank with VICReg moves both together (rank 3.8 -> 10: stock 0.071 -> 0.093, artist share 0.10 -> 0.19;
   E34). Removing the top-32 between-uploader LDA directions estimated on 92 multi-image CC-BY uploaders changes nothing
   that removing 32 random directions does not (E35: uploader share 0.155 vs 0.146 random). Whitening the students with
   an independent fit restores all seven museum lower CIs (0.48-0.56) and raises artist share to 0.11-0.18 (E36).
   The scanner probe never rises above its 0.44 majority in any of these runs, so the shared factor is not the scanner.
4. **Within a photographer the embeddings do separate stock lines (E37, supported).** Among the 15 uploaders with
   two or more stock lines (182 scans), comparing each scan only with the same uploader's scans from other days, the
   hybrid teacher's pairwise AUC is 0.640 [0.563, 0.734] against a within-uploader permutation null of [0.454, 0.547]
   (p < 0.001); colour-tone statistics alone reach 0.681 and the real-distilled student 0.673. Restricted to colour
   scans (8 uploaders, 64 queries) the teacher drops to 0.555 [0.473, 0.682], p = 0.06: much of the within-photographer
   separation is the monochrome-vs-colour cue, and the colour-only remainder is underpowered on this source.

Reading across the six: the frozen DINOv2-Small statistics carry a real photographic-rendering signal (print process,
coarse stock family, photographer's look) in the same directions, and no linear head or label-free post-processing
tested here separates them. The collapsed family projection is the one head with low photographer clustering, and it
has that property because it discards almost everything. For the product this means photographer-style similarity and
film-look similarity are not separable features of this representation; they should be presented as one
"similar look" signal with the abstention rule, and stock similarity claimed only within a user's own library
(E37) or for one photographer's rolls (BlueNeg). E38 tests whether a non-collapsed target gives the student the
same signals at an acceptable synthetic cost; its result is recorded below.

### E38_student_targets — the real-distilled student loses the teacher's museum-print signal because its distillation target is the collapsed SupCon projection (E32): a student distilled with the same recipe from the SupCon + VICReg (lambda 3) head keeps the real-photograph signals (museum p@5 lower CI above chance, within-uploader different-day stock AUC lower CI above 0.5, stock-line p@5 within 0.01 of the baseline student) at an effective rank >= 8, with the scanner probe within +0.05 and the synthetic family p@5 within 0.10 of the baseline student
- Status: done; wall 186.4 s; peak GPU 7647.0 MB (total on card).
- supported [proxy]: 0.0
- primary_museum_p5 [real]: 0.4808
- baseline_museum_p5 [real]: 0.4192
- primary_within_auc [real]: 0.7179
- primary_stock_line_p5 [real]: 0.0839
- primary_synth_family_p5 [proxy]: 0.6667
- primary_eff_rank [proxy]: 5.9
- Decision rule: neutral: no credible real-film change
- Learned: decision rejected: falsified: supported=0.0000 < 1
- Next: supported: the vic3 student becomes the promotion candidate; one final evaluation on the held-out halves and the synthetic benchmark test at the end of the session, latency check, model card update; falsified: record which signal was lost; if the seed control moves the real metrics as much as the target change, the real matrix needs more photographers before any student decision

## Decisions taken from these runs (autonomous session 2026-11-07, E8-E18) and session 2 (E19, B1, E10)

9. Benchmark v1 (`docs/BENCHMARK.md`) is the selection instrument: val for selection, test spent once (B1). The
   research embedding stays hybrid + family SupCon (only candidate above the DINOv2 reference on real scans); the
   abstention rule is top-1 similarity at 50 % coverage.
10. Diffusion img2img treatment ids are not style labels at strength 0.35 or 0.5 (E19 diagnostic, E10): they change
    each image differently without a shared signature. Diffusion sets remain controls for generator fingerprint.
11. Making the projection content-blind by loss design (E19 hard negatives, decorrelation) does not work on 70
    sources; the remaining content and generator signal is a data limitation, not an objective choice.

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
12. Session 3 (data pipeline, E22-E26): the student's real-scan failure is a distillation input-domain shift
   (E22); real unlabeled inputs fix part of it (E23b, kept as `student_style_fs_v2_real`, not promoted until the
   Commons evaluation); SDXL inputs do not (E23c). Scanner identity is not the retrieved signal (E24). Linear content
   residualisation is rejected (E25). No layer/statistic beats L4+L8 under the pre-registered three-part rule (E26),
   but BlueNeg blue-corrupted is retired as a selection metric (chance 0.565) and early-layer mean/std (L2-L4) is the
   open hypothesis E27; Gram statistics are dropped.
13. Acquisition and evaluation are now separate pipelines (`docs/PIPELINES.md`): the Commons download is resumable
   and rate-limit aware, dataset readiness is persistent, and the real-film evaluation of every eligible student
   runs automatically on the analysis half when `commons_film` is ready; the test half runs once, with `--final`.

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
