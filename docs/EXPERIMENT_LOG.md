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

## Decisions taken from these runs

1. Grain randomisation (film-synth-v2) is the fix that worked for cross-generator leakage; standardisation and the
   VAE control helped little but stay on as hygiene. Scale beyond 100 sources only with v2-style presets, and add a
   third generator (Z-Image-Turbo) before trusting any cross-generator number.
1b. Authenticity and aesthetics are separate heads: the grain/texture axis predicts film capture; colour/tone
   treatment is the transferable aesthetics signal. Never report one as the other.
2. Keep the EXIF capture-type head as the film-vs-digital signal in production; add the synthetic-trained style
   embedding for retrieval only, labelled experimental.
3. Report BlueNeg numbers as "film scans by one photographer, 1990s consumer stocks"; they are not a general film
   benchmark.
