# Learning photographic aesthetics with little labeling: research notes and proof of concept

Status 2026-11-07. Scope: the main FilmLook model (DINOv2-small family). The Mini Tool, the production site and the
evaluation repository were not touched. All numbers come from scripts in this repository, run on the development
workstation (**RTX 4090, shared**; the owner's RTX 4070 was not available for measurement).

## 1. Method comparison

| method | fits a small dataset (967 photos)? | fits DINOv2-small? | targets aesthetics rather than content? | GPU cost | verdict |
|---|---|---|---|---|---|
| SimCLR / BYOL / DINO self-distillation | weak: needs large batches or many images | redundant: DINOv2 already is DINO-style SSL | **no**: their colour-jitter and blur augmentations teach *invariance* to exactly the properties FilmLook must see | high | not used as-is |
| VICReg / Barlow Twins | better with small batches | yes, as an adapter objective | only if the "views" keep aesthetics fixed and vary content, which needs a style-preserving augmentation family | medium | candidate for stage 4 (real-photo adaptation) |
| Supervised contrastive (SupCon) on synthetic families | yes | yes, as a projection on frozen features | yes: positives share a rendering, not a scene | tiny | **tested: works (C below)** |
| Transformation prediction (classify or regress applied parameters) | yes | yes, linear heads | yes, but risks learning generator fingerprints | tiny | **tested (A, B)** |
| Pairwise ranking of relative attributes | yes; labels come free from parameter differences | yes | yes, and invariant to the scene by construction | tiny | **tested: best real transfer (section 3.4)** |
| Multi-task attribute heads | yes | yes | yes | tiny | used |
| Teacher-student distillation | yes | DINOv2 → MobileNetV3 | inherits the teacher's content bias | small | **tested: no gain** (`reports/student_kd.json`) |
| Content/style disentanglement | partially; needs paired data, which synthesis provides | yes: global embedding = content, intermediate statistics = style | yes | small | **tested implicitly** (dino_global vs dino_style) |
| Domain randomisation (nuisance) | yes | yes | prevents shortcuts | none | used in generation |
| Procedural scenes / object compositing / relighting | uncertain | — | composition yes, but compositing artifacts are a strong shortcut | high | **not done** (see section 5) |

Selected for the proof of concept: transformation prediction, pairwise ranking, SupCon projection, and the
global-vs-intermediate disentanglement probe, all on cached frozen features. Fine-tuning was limited to a MobileNetV3
check.

## 2. Film simulation engine (`filmlook_ml.filmsynth`, `configs/film_synth_v1.yaml`)

- **Deterministic:** NumPy and OpenCV; the same image, parameters and seed give identical bytes. Regenerating all 800
  variants reproduced the same sha256 set.
- **Parameters:** 20, each with a documented scale:
  - white balance and tint;
  - saturation;
  - contrast (S-curve or compression);
  - black point, shadow lift, highlight shoulder;
  - split toning (shadow and highlight hues);
  - local contrast, softness;
  - halation, bloom, vignette;
  - luma and chroma grain with size, strength and midtone weighting, spatially correlated through a blur of the noise
    field;
  - a monochrome mix with randomised filter weights.
- **9 neutral families** (`identity_v1`, `warm_negative_v1`, `neutral_finegrain_negative_v1`, `vivid_reversal_v1`,
  `cool_pastel_v1`, `tungsten_cinematic_v1`, `high_contrast_bw_v1`, `finegrain_bw_v1`, `grainy_highspeed_mono_v1`).
  - Each family is a set of overlapping parameter *ranges*.
  - Database aesthetic ids are stored only as `approximate_references`.
  - Ranges are motivated by the database's evidence: slide films have the highest saturation and contrast ratings
    and the finest grain; B&W grain rises with ISO (1.64 → 2.73 → 4.0); plus the app's character text.
- **Not done: physical calibration.** The FilmLook database has no characteristic curves, spectral sensitivities or
  RMS granularity, only descriptive notes and subjective 1–5 ratings. Manufacturer data sheets (Kodak and Ilford
  publish density curves, Print Grain Index or RMS granularity) were **not** fetched or digitised in this pass.
  Every parameter is an approximate synthetic setting, and no family claims a stock identity.
- **Nuisance:** JPEG quality 80–95, ±0.25 EV exposure, 0–3 % crop and the resampler are drawn independently of the
  family and applied to every output, including identity.

## 3. Proof of concept: 100 colour sources × 8 variants (`scripts/synth_generate.py`, `synth_experiment.py`, `synth_probes.py`)

Splits follow the existing uploader-grouped splits (70 / 15 / 15 sources); every variant stays with its source (checked).
Feature extraction took 11 s with 1.05 GB peak GPU memory; the whole study took about 8 minutes.

### 3.1 Family classification (test = 15 unseen sources, 120 variants; macro-F1)

| representation | train | val | **test** |
|---|---|---|---|
| explicit statistics (28) | 0.73 | 0.58 | 0.58 |
| frozen DINOv2 global (CLS + mean) | 1.00 | 0.31 | **0.31**: memorises sources |
| frozen DINOv2 intermediate statistics (layers 4 + 8, mean + std of patch tokens) | 1.00 | 0.60 | **0.72** |
| hybrid: explicit + global | 1.00 | 0.56 | 0.52 |
| hybrid_style: explicit + intermediate | 1.00 | 0.65 | **0.73** |
| fine-tuned MobileNetV3-Small (5 s, 1.27 GB GPU) | 0.93 | 0.72 | 0.62 |

Errors in hybrid_style are concentrated among the subtle colour-negative families: cool_pastel, neutral_finegrain,
tungsten and identity. The B&W families score F1 0.91–0.96.

### 3.2 Parameter regression (ridge; test MAE vs predicting the mean; R²)

hybrid_style:
- grain 0.083 vs 0.205 (R² 0.84);
- contrast 0.18 vs 0.22 (0.37);
- saturation 0.16 vs 0.19 (0.37);
- temperature 0.21 vs 0.24 (0.25);
- shadow lift 0.11 vs 0.14 (0.32);
- highlight compression 0.12 vs 0.16 (0.31).

Grain is learned well; the colour and tone amounts only partly.

### 3.3 Style retrieval and source memorisation (test variants; family precision@5 among other sources; chance 0.11)

| representation | precision@5 | nearest neighbour is the same source |
|---|---|---|
| DINOv2 global | 0.14 | **100 %** |
| DINOv2 intermediate | 0.23 | 100 % |
| explicit | 0.18 | 59 % |
| hybrid_style | 0.18 | 68 % |
| **SupCon projection on hybrid_style** (trained on train sources) | **0.50** | **23 %** |
| fine-tuned MobileNetV3 features | **0.61** | **18 %** |

**Learned metric objectives are what turn content-dominated features into style-dominated ones.** Raw frozen features
retrieve the same photo.

### 3.4 Pairwise relative attributes (the strongest real-photo signal)

The ranker is trained on synthetic same-source pairs and tested on (a) synthetic test pairs and (b) **real pairs from
the dataset's independently implemented edit pipeline** (`datasets/edits.py`, different code) on photos never used as
sources. Real pairs are judged by the sign of clear measured-statistic differences.

| hybrid_style | warmer | more contrast | grainier | more saturated | softer highlights |
|---|---|---|---|---|---|
| synthetic test pairs | 0.96 | 0.99 | 1.00 | 0.96 | 0.91 |
| real edit pairs (n = 488 / 1079 / 1186 / 544) | **0.83** | **0.77** | **0.84** | **0.81** | no real judge |
| frozen DINOv2 global, real pairs | 0.72 | 0.60 | 0.76 | 0.75 | — |

The explicit-features ranker scores 0.91 / 0.76 / **0.47** / 0.89 on real pairs. It is judged by its own statistics,
which is circular, and still fails on grain.

### 3.5 Transfer to real photographs (235 photos never used, val + test; no family labels exist)

| hybrid_style | value |
|---|---|
| P(mono family) vs measured monochrome, AUROC | 0.99 |
| EXIF film scan vs digital, AUROC of 1 − P(identity) | 0.65 (frozen global 0.51). The existing film-score model reached 0.65 on a different test set |
| Spearman: predicted warmth vs measured warmth / contrast vs y_std / grain vs noise / saturation vs sat_mean | 0.49 / 0.50 / 0.36 / 0.42 |
| uploader-tagged stocks (26 photos, asserted) | Velvia → vivid_reversal 2/3; B&W-tagged mono scans → B&W families 5/6; Superia and Portra → mostly identity |
| independent edits → predicted family | B&W edits → B&W families; HDR → vivid_reversal (contrast +0.34, saturation +0.29); film_emulation → negative / pastel families; clean, oversharpened and posterized → identity; fake grain → +0.03 grain only (**the other pipeline's grain is not recognised**) |

Most real photos are predicted `identity` (133/235). Real film scans mostly do not look like the presets. The film
signal transfers only weakly, which matches section 2: the families are not calibrated to real film.

### 3.6 Shortcut checks

- Nuisance variables predict the family at chance (0.125 vs 0.111).
- Source memorisation is high for raw features and reduced by the metric objectives (3.3).
- **Clipping differs by family** (high_contrast_bw 7 % clipped pixels, identity 1.6 %, others about 0–4 %). Part of
  the look, but also a possible cue; audit it in scaled runs.
- Robustness on test variants (accuracy clean → JPEG q55 / 0.5× rescale / +0.5 EV):
  - DINOv2 intermediate: 0.72 → 0.67 / 0.61 / 0.66;
  - explicit: 0.58 → 0.54 / **0.29 / 0.39**. Explicit statistics are resolution- and exposure-fragile.
- Grain fingerprint: grain does not transfer to another pipeline's grain (3.5), so this generator's grain is partly a
  fingerprint. Next: randomise the grain model itself (Gaussian, clumped, scanned-grain textures).

### 3.7 Composition probe (failed, informative)

On controlled crops of real photos (zoom 1–2×, horizontal offset), test R² on held-out sources:
- offset ≈ 0 or negative for every pooled representation;
- zoom 0.33 only for explicit statistics, through an upscaling-softness shortcut;
- DINOv2 global negative R² (overfits).

**Mean-pooled embeddings discard layout.** Composition needs spatial features: a patch-token grid, or the app's
deterministic composition axes (`photographer_dna/image_analysis.py`: complexity, layering, negative space,
geometry, centrality).

## 4. Photographer-style attribute framework

- **Representation:** the database's 25 axes.
  - 17 structural axes: colour (4), composition (5), light (5), texture (3).
  - 8 narrative axes, read by a vision model in the app.
- **139 profiles**, with style vectors that are editorial estimates.
- **What is observable:**
  - colour, tone and texture axes are learnable from pixels (sections 3.2–3.5);
  - composition axes need spatial features (3.7);
  - narrative axes (humour, intimacy, melancholy …) are not recoverable from synthetic renders and need human or
    model judgements.
- **Profiles are too close to be identities.** The median nearest-profile RMS distance is 0.12. The app's matcher
  gives near-ties (median top-2 gap 1 point) and superficial winners on ordinary photos. So: predict axes, then
  **rank** against profiles, report "supported" only with a margin, and never treat a top-1 as identity.
- **Calibration of the axis-to-profile mapping:** fit predicted axes against human ratings of the app's protocol (fit
  1–5, which of three, none), not against the editorial vectors as if they were measurements.
- **Synthetic film presets are never photographer labels.**

## 5. Not done, with reasons

- **Lighting, relighting, depth of field and haze:** realistic relighting needs depth and normals. Naive 2D gradients
  produce obvious fingerprints, the same failure mode as the grain in 3.6.
- **Object add/remove and compositing:** needs segmentation and inpainting models. Compositing seams are a strong
  shortcut, there is licence and provenance overhead, and subject matter is not style. Postponed until spatial
  composition features exist to evaluate it.
- **Curriculum** (stage 1 → 4): not tested; one controlled study cannot separate curriculum effects.
- **Human-reviewed evaluation set:** the protocol below is ready, but no human ratings exist yet.
- **Data-sheet calibration** of the film engine (section 2).
- **Pilot 2** (Claude, 30 blind images) is still pending; it was not run in this pass.

## 6. Evaluation protocol (for every future model)

1. **Synthetic, held-out sources:** family F1, parameter MAE, pairwise accuracy; reported separately and never
   headlined.
2. **Independent-pipeline transfer:** `datasets/edits.py` pairs, measured-difference judge (as in 3.4).
3. **Real photos:**
   - monochrome AUROC;
   - EXIF film-vs-digital AUROC;
   - Spearman against measured statistics;
   - uploader stock tags (descriptive only).
4. **Human set (to build):**
   - 150 rights-cleared real photos from the test split;
   - 2–3 raters; pairwise "which is warmer / more contrast / grainier / more film-like";
   - broad film family;
   - photographer fit 1–5 with "none";
   - rater agreement is the ceiling.
5. **Robustness:** JPEG q55, 0.5× rescale, ±0.5 EV, crop.
6. **Memorisation:** same-source nearest-neighbour rate.
7. **Cost:** size, peak RSS, latency on the E2-like CPU limits (`deployment/e2/bench_e2.py`).

## 7. Ablation plan (next scaled study, about 500 sources)

The first run of each cell is local and costs about 10 minutes on a 4070-class GPU.
- Transformation groups trained separately and combined (colour/tone, grain/texture, spatial composition).
- Objectives: transformation prediction alone vs + pairwise ranking vs + SupCon vs all.
- Representations: explicit / intermediate DINOv2 / patch-grid DINOv2 / adapter (LoRA on the last 2 blocks) /
  MobileNetV3 student.
- Grain-model randomisation on vs off: does real grain transfer improve?
- Measure every cell on sections 6.2–6.5. Keep only signals that improve real transfer.

## 8. Recommended architecture (current evidence)

```
image → shared preprocessing →
   style branch: DINOv2-small intermediate-layer patch statistics (layers 4, 8) ++ explicit statistics
                 → small MLP trained with (transformation prediction + pairwise ranking + SupCon)   [GPU / remote]
   distilled into MobileNetV3-Small (end-to-end; 4.3 MB fp32 ONNX)                                   [E2-capable]
   heads: attributes, broad film family, pairwise-consistent scores
   matching: film directions = nearest family + attribute profile; photographers = predicted axes vs 139 profiles
             (ranked, "supported" only with a margin; composition axes need the spatial branch first)
→ structured JSON with database ids
```

Production on E2: see `deployment/E2_MICRO_PLAN.md` (student in the app process; DINOv2 remote, optional).

## 9. What to scale next

1. Grain-model randomisation plus pairwise ranking plus SupCon on about 500 sources; check whether real-pair grain
   accuracy rises.
2. Spatial branch (patch grid) for composition axes, probed as in 3.7.
3. Distil the best style branch into MobileNetV3 and re-measure on E2-like limits.
4. Build the 150-photo human set.
5. Fetch manufacturer data sheets for the 10–20 stocks with descriptions, and record curve shape and granularity
   with sources, before any stock-specific preset.

## 10. Update 2026-11-07 (later): second generator, real film with verified stock labels, style student

Full numbers: `docs/EXPERIMENT_LOG.md`, `reports/synth_experiment_v2.json`, `reports/student_style_*.json`.

- **Second generator built:** SDXL base 1.0 image-to-image with OpenRAIL/Civitai film LoRAs (licence review in
  `docs/HF_FILM_MODEL_RESEARCH.md`). Composition preserved (own-source rank-1 0.98). FLUX-based models (FilmPortrait)
  rejected: non-commercial licence.
- **Real film with verified labels obtained:** BlueNeg (ICCV 2025), 491 real 35 mm negatives with the owner's stock
  records, commercial use permitted with credit (`docs/FILM_DATASET_RESEARCH.md`). Used only as an external
  evaluation set.
- **Leave-one-generator-out failed, and the failure is diagnosed.** Probes trained on filmsynth score SDXL outputs
  below chance (AUROC 0.35) because the two pipelines move measured noise in opposite directions (filmsynth adds
  grain, SDXL's VAE removes it). The E1 "treated" signal is largely a generator signature. Section 3.6's suspicion
  is confirmed.
- **The signal still transfers to real film along one axis:** BlueNeg vs digital AUROC 0.80 with the frozen probe
  and 0.81 with the 4.4 MB student, with no real labels used. The EXIF-supervised student remains better (0.86).
- **Same-stock retrieval above chance on real scans** (0.23-0.24 vs 0.12) with SupCon-trained embeddings; the
  student's embedding is content-invariant (own-source rank-1 0.28 vs 1.00 for raw DINOv2).
- **Not achieved:** cross-generator generalisation; composition (unchanged from section 3.7); Z-Image-Turbo as a
  third generator (deferred until nuisance parity is fixed).
- **Next:** nuisance parity (pass untreated images through the same VAE/JPEG path; randomise the grain and sharpening
  models per generator), then re-run leave-one-generator-out; only then scale sources.

## 11. Update 2026-11-07 (later still): leakage fixed by grain randomisation; authenticity audited; spatial grid

- **Cause of the inverted cross-generator AUROC found and fixed.** Cell by cell (`docs/EXPERIMENT_LOG.md` E4):
  size/JPEG standardisation and an SDXL VAE round-trip control changed little; drawing the grain model and strength
  independently of the treated/untreated label raised filmsynth→SDXL from 0.35 to 0.58 [0.52, 0.67] and cut
  generator-identity accuracy from 0.86 to 0.79.
- **The real-film signal was mostly label-tied texture structure.** With grain randomised, BlueNeg film-vs-digital
  fell from 0.80 to 0.70 while same-stock retrieval stayed at 0.25 (chance 0.12). Grain/texture ≈ authenticity,
  colour/tone ≈ aesthetics; they are now separate outputs (`filmlook_infer.shadow`).
- **The EXIF authenticity baseline is weaker than claimed:** within-source AUROC 0.78 [0.69, 0.88], ECE 0.20; the
  cross-source 0.85 includes source identity (it ranks BlueNeg above CommonCatalog film at 0.63). Robust to JPEG,
  rescale and crop. Still the production baseline.
- **Composition:** a 2×2 patch grid recovers signed framing offsets (R² ≈ 0.23) and mirror sensitivity (0.18-0.29)
  where pooled features recover nothing; still weak. Section 3.7's conclusion stands, with a known remedy.
- **Students:** the grain-randomised student (4.4 MB, 1.2 ms) is no longer inverted on an unseen generator (0.54)
  and matches the earlier student on real film within noise. The EXIF student remains the authenticity baseline.
- **Shipped research interfaces:** `filmlook_infer.shadow.ShadowAnalyzer` (separate fields, uncertainty stated),
  `docs/SHADOW_MODE_INTEGRATION.md`, `docs/BADGE_ATTRIBUTE_MAPPING.md` (deterministic attribute badges only).
- **Still true:** every number here is a proxy; no human has judged aesthetics; BlueNeg is one photographer's
  1990s consumer film; nothing was deployed.
