# LUT / preset film-style research — report (2026-11-10)

Question from the owner: can film-simulation presets (13 Lightroom `.xmp` in `~/misc/lut`, no `.cube`) identify the film
colour style of a user photograph? Plus (same day): a deterministic, validated local engine built from the collection, a
film-likeness estimate, and a mapping to the website catalog. Everything below was run today on this workstation; every
number comes from a file under `reports/lut/` (git-ignored) produced by the listed command. No number is estimated.

Budget: 28.1 GPU-seconds (`reports/lut/gpu_ledger.jsonl`, RTX 4090, MLP training only; everything else CPU), 0.96 GB new
disk under `data/lut/` (synthetic renders 8,188 JPEGs, features, real-set features) + 7.2 MB committed database. Disk
92 % -> 93 %. Preflight `python scripts/preflight_check.py` PASS before training (labels are deterministic preset renders,
provenance `synthetic`; no model output is a label anywhere).

## 0. Headline

| question | answer | evidence |
|---|---|---|
| Which of the 13 presets was applied (synthetic, scene-disjoint, 14 classes incl. none) | best learned model top-1 **0.47** [0.43, 0.52], top-3 0.73 (chance 0.071); deterministic engine top-1 0.25, top-3 0.51 | sections 4, 6 |
| Can a held-out (unseen) preset or an unrelated edit be rejected | **No**: max-softmax AUROC 0.44-0.52 vs held-out presets, 0.32-0.41 vs filtered uploads; engine best-distance AUROC 0.70 vs filtered uploads | sections 4, 6 |
| Does any of it identify the real film stock of a real scan (353 scans, 125 uploaders) | **No**: every method 0.05-0.10 line-level top-1 vs chance 0.10 and majority 0.48; family level 0.17-0.34 vs majority 0.54; models trained *on the real scans* also stay below majority | section 5 |
| Film-likeness (scan vs digital), deterministic | AUROC **0.825** [0.79, 0.86] on 502 held-out uploaders (colour-only 0.81), ECE 0.035; baselines monochrome 0.58, fine noise 0.68; JPEG q50 drops it to 0.72 | section 7 |
| Production | ship the engine as *ranked LUT-derived similarity* + film-likeness with the uncertainty flags; never as stock identification | section 9 |

## 1. What the XMP presets contain (Phase 1)

Command: `python -m research.lut.xmp_parser /home/shanmo/misc/lut --out data/lut/presets.json` (parser:
`research/lut/xmp_parser.py`; the owner's files stay where they are, only parsed numbers are written).

All 13 files: Camera Raw 17.2, `ProcessVersion 15.4`, `PresetType Normal`, `RequiresRGBTables=False`, **no**
`LookTable` / `RGBTable` / `Look` (no embedded 3D LUT), no `CameraProfile`, no white balance, exposure, contrast,
highlights/shadows/whites/blacks, vibrance, saturation or calibration (LomoChrome 92 carries zeros). Present everywhere:

| block | content |
|---|---|
| `ToneCurvePV2012` | 2-6 points; black lifted to 13/255 in 8 presets, 14 (Ultramax), 19 (Fujifilm 200), 26 (ColorPlus, Phoenix), 27 (LomoChrome); Ektar keeps 0. White 255 except Fujifilm 200 (252) and Phoenix (246). Portra 160/400, Lomography, Gold: lifted toe, straight above; Ektar, Fujifilm: brighter midtones; Ultramax, LomoChrome, Phoenix: darker midtones |
| `ToneCurvePV2012Red/Green/Blue` | identity in 12 presets; Candido 800T: red lifted highlights, blue clipped (13 -> 0) |
| HSL | 24 sliders per preset, 18-24 non-zero; e.g. Portra 400 orange hue +50, red saturation -35, blue luminance -28 |
| split toning / colour grading | shadow + highlight hue/sat (legacy names) and `ColorGradeMidtone{Hue,Sat}`; global wheel 0, blending 50, balance 0. Strongest: Phoenix midtone sat 71, highlight 52; Gold shadow sat 35; Fujifilm midtone 41 |
| `PointColors` | Gold 200: 3 point-colour edits; Cinestill 50D and Film One Hundred: 1 each; others the empty -1 row |
| grain | amount 20-55 (Phoenix 0), size 23-70, frequency 54-58 |
| clarity / dehaze / sharpen | clarity -36...0, dehaze -14 (Phoenix) else 0, sharpening 40 |

So a preset is a *colour recipe on top of Adobe Color*: the result depends on the photo it is applied to.

## 2. Forward model and its approximations

`research/lut/forward_model.py` (docstring = the full list). sRGB -> linear -> ProPhoto (Bradford D50) -> sRGB TRC encode
-> RGB channel curves -> main curve (PCHIP through the points) -> back to sRGB -> HSL (8 HSV bands with triangular
weights at 0/30/60/120/180/240/280/320 deg, +-100 = +-30 deg hue, saturation x(1+v/100) for v<0 and x(1+0.6v/100) for v>0,
luminance x(1+0.5v/100), all masked by a smoothstep on HSV saturation) -> Point Color (windowed HSV edits) -> colour
grading (CIELAB chroma offset 24*sat/100 in the wheel-hue direction, lightness windows for shadows / midtones /
highlights) -> dehaze veil -> clip. Grain, clarity, texture and sharpening are **not** colour transforms and are not in
the LUT (grain is simulated separately in the synthetic generator). Nothing is fitted to Lightroom (not available here);
renders are "plausible looks in the direction the preset describes". Verified: Lab and HSV round trips exact (1e-14 /
1e-15), 33^3 lattice vs exact path mean 0.31 / max 3 8-bit steps, all grey ramps monotone, 13 lattices pass
`LUT3D.validate()`.

## 3. Inversion and ambiguity (Phase 1)

Command: `python -m research.lut.inverse_analysis --out reports/lut/inverse_analysis.json` (20,000 natural pixels from
120 CC-BY digital photos; grid inverse = 33^3 lattice + KD-tree + 6 damped Gauss-Newton steps).

| preset | min curve slope | black lift (8-bit) | neg. det % | near-singular % | clipped grid % | round-trip dE p95 | dE>2 % | sRGB cube reachable (dE<1) | dark cube reachable (dE<3) |
|---|---|---|---|---|---|---|---|---|---|
| Candido 800T | 0.00 (blue clip) | 13 | 5.2 | 11.6 | 0.5 | 3.46 | 9.4 | 0.41 | 0.27 |
| Cinestill 50D | 0.95 | 13 | 1.1 | 1.8 | 0.3 | 0.00 | 0.0 | 0.47 | 0.50 |
| Film One Hundred | 0.95 | 13 | 1.7 | 4.2 | 0.4 | 0.00 | 0.0 | 0.44 | 0.45 |
| Fujifilm 200 | 0.32 | 19 | 0.3 | 5.7 | 5.6 | 0.00 | 0.5 | 0.48 | 0.16 |
| Kodak ColorPlus 200 | 0.45 | 26 | 4.0 | 15.7 | 8.3 | 0.00 | 0.5 | 0.51 | 0.02 |
| Kodak Ektar 100 | 0.75 | 0 | 5.4 | 19.9 | 5.6 | 0.00 | 0.9 | 0.47 | 1.00 |
| Kodak Gold 200 | 0.86 | 13 | 8.8 | 19.2 | 4.9 | 0.33 | 0.8 | 0.53 | 0.39 |
| Kodak Portra 160 | 0.85 | 13 | 1.1 | 3.1 | 0.0 | 0.00 | 0.0 | 0.53 | 0.53 |
| Kodak Portra 400 | 0.49 | 13 | 3.5 | 11.8 | 6.7 | 0.00 | 0.2 | 0.52 | 0.47 |
| Kodak Ultramax 400 | 0.56 | 14 | 6.1 | 22.2 | 12.5 | 0.00 | 0.9 | 0.67 | 0.56 |
| LomoChrome 92 | 0.13 | 27 | 8.9 | 32.9 | 2.1 | 0.00 | 1.7 | 0.43 | 0.02 |
| Lomography Color 400 | 0.84 | 13 | 0.9 | 4.7 | 4.1 | 0.00 | 0.1 | 0.55 | 0.56 |
| Phoenix 200 | 0.30 | 26 | 8.6 | 39.1 | 0.0 | 0.34 | 3.8 | 0.29 | 0.00 |

Reading: the maps are invertible almost everywhere for natural colours (median round-trip dE 0.00; failures are
Candido's blue clip and Phoenix's compressed range), but 1-9 % of the cube has an orientation flip (band-wise hue shifts
fold neighbouring hues), 30-70 % of the sRGB cube is unreachable (lifted blacks, compressed whites, desaturation), and
the ambiguity test shows **87 % of natural pixels rendered with preset A have an exact pre-image (dE<1) under another
preset B** (min 52 %, Phoenix column: its gamut is narrow). Mean dE between presets on natural pixels 6.9 (min 3.3,
Ultramax vs none 3.4); after an optimal per-channel gain (unknown white balance + exposure) 5.4 (min 2.8: Cinestill 50D
vs Portra 160). Per-pixel identification is impossible; identification can only come from distributions, and an unknown
WB/exposure removes a fifth of the signal. No photo is uniquely invertible.

## 4. Synthetic data and classifiers (Phases 2-3)

Data: `python -m research.lut.synthetic --out data/lut/synth --negatives` -> 366 CC-BY CommonCatalog digital scenes
(digital_camera + smartphone, grey frames dropped; `training_allowed: true`), uploader-disjoint split 211 / 74 / 81
scenes (train / val / test), 13 presets + none, nuisance before the preset (exposure +-0.5 EV, WB +-6 %) and after (grain
from the preset's GrainAmount, 80-100 % crop, JPEG q75-95); 2 variants per train item; 8,188 JPEGs at 512 px (0.9 GB) +
110 untouched *filtered* uploads as unknown-edit negatives. Features: `python -m research.lut.features --dinov2`
(colour 312-d = 24 film_physics statistics + 80 `filmlook_infer.colour_rendition` + 208 CIELAB histograms; DINOv2-small
ONNX 768-d, CPU). Training: `python -m research.lut.train --feature-set F --model M --name N [--holdout ...]`
(default hold-out Candido 800T, Phoenix 200, Lomography Color 400 -> 11 classes; `--holdout ""` -> 14 classes).

| model (11 classes, 3 presets held out) | dim | test top-1 [95 % CI, scene bootstrap] | top-3 | ECE (T-scaled) | reject AUROC: held-out presets / filtered uploads | NN same scene / same preset |
|---|---|---|---|---|---|---|
| stats24 LR | 24 | 0.276 [0.248, 0.305] | 0.563 | 0.082 | 0.49 / 0.38 | 0.96 / 0.01 |
| stats24 kNN | 24 | 0.182 | 0.438 | 0.253 | 0.51 / 0.48 | |
| cr80 LR | 80 | 0.431 [0.387, 0.476] | 0.746 | 0.107 | 0.43 / 0.39 | 0.97 / 0.01 |
| labhist LR | 208 | 0.352 | 0.645 | 0.196 | 0.48 / 0.34 | 0.89 / 0.03 |
| colour LR | 312 | 0.419 [0.369, 0.467] | 0.716 | 0.204 | 0.47 / 0.34 | 0.94 / 0.02 |
| colour kNN | 312 | 0.203 | 0.483 | 0.250 | 0.47 / 0.51 | |
| **colour MLP** (seeds 0/1/2) | 312 | **0.484 / 0.470 / 0.474** [0.44, 0.53] | 0.756 / 0.747 / 0.744 | 0.15 / 0.13 / 0.12 | 0.46 / 0.34 | 0.94 / 0.02 |
| DINOv2 LR | 768 | 0.228 | 0.526 | 0.306 | 0.51 / 0.49 | 1.00 / 0.00 |
| DINOv2 MLP | 768 | 0.279 | 0.541 | 0.043 | 0.52 / 0.41 | 1.00 / 0.00 |
| DINOv2+colour LR | 1080 | 0.357 | 0.672 | 0.139 | 0.47 / 0.37 | 0.99 / 0.00 |
| DINOv2+colour MLP | 1080 | 0.444 [0.403, 0.485] | 0.718 | 0.133 | 0.46 / 0.32 | 0.99 / 0.00 |
| chance | | 0.091 | 0.273 | | 0.50 | 1/81 / 1/11 |

All 14 classes (`colour_mlp_all`): top-1 0.474 [0.430, 0.522], top-3 0.731, ECE 0.120; per class Phoenix 0.86, Candido
0.68, LomoChrome 0.62, Gold 0.59, ColorPlus / Portra 160 0.51, Fujifilm 0.47, Ultramax 0.42, none 0.40, Cinestill 0.35,
Lomography 0.33, Ektar 0.28, Film One Hundred 0.27 (confusion matrices in `reports/lut/train_colour_mlp_all.json`:
Portra 160 <-> Cinestill 50D, Film One Hundred <-> Lomography / Portra 160, Ektar <-> none are the main confusions, which
matches the dE table of section 3). `dinov2_colour_mlp_all` 0.418 / 0.669.

Reading: explicit colour statistics beat DINOv2 (a content embedding: nearest neighbour is the same scene 99.7 % of the
time) and adding DINOv2 does not help. Even the best model is at 0.47 top-1 -- the presets are 3-7 dE apart and the
nuisance already eats part of that. **Calibration** after temperature scaling is 0.12-0.15 ECE (over-confident).
**Rejection fails**: held-out presets are assigned confidently to their nearest known preset (Candido -> Ultramax, Phoenix
-> ColorPlus, Lomography -> Film One Hundred) with AUROC 0.46; filtered real uploads are rejected *worse* than chance
(0.34), i.e. the model is more confident on unknown edits than on known presets. An open-set head would need explicit
unknown training data; max-softmax is not a usable unknown detector here.

Non-learned baselines (`python -m research.lut.baselines --n 400`): forward matching against an estimated neutral
(grey-world + levels, render all presets, nearest colour-feature) top-1 0.150 [0.118, 0.183] / top-3 0.333; inverse-LUT
plausibility (unreachable pixels + cast + black gap) 0.200 [0.154, 0.246] / 0.448; chance 0.071. Both above chance, both
far below the learned models; the neutral estimate is itself biased by the preset it is trying to undo.

Content vs style: splits are scene-disjoint by uploader; the nearest-neighbour probe shows every feature set is
content-dominated (same scene 0.89-1.00 vs same preset 0.00-0.05), so the classifiers work *despite* content, by
learning preset-specific directions in a content-dominated space -- which is why they do not transfer (section 5).

## 5. Real-film validation (Phase 4)

Set (`python -m research.lut.real_eval build`): 1,146 real scans -- 353 colour-negative scans whose uploader stock tag maps
to a preset (ektar 64, gold 28, ultramax 26, fuji C200 22, portra 191 [47 unspecified line, 144 with 160/400], colorplus
10, lomography 400 11, cinestill 50D 1; 125 uploaders; 158 with the exact stock), 505 colour scans of stocks with no
preset (Superia, Natura, Vista, ProImage, Vision3, Reala, Pro 400H, Pro 160, Konica, Cinestill 800T, reversal films) and
288 unlabelled film scans from the training manifest, as unknowns. Labels are uploader claims (class B) except ARTeFACT
(class A). `python -m research.lut.real_eval eval --models ...` (uploader bootstrap, 95 % CI):

| method | n | line-level top-1 (accept 160/400 for "Portra") | chance / majority | top-3 | merged-line top-1 (maj 0.54) | family top-1 (maj 0.54) | exact-stock top-1 (n=150/158) | predicted "none" |
|---|---|---|---|---|---|---|---|---|
| colour MLP (11 cl.) | 342 | 0.094 [0.063, 0.125] | 0.13 / 0.50 | 0.365 | 0.211 | 0.216 [0.17, 0.26] | 0.067 | 0.29 |
| colour MLP (14 cl.) | 353 | 0.079 [0.051, 0.110] | 0.10 / 0.48 | 0.346 | 0.204 | 0.212 | 0.070 | 0.28 |
| colour LR | 342 | 0.102 [0.069, 0.136] | 0.13 / 0.50 | 0.333 | 0.243 | 0.219 | 0.080 | 0.32 |
| cr80 LR | 342 | 0.088 | | 0.351 | 0.173 | 0.181 | 0.093 | 0.28 |
| stats24 LR | 342 | 0.082 | | 0.325 | 0.266 | 0.202 | 0.053 | 0.16 |
| DINOv2 MLP | 342 | 0.088 | | 0.488 | 0.389 | 0.342 [0.28, 0.41] | 0.080 | 0.45 |
| DINOv2+colour MLP (11 / 14 cl.) | 342 / 353 | 0.085 / 0.068 | | 0.342 / 0.278 | 0.202 / 0.170 | 0.193 / 0.167 | 0.087 / 0.057 | 0.35 / 0.31 |
| engine A / B / C / CW / **D** / E | 353 | 0.006 / 0.042 / 0.096 / 0.065 / **0.051** / 0.071 | 0.10 / 0.48 | 0.05 / 0.27 / 0.33 / 0.29 / 0.26 / 0.37 | 0.03 / 0.16 / 0.16 / 0.14 / 0.14 / 0.16 | | | |
| comparator: LR trained **on the real scans**, leave-one-uploader-out, stats24 / colour312 / DINOv2 / DINOv2+colour | 353 | 0.176 / 0.280 / 0.456 / 0.436 (line, balanced classes) | majority 0.54 | | | 0.23 / 0.33 / 0.47 / 0.45 | | |

Reading: **nothing identifies the stock.** Synthetic-trained models are at or below chance on real scans; the most
frequent real predictions are Ektar (81 / 353) and none (98 / 353), and 21 % of unknown-stock scans are also called
"none". Rejection of unknown stocks: AUROC 0.46-0.55 for every method. Even supervised models trained on the real scans
with uploader-disjoint folds stay below the majority rate (Portra is 54 % of the labelled rows), consistent with the
earlier findings (claim 20 in `docs/FINDINGS.md`: same-stock retrieval at chance across photographers; film_physics
2026-11-09: 0.68 family accuracy only when the B&W class is included -- our set is colour only, where the 2026-11-09
evaluation itself had warm 4/12, cool 25/42, vivid 8/12). The presets' colour recipes do not correspond to anything
separable in real scans across photographers, scanners and labs. Cinestill 800T scans (34, no preset) are called Ektar
(8), Ultramax (4), none (3), Candido (1) -- the "tungsten cine" Candido preset does not capture them either.

## 6. Deterministic engine (`research/lut/engine/`, NumPy / SciPy / OpenCV)

Database: `python -m research.lut.engine.lut_db build --n-ref 32` (111 s) -> `research/lut/db/lut_db.json` (0.19 MB) +
`lut_db_arrays.npz` (7.3 MB). Per style: RGB transfer (33-step grey ramp), CIELAB transformation (overall and by
lightness band), hue shift / chroma ratio / dL per 30-degree hue region, saturation response in 5 chroma bins, luminance
response + midtone / shadow / highlight slopes, shadow / midtone / highlight colour bias, 3x3 channel-interaction fit,
tone-curve control points, gamut compression (chroma ratios, unreachable fraction), skin-tone shift on 24 CIELAB patches,
clipping; plus the 33^3 lattice, its grid inverse + residual, descriptors of 32 reference scenes (CC-BY, train split
only) rendered through every style and untouched, their neutralised versions, pixel-cloud signatures, Fisher weights.
Validation (`LUT3D.validate`): all 13 finite, in range, grey-monotone. `.cube` files are supported (`LUT3D.from_cube`,
red-fastest order, DOMAIN_MIN/MAX; round-trip test) -- none exist today. Styles from XMP carry the approximation note.

Photo descriptors (`engine/descriptors.py`, 205 values, 384 px copy): CIELAB L / (a,b) / hue x lightness histograms, HSV
saturation and value percentiles, RGB correlations and std ratios, warm/cool balance, shadow / midtone / highlight cast,
colourfulness, chroma percentiles; luminance percentiles, contrast, dynamic range, highlight roll-off, shadow
compression, midtone share, black / white levels; local contrast, high-frequency energy, noise/grain, edge density,
Laplacian variance. Texture is reported, never matched (a colour LUT cannot describe grain, halation or sharpness).

Approaches (`python -m research.lut.engine_eval --n 700 --write-config`; 700 held-out synthetic test images, 110
filtered-upload negatives, 353 labelled real scans):

| approach | synthetic top-1 [CI] | top-3 | negatives -> "none" | reject AUROC (best distance) | real line top-1 | real top-3 | unknown-stock reject AUROC |
|---|---|---|---|---|---|---|---|
| A descriptor vs pixel-cloud signature | 0.124 [0.107, 0.143] | 0.223 | 0.00 | 0.675 | 0.006 | 0.048 | 0.556 |
| B inverse-LUT plausibility | 0.190 [0.157, 0.223] | 0.439 | 0.61 | 0.605 | 0.042 | 0.269 | 0.547 |
| C nearest reference render | 0.194 [0.165, 0.223] | 0.414 | 0.15 | 0.712 | 0.096 | 0.331 | 0.545 |
| CW Fisher-weighted C | 0.210 [0.178, 0.242] | 0.467 | 0.25 | 0.718 | 0.065 | 0.286 | 0.546 |
| **D Fisher-weighted nearest centroid** | **0.251 [0.214, 0.291]** | 0.510 | 0.37 | 0.696 | 0.051 | 0.261 | 0.555 |
| E scene-conditioned deltas | 0.184 [0.158, 0.212] | 0.383 | 0.22 | 0.653 | 0.071 | 0.371 | 0.574 |
| chance | 0.071 | 0.214 | | 0.5 | 0.101 (majority 0.484) | | 0.5 |

D is the default (`engine_config.json`: `margin_min 0.08`, `abs_distance_max` = 1.25 x p95 of known best distances, near-
monochrome rule). Per class (D): Phoenix 0.62, Portra 400 0.38, Candido 0.37, Lomography 0.31, Ektar 0.30, Gold 0.30;
Cinestill 50D 0.06, LomoChrome 0.08. The margin between the two best candidates separates correct from wrong rankings only
weakly (AUROC 0.57), so `uncertain` is a soft hint. The engine is half as accurate as the colour MLP on synthetic data
(0.25 vs 0.47) and equally useless for real stocks; its advantages are determinism, inspectability (every number in the
database is a documented measurement) and no torch.

Output contract: `analyze()` -> top-3 `{id, name, similarity 0-100 (relative: best = 100, median candidate = 50; not a
probability), explanation (parameter phrases from the XMP), key_features (style vs photo numbers), label_basis
"LUT-derived aesthetic", approximation}`, `uncertain` + reason, `photo` and `texture` summaries, timing.
`catalog_map.json` maps each style to the website catalog (`python -m research.lut.catalog_map`): 10 exact, 3 nearest-
direction (`exact: false`), with zh/en descriptions templated from the database numbers.

## 7. Film-likeness estimate (deterministic)

`python -m research.lut.film_likeness_train build` -> 2,243 images (film 1,743: commoncatalog_film 905 + hd 474 +
ARTeFACT 44 + 320 CommonCatalog film scans; digital 500 CommonCatalog digital_camera / smartphone / filtered), 988
uploaders, split by uploader hash (train 1,104 / test 1,139 -- 893 film, 246 digital, 502 uploaders). `fit`: 36 features
(engine descriptor subset + fine-scale texture on a 1024 px copy: noise sigma in flat regions, lag-1 autocorrelation h/v,
chroma/luma noise ratio, halation proxy, clipping shares, monochrome flag), balanced logistic regression (C=0.3), Platt
recalibration on grouped 5-fold out-of-fold logits.

| metric (held-out uploaders) | value |
|---|---|
| AUROC all | **0.825 [0.792, 0.864]** |
| AUROC colour images only | 0.807 [0.766, 0.839] |
| AUROC train (for reference) | 0.857 |
| ECE (10 bins) | **0.035**; reliability: pred 0.56 -> 0.50 film, 0.66 -> 0.67, 0.76 -> 0.78, 0.86 -> 0.90, 0.95 -> 0.96 (bins below 0.5 hold 5-46 images and are noisy) |
| accuracy at 0.5 | 0.826 [0.791, 0.854] |
| baselines (same test) | monochrome flag 0.577, fine-noise sigma alone 0.680, 384-px noise 0.564 |
| robustness (400-image subsample, clean 0.791) | JPEG q80 0.782 (mean shift 0.04), JPEG q50 **0.717** (0.09), resize 0.5x 0.769 (0.04) |
| strongest weights | fine_noise_sigma +1.39, midtone_share -0.62, noise chroma ratio -0.55, hf_energy_ratio -0.52, chroma_p50 +0.46, clip_high_share -0.46, chroma_p95 -0.40, colourfulness +0.38, monochrome +0.36 |

Reading: it beats both baselines (0.62 / 0.61 in `film_physics/evaluation/README.md`) and matches the old student head
(0.78) with far better calibration -- but it is a Flickr-scan-vs-Flickr-digital detector: fine noise is the main cue, so
heavy JPEG recompression (q50) costs 0.07 AUROC and a noisy high-ISO digital photo will score "film-like". The 246
digital test images come from one source (CommonCatalog), and scan resolution (1024-4096 px vs 1024-1600 px) may leak
into the fine-noise statistics. Verdict for the UI: **"measured estimate" with the evidence list; show the band, not a
score to two decimals; mark `confidence: low` when the image is monochrome or the estimate is within +-0.15 of 0.5.**

## 8. Robustness and cost

Classifier robustness (`python -m research.lut.robustness --models colour_mlp_all,dinov2_colour_mlp_all --n 300`, top-1 on
300 held-out synthetic test images, perturbations applied *after* the preset):

| perturbation (after the preset) | colour MLP top-1 [CI] | DINOv2+colour MLP top-1 [CI] |
|---|---|---|
| clean | 0.510 [0.439, 0.579] | 0.463 [0.404, 0.527] |
| white balance warm (R x1.08, B x0.92) | 0.477 [0.411, 0.554] | 0.440 |
| white balance cool (R x0.92, B x1.08) | 0.463 [0.395, 0.529] | 0.430 |
| exposure -1 EV | **0.277** [0.226, 0.328] | 0.310 [0.254, 0.365] |
| exposure +1 EV | 0.447 [0.382, 0.513] | 0.417 |
| JPEG q50 | 0.483 [0.418, 0.550] | 0.427 |
| JPEG q80 | 0.490 [0.423, 0.560] | 0.437 |
| contrast +10 % (S-curve) | 0.477 | 0.437 |
| saturation -15 % / +15 % | 0.477 / 0.480 | 0.430 / 0.440 |
| 60 % centre crop | 0.463 [0.401, 0.534] | 0.440 |

Reading: white balance, JPEG, mild contrast / saturation edits and crops cost 2-5 points; **-1 EV halves the accuracy**
(the lifted-black signature is destroyed when the photo is darkened after the preset); +1 EV costs 6 points. Mean
confidence barely moves under any perturbation (0.35-0.41), so confidence does not signal the damage.

Latency and memory, one CPU thread (`OMP_NUM_THREADS=1 python -m research.lut.bench --models ... --engine`, 1024x1544 JPEG,
20 repetitions, medians):

| component | median ms (p90) | peak RSS | artefacts |
|---|---|---|---|
| decode + resize (PIL draft, 384 px) | 10 | | |
| colour features (312-d) | 74 | | |
| colour MLP classify (`colour_mlp_all`) | 0.08 -> **84 ms total** | 123 MB (process incl. numpy/PIL) | 0.6 MB npz |
| DINOv2-small ONNX (1 thread) + colour | 104 + 74 -> 190 ms total | 306 MB | 88 MB onnx + 1.4 MB |
| engine `analyze()` (method D) | **80 ms** (p90 82) | 57 MB after load, 92 MB after analyze (fresh process: numpy+scipy+cv2 46 MB) | 7.5 MB db, load 7 ms |
| engine `film_likeness()` | **258 ms** | 237 MB peak (1024 px float32 texture maps; reducible by processing a crop) | 36-weight json |

Measured in `reports/lut/bench.json` and `reports/lut/bench_engine_only.json` (`OMP_NUM_THREADS=1`, cv2 1 thread, 1024x1544
JPEG). Both the engine and the colour MLP fit the 640 MB container with a wide margin; the DINOv2 variant does not add
accuracy and should not be shipped.

## 9. Recommendation

1. **Do not ship stock identification** from these presets, learned or not: real-photo validation is below the majority
   baseline everywhere (section 5), consistent with every earlier session. The honest product statement is "closest
   LUT-derived aesthetic in this collection" with the `uncertain` flag and the `none` candidate visible.
2. The deterministic engine (section 6) is fit for that statement: 7.3 MB database, NumPy/SciPy/OpenCV, 80 ms per
   photo on one thread, explanations traceable to XMP parameters, catalog ids attached. Its ranking quality is modest
   (0.25 top-1, 0.51 top-3 on synthetic data) -- a weighted-centroid match, not a classifier. The colour MLP (0.47 / 0.73)
   is better on synthetic data but needs the 312-d feature code and is equally blind on real film; if the website wants
   the stronger synthetic ranking it can load `data/lut/models/colour_mlp_all.npz` with
   `research.lut.train.NumpyClassifier` (numpy only, 0.3 MB) -- but its confidences are over-confident (ECE 0.12) and must
   not be shown as probabilities.
3. Ship the film-likeness estimate (section 7) as a measured, explained estimate with bands; it is the one component with
   a real-data validation that beats the baselines.
4. Before any claim of recognising a *real* film look, the collection would need Lightroom-exported `.cube` files (exact
   lattices instead of this approximation) and, more importantly, labelled real scans per look from many photographers;
   the available 353 say the presets do not separate them.

## 10. Files and commands

```
research/lut/xmp_parser.py          python -m research.lut.xmp_parser /home/shanmo/misc/lut --out data/lut/presets.json
research/lut/forward_model.py       python -m research.lut.forward_model
research/lut/inverse_analysis.py    python -m research.lut.inverse_analysis
research/lut/explain.py             python -m research.lut.explain
research/lut/synthetic.py           python -m research.lut.synthetic --out data/lut/synth --negatives
research/lut/features.py            python -m research.lut.features --manifest data/lut/synth/manifest.jsonl --out data/lut/synth/features.npz --dinov2
research/lut/train.py               python -m research.lut.train --feature-set colour --model mlp --name colour_mlp_all --holdout ""
research/lut/baselines.py           python -m research.lut.baselines --n 400
research/lut/real_eval.py           python -m research.lut.real_eval build | eval --models colour_mlp_all,...
research/lut/robustness.py          python -m research.lut.robustness --models colour_mlp_all,dinov2_colour_mlp_all
research/lut/bench.py               OMP_NUM_THREADS=1 python -m research.lut.bench --models colour_mlp_all,dinov2_colour_mlp_all --image <jpg> --engine
research/lut/engine/                FilmStyleEngine, lut3d, descriptors, lut_db (build: python -m research.lut.engine.lut_db build), matching, film_likeness
research/lut/engine_eval.py         python -m research.lut.engine_eval --n 700 --write-config
research/lut/catalog_map.py         python -m research.lut.catalog_map
research/lut/film_likeness_train.py python -m research.lut.film_likeness_train build | fit
tests/test_lut_*.py                 pytest -q tests/test_lut*.py
reports/lut/*.json|.log             all numbers quoted above (git-ignored)
```
