# LUT-based film style: can presets identify the film colour style of a photo? (2026-11-10)

Summary for the website team. Full record with commands and every number: `research/lut/REPORT.md`.

## Verdict in three lines

- **Which Lightroom preset was applied** (13 presets + "none", synthetic held-out scenes): the best learned model reaches
  top-1 0.47 [0.43, 0.52] / top-3 0.73 against chance 0.09; the deterministic engine reaches 0.25 [0.21, 0.29] / 0.51. Usable as a
  *ranked similarity*, not as an identification.
- **Which real film stock exposed a real scan** (353 uploader-labelled scans, 125 uploaders, 9 stocks mapped to presets):
  every method, learned or deterministic, is **below the majority-class baseline** (best line-level top-1 0.10 vs
  majority 0.48). The presets do not identify film stocks; neither does anything trained on them. Say "LUT-derived
  aesthetic", never "shot on X".
- **Film-likeness (scan vs digital, 0-100)**: deterministic logistic estimate on measured statistics, held-out-uploader
  AUROC 0.825 [0.79, 0.86] (colour-only 0.825 [0.79, 0.86]_COL), ECE 0.035. Baselines: monochrome flag 0.58, fine noise 0.68.
  Verdict: usable as a *measured estimate* with its evidence list and bands (JPEG q50 lowers AUROC to 0.72; fine noise is the main cue, so noisy digital photos score film-like).

## What the presets contain

13 Adobe Camera Raw presets ("… parvec.xmp", process version 15.4, Camera Raw 17.2): a main point tone curve (every
preset lifts black to 13-27/255 except Ektar; Fujifilm 200 and Phoenix 200 cap white at 252/246), per-channel RGB curves
(Candido only), HSL (8 bands x hue/saturation/luminance, 18-24 non-zero values each), colour grading (shadow / midtone /
highlight hue + saturation; no global wheel, blending 50, balance 0), Point Color (Gold 200: 3 entries; 2 presets: 1),
grain (amount 0-55), clarity / dehaze (Phoenix: dehaze -14), sharpening. **No embedded 3D LUT** (`RequiresRGBTables=False`,
no LookTable/RGBTable), no camera profile, no white balance, no exposure/contrast sliders, no calibration. They are
"Adobe Color + colour recipe" presets: what they do depends on the photo they are applied to.

## The forward model is an approximation (say so in the UI)

`research/lut/forward_model.py` renders a preset from an sRGB JPEG: PCHIP point curves in a ProPhoto-primaries space with
the sRGB transfer curve, HSL as banded HSV edits (±100 = ±30° hue, ×(1+v/100) saturation, ×(1+0.5v/100) luminance),
colour grading as CIELAB chroma offsets weighted by lightness windows, Point Color as windowed HSV edits, dehaze as a
veil. It is not fitted to Lightroom output (Lightroom's pipeline is proprietary and no Lightroom is available here);
grain / clarity / sharpening are texture operations and are not in the LUT. Every style in the database carries this
`approximation` note. If the owner can export the 13 presets as `.cube` from Lightroom, `LUT3D.from_cube` ingests them
and the database builder uses the exact lattice instead (flagged `source: cube`).

## Inversion: a photo cannot be uniquely inverted

`research/lut/inverse_analysis.py` (reports/lut/inverse_analysis.json): all curves monotone except Candido's blue
(clipped 0-13), 1-9 % of the colour cube has an orientation flip (banded hue shifts fold neighbouring hues), 7-29 % of
the sRGB cube is unreachable (lifted blacks), and **87 % of natural pixels rendered with preset A have an exact pre-image
under a different preset B** (min 52 %). Pixel-level identification is impossible; only distributions carry evidence, and
an unknown white balance / exposure absorbs 20-30 % of the inter-preset difference (mean ΔE 6.9 → 5.4 after a
per-channel gain; closest pair after gain: Cinestill 50D vs Portra 160, 2.8 ΔE).

## Engine (deterministic, NumPy / SciPy / OpenCV, no torch) — `research/lut/engine/`

```python
from research.lut.engine import FilmStyleEngine
engine = FilmStyleEngine()                 # db_dir=research/lut/db by default; method from engine_config.json
res = engine.analyze(image, top_k=3)       # image: path | bytes | uint8 RGB array | PIL image
res["styles"]  # [{id, name, similarity 0-100 (relative, not a probability), explanation, key_features, label_basis, approximation}]
res["uncertain"], res["uncertainty_reason"], res["photo"], res["texture"], res["timing_ms"]
fl = engine.film_likeness(image)           # {"estimate": 0-100, "band", "confidence", "evidence": [(stat, value, direction)], "contributions"}
```

Artefacts (all under `research/lut/db/`, committed): `lut_db.json` (0.19 MB: per-style descriptors — RGB transfer, CIELAB
transformation, hue shift per 30° region, saturation / luminance response, shadow / midtone / highlight bias, midtone
contrast, channel interactions, tone curves, gamut compression, skin-tone shift, clipping — plus explanation phrases and
validation), `lut_db_arrays.npz` (7.3 MB: 33³ lattices float16, inverse lattices + residual, 32-scene reference bank,
neutralised bank, pixel-cloud signatures, Fisher weights), `engine_config.json` (chosen method + uncertainty thresholds),
`catalog_map.json` (style → website catalog id, exact flag, zh/en descriptions), `film_likeness.json` (36-weight logistic
model). Reference scenes are 32 CC-BY CommonCatalog digital photographs from the synthetic *train* split (attributions
in `lut_db.json`); only their descriptors are stored.

Matching approaches evaluated (`research/lut/engine_eval.py`): A descriptor-vs-pixel-cloud, B inverse-LUT plausibility,
C nearest reference render, CW Fisher-weighted C, D Fisher-weighted nearest centroid, E scene-conditioned delta
matching. Chosen: **D (Fisher-weighted nearest centroid over 32 reference scenes)** (synthetic top-1 0.25 [0.21, 0.29]). Latency / memory, one CPU thread: engine `analyze()` 80 ms, `film_likeness()` 258 ms, 92 MB RSS after analyze / 237 MB peak during film-likeness (fresh process, numpy+scipy+cv2 baseline 46 MB), 7.5 MB database, 7 ms load; colour MLP 84 ms / 123 MB; DINOv2 variant 190 ms / 306 MB (`reports/lut/bench*.json`).

Catalog mapping (`catalog_map.json`): 10 of 13 styles map exactly to a catalog direction (Portra 400 → `portra_like`,
Portra 160 → `kodak_portra_160_like`, Ektar → `ektar_like`, Gold → `gold_like`, ColorPlus → `kodak_colorplus_200_like`,
UltraMax → `kodak_ultramax_400_like`, Fujifilm 200 → `fuji_c200_like`, Cinestill 50D → `cinestill_50d_like`, LomoChrome 92 →
`lomochrome_color_92_like`, Phoenix → `harman_phoenix_200_like`); Candido 800T → `cinestill_like`, Film One Hundred →
`fuji_industrial_100_like`, Lomography Color 400 → `kodak_ultramax_400_like` are flagged `exact: false`.

## Does it beat the current baseline?

| task | current baseline | this work | verdict |
|---|---|---|---|
| film family / stock on real scans | statistics 0.68 acc on 90 scans *with a B&W class* (film_physics, 2026-11-09); stock-line p@5 at chance (claim 20) | 353 colour scans, 125 uploaders: every preset-based method 0.05-0.10 line top-1 vs majority 0.48; family 0.17-0.34 vs majority 0.54; supervised-on-real LR also below majority | **no improvement; stock claims stay unsupported** |
| which preset / look was applied (synthetic) | none existed | colour MLP 0.47 / 0.73 top-1 / top-3; engine 0.25 / 0.51 (chance 0.07 / 0.21) | new capability, synthetic only, modest |
| film-likeness | monochrome 0.62, fine noise 0.61, student head 0.78 (ECE 0.20) | **0.825 [0.79, 0.86]**, ECE 0.035, 502 held-out uploaders | **beats the baselines; calibrated** |
| unknown-style rejection | - | max-softmax AUROC 0.46 (presets) / 0.34 (edits); engine distance 0.70 vs edits | weak; show `uncertain`, never hide the `none` candidate |
| robustness | - | WB / JPEG / mild edits cost 2-5 points; -1 EV halves accuracy | state in the UI that exposure edits after the look break the match |

## Integration recommendation

1. Vendor `research/lut/engine/` + `research/lut/db/` (7.5 MB) into the website; dependencies numpy, scipy, opencv-python-headless,
   Pillow (optional, for EXIF orientation). Call `FilmStyleEngine().analyze(bytes)` and `.film_likeness(bytes)`; cache both
   results per upload hash (the database itself is precomputed, nothing is recomputed per upload).
2. Present the top-3 as **"closest LUT-derived aesthetic" with similarity 0-100 (relative)**, the catalog direction from
   `catalog_map.json` (show the `exact: false` note for Candido / Film One Hundred / Lomography 400), the zh/en description and
   the `key_features`; when `uncertain` is true, say "no clear match". Never write "shot on <stock>".
3. Present `film_likeness.estimate` as a *measured estimate* with its band and the six evidence statistics; label it
   "weak evidence" when `confidence` is `low` (monochrome or within +-15 of 50).
4. If a stronger synthetic ranking is wanted later: `data/lut/models/colour_mlp_all.npz` + `research.lut.train.NumpyClassifier`
   + `research.lut.features.colour_features` (numpy/PIL, 84 ms, 123 MB) doubles the synthetic top-1 (0.47) with the same
   blindness on real film and over-confident scores (ECE 0.12); it is not required for the honest product statement.
5. Do not ship the DINOv2 variant (no gain, 88 MB, 190 ms). Do not re-add Gemini for these two features.

## Known limitations

- The styles are approximations of Lightroom presets applied to JPEGs, not Lightroom renders of raw files.
- Similarity is relative to this 13-preset collection; a photo with no film look still gets a "closest" style, so the
  `uncertain` flag and the `none` candidate must be shown.
- Real-photo validation does not support any stock claim (every method below the majority baseline on 353 scans).
- Texture (grain, halation, sharpness) is measured for the film-likeness estimate and for information only; no colour
  LUT describes it.
- Labels of the real scans are uploader claims (class B) except ARTeFACT (class A).
