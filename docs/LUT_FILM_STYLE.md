# LUT-based film style: can presets identify the film colour style of a photo? (2026-11-10)

Summary for the website team. Full record with commands and every number: `research/lut/REPORT.md`.

## Verdict in three lines

- **Which Lightroom preset was applied** (13 presets + "none", synthetic held-out scenes): the best learned model reaches
  top-1 TBD_SYN_TOP1 / top-3 TBD_SYN_TOP3 against chance 0.09; the deterministic engine reaches TBD_ENG_SYN_TOP1 / TBD_ENG_SYN_TOP3. Usable as a
  *ranked similarity*, not as an identification.
- **Which real film stock exposed a real scan** (353 uploader-labelled scans, 125 uploaders, 9 stocks mapped to presets):
  every method, learned or deterministic, is **below the majority-class baseline** (best line-level top-1 TBD_REAL_BEST vs
  majority 0.48). The presets do not identify film stocks; neither does anything trained on them. Say "LUT-derived
  aesthetic", never "shot on X".
- **Film-likeness (scan vs digital, 0-100)**: deterministic logistic estimate on measured statistics, held-out-uploader
  AUROC TBD_FL_AUROC (colour-only TBD_FL_AUROC_COL), ECE TBD_FL_ECE. Baselines: monochrome flag TBD_FL_MONO, fine noise TBD_FL_NOISE.
  Verdict: TBD_FL_VERDICT.

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
matching. Chosen: **TBD_ENG_METHOD** (synthetic top-1 TBD_ENG_SYN_TOP1). Latency / memory, one CPU thread: TBD_BENCH.

Catalog mapping (`catalog_map.json`): 10 of 13 styles map exactly to a catalog direction (Portra 400 → `portra_like`,
Portra 160 → `kodak_portra_160_like`, Ektar → `ektar_like`, Gold → `gold_like`, ColorPlus → `kodak_colorplus_200_like`,
UltraMax → `kodak_ultramax_400_like`, Fujifilm 200 → `fuji_c200_like`, Cinestill 50D → `cinestill_50d_like`, LomoChrome 92 →
`lomochrome_color_92_like`, Phoenix → `harman_phoenix_200_like`); Candido 800T → `cinestill_like`, Film One Hundred →
`fuji_industrial_100_like`, Lomography Color 400 → `kodak_ultramax_400_like` are flagged `exact: false`.

## Does it beat the current baseline?

TBD_BASELINE_COMPARISON

## Integration recommendation

TBD_RECOMMENDATION

## Known limitations

- The styles are approximations of Lightroom presets applied to JPEGs, not Lightroom renders of raw files.
- Similarity is relative to this 13-preset collection; a photo with no film look still gets a "closest" style, so the
  `uncertain` flag and the `none` candidate must be shown.
- Real-photo validation does not support any stock claim (every method below the majority baseline on 353 scans).
- Texture (grain, halation, sharpness) is measured for the film-likeness estimate and for information only; no colour
  LUT describes it.
- Labels of the real scans are uploader claims (class B) except ARTeFACT (class A).
