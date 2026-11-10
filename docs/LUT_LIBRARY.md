# Film-emulation LUT library (owner summary, 2026-11-10)

Full record: `research/lut_library/README.md`, `manifest.json`, `candidates.json`, `source_photos.json`, `LOG.md`.
Downloads: `data/lut_library/` (git-ignored, 572 MB = 546 MiB on disk). Nothing was pushed, deployed or trained; `research/lut/` untouched.

## What was found

34 candidate packs/sources were surveyed (EN/JA/ZH web search, GitHub, Hugging Face, pixls.us, G'MIC, RawTherapee,
darktable, Fujifilm, spektrafilm/SpektraLab). 4 approved and downloaded, 24 rejected, 6 kept as reference only.

| Approved source | Licence (text read) | Files | Target stocks covered |
|---|---|---|---|
| RawTherapee Film Simulation Collection 2015-09-20 | CC BY-SA 4.0 (archive README.txt) | 296 HaldCLUT PNG (144^3) | Portra 160/400/800, Ektar, Ektachrome 100VS/E100GX, Pro 400H, Superia 100-1600, Velvia 50/100, Provia, HP5, Delta, Tri-X, T-Max, Agfa Vista/APX + Polaroid/instant/Kodachrome/... |
| GlassMeter Film Looks (Glass Works) | CC BY 4.0 | 30 .cube (33^3, sRGB) | Portra 160/400/800, Gold 200, ColorPlus 200, Ektar, UltraMax 400, Pro 400H, Superia X-TRA 400, CineStill 800T/50D/400D, E100, Velvia 50/100, Provia 100F, HP5, Delta, Tri-X, T-Max |
| filmlab-resolve (spektrafilm) Vision3 | CC BY-SA 4.0 | 8 .cube (65^3, DaVinci Intermediate -> Rec.709 chain) | CineStill 800T/50D base stocks (Vision3 500T/50D), 200T, 250D |
| sguyader/FilmSim | CC0 1.0 | 4 HaldCLUT TIFF + 4 pp3 | Classic Chrome / Provia / Velvia / Acros *camera* looks |

Not found in any approved pack: Fuji C200 and a Classic Negative-inspired look. No stock was invented.

Rejected with reasons (details in `candidates.json`): PictureFX (CC BY-NC), Fujifilm official LUTs (site terms:
personal, non-commercial, no modification), packs without a licence (Sowerby X-Trans III, DocM88, cschmertz,
abpy, Juan Melara), re-uploads that relicense the RawTherapee collection (YahiaAngelo MIT, MokshC MPL, cedeber GPL,
cyangraphics), Hugging Face LUT re-uploads of commercial packs, e-mail-gated marketing packs, Q-DDL (site unreachable).

## Decisions the owner should make

1. **Share-alike.** 304 of the 337 LUT files are CC BY-SA 4.0. The licence permits commercial and derivative use; it
   requires derivatives of the LUTs to stay CC BY-SA. Whether synthetic training images or a model trained on them are
   "adaptations" is unresolved in law. `configs/datasets.yaml` denylists CC-BY-SA for photographs; this library records
   the caveat per record (`ml_derivative_use`) and leaves the call to you. The 34 CC BY / CC0 files carry no such caveat.
2. **Attribution** strings are in the manifest (`attribution_text`) and must ship with anything derived.
3. **Trademarks**: stock names describe what a LUT approximates; the product must not claim film-stock identification.

## Validation

`research/lut_library/validate.py`: 334/337 pass; the 3 failures are intentional inversions (Negative.png,
CreativePack ColorNegative, Rollei IR 400). 42 pass with a clipping warning.

## Source photographs for synthetic generation

`source_photos.json`: 669 tier-1 images (321 CommonCatalog CC-BY digital/phone originals, 128 PD12M CC0, 220 PD12M
iNaturalist CC0) plus 393 tier-2 CC-BY film scans / filtered uploads; test split, quarantine and all evaluation-only
holdouts excluded; heuristic scene categories (no model). Birds/flowers come only from the iNaturalist set; indoor
scenes are under-represented (about 10). No new photo set was downloaded.
