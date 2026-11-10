"""Build research/lut_library/manifest.json from the downloaded LUT library in data/lut_library/.

One record per LUT file (plus the non-LUT presets that travel with a source), with the source's licence
record, SHA-256, parsed film-stock claim, variant, format, size, colour spaces and the validate.py result.
The candidate survey (every pack found, approved or rejected) is written alongside as candidates.json and
embedded in the manifest.

Run from the repository root:
    .venv/bin/python -I research/lut_library/build_manifest.py

Nothing is downloaded here; the downloads and their licence checks are logged in LOG.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "data" / "lut_library"
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from validate import validate  # noqa: E402
from filmlook_ml.datasets.film_stock import stock_claim  # noqa: E402

TODAY = "2026-11-10"

# --------------------------------------------------------------------------------------------------
# Approved sources (one directory each under data/lut_library/). Licence text was read on the date given.
# --------------------------------------------------------------------------------------------------
SOURCES = {
    "rawtherapee_film_simulation": {
        "name": "RawTherapee Film Simulation Collection (version 2015-09-20)",
        "source_url": "https://rawtherapee.com/shared/HaldCLUT.zip",
        "info_url": "https://rawpedia.rawtherapee.com/Film_Simulation",
        "author": "Pat David, Pavlov Dmitry, Michael Ezra (credits in README.txt)",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_quote": "RawTherapee Film Simulation Collection version 2015-09-20 / CC BY-SA 4.0 (HaldCLUT/README.txt inside the archive; the RawPedia page carries the same text).",
        "license_verified_on": TODAY,
        "free_download": True, "personal_use": True, "commercial_permitted": True, "modification_permitted": True,
        "redistribution_permitted": "yes, under CC BY-SA 4.0 with attribution",
        "ml_derivative_use": "permitted (CC BY-SA allows any use incl. derivatives); SHARE-ALIKE CAVEAT: adaptations of the LUTs (re-encodings, blends, images rendered through them?) must be released CC BY-SA 4.0; whether a trained model is an adaptation is unresolved - record and let the owner decide before shipping anything derived",
        "approved_for_training": True,
        "attribution_text": "Film simulation HaldCLUTs from the RawTherapee Film Simulation Collection (Pat David, Pavlov Dmitry, Michael Ezra), CC BY-SA 4.0, https://rawpedia.rawtherapee.com/Film_Simulation",
        "archive_sha256": "0ffca81f30c72d7bbb85adab0ed98bbdeb0e1034cd6f5986f9f993bb66999dc2",
        "archive_bytes": 421602289,
        "download_date": TODAY,
        "format": "HaldCLUT PNG (level 12 = 144^3, 8-bit sRGB); identity in TIFF",
        "input_color_space": "sRGB (display-referred, 8-bit)",
        "output_color_space": "sRGB",
        "technical_notes": "Hand-tuned by eye against reference scans (Pat David, pixls.us thread 34265: 'less about color accuracy and more about a subjective tweaking'); some CLUTs were reverse-generated from other colour-curve packages. Push/pull variants are numbered (1 -, 2, 3 +, 4 ++ ...); 'Cold'/'Warm' = white-balance variants; 'Alt'/'HC' = alternative/high-contrast; 'Generic' = generic rendition; 'Negative' = negative-image LUTs (inverting; fail the monotonic check by design). CreativePack-1 and Negative.png are not film stocks.",
        "local_dir": "rawtherapee_film_simulation/HaldCLUT",
    },
    "glassmeter_film_looks": {
        "name": "GlassMeter Film Looks (Glass Works) - 30 film simulations as open data",
        "source_url": "https://github.com/TrondW/glassmeter-film-looks",
        "info_url": "https://glass-works.ai",
        "author": "Glass Works (GlassMeter app); repository TrondW/glassmeter-film-looks",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_quote": "README: 'Everything here is licensed CC BY 4.0: use it, change it, ship it in free or paid apps. The one condition is credit.' LICENSE file = Creative Commons Attribution 4.0 International. Each .cube header repeats 'Licensed CC BY 4.0'.",
        "license_verified_on": TODAY,
        "free_download": True, "personal_use": True, "commercial_permitted": True, "modification_permitted": True,
        "redistribution_permitted": "yes, with credit",
        "ml_derivative_use": "permitted (CC BY 4.0: any use incl. derivatives, attribution required)",
        "approved_for_training": True,
        "attribution_text": "Film simulations from GlassMeter by Glass Works (glass-works.ai), CC BY 4.0",
        "source_commit": "6a65ee016120d6fc3e17219c8a7d7a1d187abd90",
        "download_date": TODAY,
        "format": ".cube 33^3 (also Lightroom .xmp profiles and profiles/film-stocks.json in the repo; only the cubes, README, LICENSE, AGENT_GUIDE.md and film-stocks.json were fetched)",
        "input_color_space": "sRGB (display-referred, normally exposed)",
        "output_color_space": "sRGB",
        "technical_notes": "Original curve sets authored for the GlassMeter app (channel mixer, per-channel tone curves, saturation, print contrast, characteristic curve taken from each film's published H&D curve character); explicitly 'not derived from their LUTs or anyone else's'. Grain and halation are not in the LUTs. Trademark disclaimer in README.",
        "local_dir": "glassmeter_film_looks/luts/cube-srgb-33",
    },
    "filmlab_resolve_vision3": {
        "name": "filmlab-resolve: spektrafilm-derived Kodak Vision3 negative + print LUTs for DaVinci Resolve",
        "source_url": "https://github.com/peterskotte123/filmlab-resolve",
        "info_url": "https://github.com/andreavolpato/spektrafilm",
        "author": "LUTs generated by spektrafilm (Andrea Volpato) profiles, baked by peterskotte123",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_quote": "README 'Licences': 'LUTs and fits.json (derived from spektrafilm's profiles): CC BY-SA 4.0 + luts/SPEKTRAFILM_LICENSE.txt (commercial use OK, free to share, no resale). Credit: spektrafilm by Andrea Volpato.' Each .cube header: 'Copyright (c) 2026 Andrea Volpato. Generated by spektrafilm. Licensed under CC BY-SA 4.0.'",
        "license_verified_on": TODAY,
        "free_download": True, "personal_use": True, "commercial_permitted": True, "modification_permitted": True,
        "redistribution_permitted": "yes, CC BY-SA 4.0 with credit; SPEKTRAFILM_LICENSE.txt preamble asks for a CHANGELOG with modified copies and forbids resale of the LUTs as such",
        "ml_derivative_use": "permitted with the same share-alike caveat as the RawTherapee collection",
        "approved_for_training": True,
        "attribution_text": "Kodak Vision3 negative/print LUTs generated by spektrafilm (Andrea Volpato, https://github.com/andreavolpato/spektrafilm), CC BY-SA 4.0, via peterskotte123/filmlab-resolve",
        "source_commit": "7f0ca0f5b16cbc66f0f987946ad30487ed54b417",
        "download_date": TODAY,
        "format": ".cube 65^3; 8 of the 28 LUTs fetched (per stock: <stock>_film.cube and <stock>_2383_print_rec709.cube)",
        "input_color_space": "film LUT: DaVinci Intermediate / DaVinci Wide Gamut (cctf on); print LUT: the intermediate 'density wire' D = code*4.2-0.2",
        "output_color_space": "film LUT: density wire (not viewable); print LUT: Rec.709 (cctf on)",
        "technical_notes": "Two-stage chain: scene (DaVinci Intermediate) -> *_film.cube -> density wire -> *_2383_print_rec709.cube -> Rec.709. Never apply a single file alone to an sRGB photo; convert sRGB -> DaVinci Intermediate first. Relevant to CineStill 800T / 50D (respooled Vision3 500T / 50D without the remjet layer) and to a 'cine print' look; physically modelled (spectral sensitivities, H&D curves, dyes, print stock), not hand-tuned. HLG/PQ outputs and the 2393 print variants were not fetched.",
        "local_dir": "filmlab_resolve_vision3/luts",
    },
    "sguyader_filmsim": {
        "name": "sguyader/FilmSim - Fujifilm-camera-look HaldCLUTs and RawTherapee pp3 presets",
        "source_url": "https://github.com/sguyader/FilmSim",
        "info_url": "https://discuss.pixls.us/t/classic-chrome-haldclut-like-the-x-t2-for-rt/1797",
        "author": "Sebastien Guyader (sguyader)",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_quote": "Repository LICENSE = CC0 1.0 Universal; author on pixls.us: 'Yes of course, I release them for Public Domain!'",
        "license_verified_on": TODAY,
        "free_download": True, "personal_use": True, "commercial_permitted": True, "modification_permitted": True,
        "redistribution_permitted": "yes (CC0)",
        "ml_derivative_use": "permitted (CC0)",
        "approved_for_training": True,
        "attribution_text": "Fujifilm-look HaldCLUTs by Sebastien Guyader (sguyader/FilmSim), CC0 1.0 (credit appreciated, not required)",
        "source_commit": "1453b2b55c48d99a889b1e455f91f6898ba2db41",
        "download_date": TODAY,
        "format": "HaldCLUT TIFF (level 12 = 144^3, 8-bit RGB) + RawTherapee .pp3 processing profiles",
        "input_color_space": "sRGB (meant for a neutral RawTherapee rendering of a raw file)",
        "output_color_space": "sRGB",
        "technical_notes": "Emulates Fujifilm X-series in-camera film simulations (Classic Chrome, Provia standard, Velvia vivid, Acros), i.e. digital camera looks named after film stocks - not measurements of the film stocks themselves. Made by matching RawTherapee output to X-T2 JPEGs by eye.",
        "local_dir": "sguyader_filmsim",
    },
}

# --------------------------------------------------------------------------------------------------
# Every candidate found during discovery (approved and rejected). status: approved | rejected | reference
# --------------------------------------------------------------------------------------------------
CANDIDATES = [
    {"candidate": "RawTherapee Film Simulation Collection", "url": "https://rawtherapee.com/shared/HaldCLUT.zip", "status": "approved", "license": "CC-BY-SA-4.0", "reason": "README.txt in the archive states CC BY-SA 4.0", "downloaded": True, "source_slug": "rawtherapee_film_simulation", "stocks": "Portra 160/400/800 (+NC/VC/UC), Ektar 100, Ektachrome 100 VS / E100GX, Kodachrome, Elite Chrome, Tri-X, T-Max 100/400/3200, BW400CN, Fuji 160C/400H/800Z, Superia 100-1600, Reala, Provia, Velvia, Astia, Sensia, Neopan, Acros, Ilford HP5/FP4/Delta/Pan F/XP2, Agfa APX/Vista/Ultra/Precisa, Polaroid, Rollei, Lomography, Fuji FP instant"},
    {"candidate": "GlassMeter Film Looks (TrondW/glassmeter-film-looks)", "url": "https://github.com/TrondW/glassmeter-film-looks", "status": "approved", "license": "CC-BY-4.0", "reason": "LICENSE + README CC BY 4.0", "downloaded": True, "source_slug": "glassmeter_film_looks", "stocks": "Portra 160/400/800, Ektar, Gold 200, UltraMax 400, ColorPlus 200, Pro 400H, Industrial 100, Superia X-TRA 400, CineStill 800T/50D/400D, Ektachrome E100, Velvia 50/100, Provia 100F, HP5, FP4, Delta 100/400/3200, Pan F, XP2, Tri-X, T-MAX 100/400/P3200, Double-X, Acros 100 II"},
    {"candidate": "filmlab-resolve Vision3 LUTs (spektrafilm)", "url": "https://github.com/peterskotte123/filmlab-resolve", "status": "approved", "license": "CC-BY-SA-4.0", "reason": "README licence section + SPEKTRAFILM_LICENSE.txt + per-file header", "downloaded": True, "source_slug": "filmlab_resolve_vision3", "stocks": "Kodak Vision3 50D/250D/200T/500T + 2383/2393 print (CineStill 50D/800T base stocks)"},
    {"candidate": "sguyader/FilmSim (Classic Chrome, Provia, Velvia, Acros HaldCLUTs)", "url": "https://github.com/sguyader/FilmSim", "status": "approved", "license": "CC0-1.0", "reason": "LICENSE = CC0 1.0; public-domain release stated on pixls.us", "downloaded": True, "source_slug": "sguyader_filmsim", "stocks": "Fujifilm camera simulations: Classic Chrome, Provia Std, Velvia Vivid, Acros"},
    {"candidate": "G'MIC color_presets film packs (gmic.eu)", "url": "https://gmic.eu/color_presets/index.html", "status": "rejected", "license": "not stated for the CLUT data (G'MIC itself CeCILL)", "reason": "Packs are re-encodings of the RawTherapee collection, PictureFX and others; no CLUT licence on gmic.eu. Originals taken from RawTherapee instead; PictureFX part is CC BY-NC. Duplicate, not downloaded.", "downloaded": False},
    {"candidate": "Pat David film emulation presets (patdavid.net / G'MIC)", "url": "https://patdavid.net/2013/08/film-emulation-presets-in-gmic-gimp/", "status": "rejected", "license": "site footer CC BY-SA 4.0 'unless otherwise noted'; no preset-specific statement", "reason": "Same CLUTs are in the RawTherapee collection with an explicit CC BY-SA 4.0 README; the blog offers no separate download. Covered by rawtherapee_film_simulation.", "downloaded": False},
    {"candidate": "PictureFX HaldCLUTs / LUTs (Marc Roovers)", "url": "https://marcrphoto.wordpress.com/picturefx-pro-2024/", "status": "rejected", "license": "CC-BY-NC-4.0 (Kodak Aerochrome LUT CC BY-SA 4.0)", "reason": "'may not use the Luts and the brand PictureFX for commercial purposes' - non-commercial; FilmLook is a commercial product. Not downloaded.", "downloaded": False},
    {"candidate": "PictureFX CineStill 800T v2.0 (itch.io)", "url": "https://picturefx.itch.io/picturefx-cinestill-800t-v20", "status": "rejected", "license": "no licence on the page; author's site: CC BY-NC 4.0", "reason": "Falls under the PictureFX non-commercial licence.", "downloaded": False},
    {"candidate": "Fujifilm official 3D LUTs (F-Log / film simulation)", "url": "https://www.fujifilm-x.com/global/support/download/lut/", "status": "rejected", "license": "Fujifilm site terms (global.fujifilm.com/en/terms): personal, non-commercial; modification prohibited without permission", "reason": "Terms do not permit commercial or derivative use.", "downloaded": False},
    {"candidate": "Stuart Sowerby Fuji X-Trans III film simulation HaldCLUTs", "url": "https://discuss.pixls.us/t/new-fujifilm-xtrans-film-simulations/6333", "status": "rejected", "license": "unverified (none stated; forum users asked and got no answer)", "reason": "No licence.", "downloaded": False},
    {"candidate": "abpy/FujifilmCameraProfiles", "url": "https://github.com/abpy/FujifilmCameraProfiles", "status": "rejected", "license": "unverified (no LICENSE)", "reason": "No licence; derived from Fujifilm camera output.", "downloaded": False},
    {"candidate": "pixls.us 'Fujifilm film simulation LUT in ART' HaldCLUT conversion", "url": "https://discuss.pixls.us/t/fujifilm-film-simulation-lut-in-art/55019", "status": "rejected", "license": "derived from Fujifilm official .cube files (Fujifilm terms apply)", "reason": "Upstream terms are non-commercial / no modification.", "downloaded": False},
    {"candidate": "YahiaAngelo/Film-Luts (MIT claim)", "url": "https://github.com/YahiaAngelo/Film-Luts", "status": "rejected", "license": "MIT claimed, but README disclaims ownership of the LUTs", "reason": "Re-upload of G'MIC/RawTherapee CLUTs; the uploader cannot relicense them. Duplicate of the RawTherapee source.", "downloaded": False},
    {"candidate": "Jack4B/Retro-Film-LUTs", "url": "https://github.com/Jack4B/Retro-Film-LUTs", "status": "rejected", "license": "CC-BY-SA-4.0 (derivative of RawTherapee collection, properly attributed)", "reason": "Lossless re-encoding of the RawTherapee collection; duplicate.", "downloaded": False},
    {"candidate": "MokshC/Film-Emulation-LUTs (65^3 .cube from RawTherapee)", "url": "https://github.com/MokshC/Film-Emulation-LUTs", "status": "rejected", "license": "MPL-2.0 declared; upstream CC BY-SA 4.0", "reason": "Converted RawTherapee collection; MPL relicensing of CC BY-SA data is not valid. Duplicate.", "downloaded": False},
    {"candidate": "cyangraphics/clut", "url": "https://github.com/cyangraphics/clut", "status": "rejected", "license": "CC-BY-SA-4.0 declared", "reason": "Mirror of Pat David's CLUTs + PictureFX (CC BY-NC upstream) + Juan Melara print LUTs (no licence upstream); the mirror's licence cannot cover the latter two. Duplicate for the Pat David part.", "downloaded": False},
    {"candidate": "cedeber/hald-clut", "url": "https://github.com/cedeber/hald-clut", "status": "rejected", "license": "GPL-3.0 declared on a collection of third-party CLUTs (RawTherapee, PictureFX, Fuji X-Trans III, Apple Photos, Pixelmator)", "reason": "Includes Apple/Pixelmator-derived LUTs and PictureFX; relicensing invalid.", "downloaded": False},
    {"candidate": "Lts1sds/filmcam", "url": "https://github.com/Lts1sds/filmcam", "status": "rejected", "license": "GPL-3.0 (code); LUTs are RawTherapee collection files", "reason": "Duplicate.", "downloaded": False},
    {"candidate": "cschmertz/rawtherapee-film-looks", "url": "https://github.com/cschmertz/rawtherapee-film-looks", "status": "rejected", "license": "unverified (no LICENSE)", "reason": "No licence; cluts/ are RawTherapee collection files.", "downloaded": False},
    {"candidate": "DocM88/Free-Lut-Pack (Lightroom .xmp)", "url": "https://github.com/DocM88/Free-Lut-Pack", "status": "rejected", "license": "unverified (no LICENSE)", "reason": "No licence.", "downloaded": False},
    {"candidate": "sohan-shingade/fujiLUT, sadziu/Fuji-Recipe-Converter", "url": "https://github.com/sadziu/Fuji-Recipe-Converter", "status": "rejected", "license": "MIT (converter code) / none", "reason": "Converters, no LUT files shipped; recipes are community camera settings.", "downloaded": False},
    {"candidate": "HNGSeries/mfilm-canon-film-simulation", "url": "https://github.com/HNGSeries/mfilm-canon-film-simulation", "status": "rejected", "license": "GPL-2.0 (code)", "reason": "Magic Lantern film simulation; the repo ships no LUT files ('LUTs are not stored in git, most are copyrighted').", "downloaded": False},
    {"candidate": "t3mujin/t3mujinpack (darktable styles, 51 stocks)", "url": "https://github.com/t3mujin/t3mujinpack", "status": "rejected", "license": "unverified (README has no licence; repo moved)", "reason": "darktable .dtstyle presets, not LUTs; licence not verified.", "downloaded": False},
    {"candidate": "Juan Melara print film emulation LUTs (2383/2393/3510)", "url": "http://juanmelara.com.au/blog/print-film-emulation-luts-for-download", "status": "rejected", "license": "unverified (no licence on the page)", "reason": "No licence statement; also expects log scans, not display-referred stills.", "downloaded": False},
    {"candidate": "Q-DDL 800+ free LUTs (CC BY 4.0 per cgchannel.com)", "url": "https://quickddl.net/free_color_luts/", "status": "rejected", "license": "unverified (CC BY 4.0 reported second-hand)", "reason": "quickddl.net does not resolve (2026-11-10); downloads reportedly went through ad-shortener links. Could not verify.", "downloaded": False},
    {"candidate": "freevisuals.net / cinecolor.io / presetpro.com free film LUTs", "url": "https://www.freevisuals.net/free-luts", "status": "rejected", "license": "site-specific 'free' terms, no open licence; cinecolor/presetpro distribute through shop checkouts (e-mail)", "reason": "Marketing packs; no clear derivative/ML permission; e-mail-gated checkouts not used.", "downloaded": False},
    {"candidate": "Hugging Face NeoGraph/Luts_Cube", "url": "https://huggingface.co/datasets/NeoGraph/Luts_Cube", "status": "rejected", "license": "other", "reason": "Re-upload of commercial LUT packs (RocketStock-style names); no rights.", "downloaded": False},
    {"candidate": "Hugging Face liguanwei/luts, cgauss/comfy_luts", "url": "https://huggingface.co/datasets/liguanwei/luts", "status": "rejected", "license": "apache-2.0 tag / none", "reason": "Unknown origin cinematic LUTs, not film stocks; a card tag cannot license third-party LUTs.", "downloaded": False},
    {"candidate": "JanLohse/spectral_film_lut (generator)", "url": "https://github.com/JanLohse/spectral_film_lut", "status": "reference", "license": "MIT", "reason": "Datasheet-based LUT generator (GUI/AppImage), no pre-built LUTs shipped. Option for future generation of Portra/Gold/Ektar/Velvia/Provia LUTs under MIT; not run in this session.", "downloaded": False},
    {"candidate": "andreavolpato/spektrafilm (profiles + LUT creator)", "url": "https://github.com/andreavolpato/spektrafilm", "status": "reference", "license": "GPL-3.0 code; profiles/LUTs CC BY-SA 4.0", "reason": "No pre-built LUT releases; local checkout already in film_physics/third_party (owner-approved). LUTs could be generated locally under CC BY-SA 4.0.", "downloaded": False},
    {"candidate": "JamesQiu2005/spektralab-engine, SpektraLab", "url": "https://github.com/JamesQiu2005/spektralab-engine", "status": "reference", "license": "GPL-3.0 / none", "reason": "Render engines (macOS/Metal), no LUT files.", "downloaded": False},
    {"candidate": "bean-mhm/flim", "url": "https://github.com/bean-mhm/flim", "status": "reference", "license": "AGPL-3.0", "reason": "Generic filmic transform (.spi3d), not a film-stock emulation.", "downloaded": False},
    {"candidate": "Filmulator, OpenColorIO configs", "url": "https://github.com/CarVac/filmulator-gui", "status": "reference", "license": "GPL-3.0 / BSD-3", "reason": "Not LUT packs (process simulation / colour management).", "downloaded": False},
    {"candidate": "Kodak technical data sheets (E-4050 Portra 400, E-4046 Ektar 100, E-7022 Gold 200, ...)", "url": "https://125px.com/techdocs/kodak/", "status": "reference", "license": "Kodak Alaris copyright", "reason": "Reference curves only; not LUTs, not downloaded.", "downloaded": False},
]

# filename tokens that are variants, not part of the stock name
_VARIANT_TOKENS = {"cold", "warm", "cool", "hc", "negative", "early", "generic", "alt", "xpro", "expired", "v2", "ii"}
_PUSHPULL = re.compile(r"^[+-]{1,3}$")

# fallbacks for names the project canonicaliser does not resolve
_OVERRIDES = [
    (re.compile(r"fuji 160c", re.I), "fuji_pro160c", "fuji", "colour_negative"),
    (re.compile(r"fuji 800z", re.I), "fuji_pro800z", "fuji", "colour_negative"),
    (re.compile(r"industrial.?100", re.I), "fuji_industrial_100", "fuji", "colour_negative"),
    (re.compile(r"precisa", re.I), "agfa_ctprecisa_100", "agfa", "colour_reversal"),
    (re.compile(r"hps 800", re.I), "ilford_hps_800", "ilford", "bw_negative"),
    (re.compile(r"redscale", re.I), "lomography_redscale_100", "lomography", "colour_negative_special"),
    (re.compile(r"x-?pro slide", re.I), "lomography_xpro_slide_200", "lomography", "colour_reversal"),
    (re.compile(r"classic.?chrome", re.I), "fujifilm_sim_classic_chrome", "fuji", "camera_simulation"),
    (re.compile(r"provia.?std", re.I), "fujifilm_sim_provia", "fuji", "camera_simulation"),
    (re.compile(r"velvia.?vivid", re.I), "fujifilm_sim_velvia", "fuji", "camera_simulation"),
    (re.compile(r"^acros$", re.I), "fujifilm_sim_acros", "fuji", "camera_simulation"),
    (re.compile(r"fp-?100c", re.I), "fuji_fp100c", "fuji", "instant_colour"),
    (re.compile(r"fp-?3000b", re.I), "fuji_fp3000b", "fuji", "instant_bw"),
    (re.compile(r"polaroid", re.I), None, "polaroid", "instant"),
    (re.compile(r"rollei", re.I), None, "rollei", "bw_negative"),
    (re.compile(r"kodak hie", re.I), "kodak_hie", "kodak", "bw_infrared"),
    (re.compile(r"bw 400 ?cn", re.I), "kodak_bw400cn", "kodak", "bw_negative_c41"),
    (re.compile(r"elite (chrome|color|extracolor|100 xpro)", re.I), "kodak_elite_chrome", "kodak", "colour_reversal"),
    (re.compile(r"double-?x", re.I), "kodak_double_x", "kodak", "bw_negative_cine"),
    (re.compile(r"cinestill.?400d", re.I), "cinestill_400d", "cinestill", "colour_negative_cine"),
    (re.compile(r"^(50d|250d|200t|500t)", re.I), None, "kodak", "colour_negative_cine"),
]
_VISION3 = {"50d": "kodak_vision3_50d", "250d": "kodak_vision3_250d", "200t": "kodak_vision3_200t", "500t": "kodak_vision3_500t"}


def parse_name(stem: str, slug: str) -> dict:
    """Split a file stem into stock name, variant tokens and push/pull level, then canonicalise the stock."""
    tokens = stem.replace("_", " ").replace("-", " ").split()
    base, variants, push = [], [], 0
    idx = None
    for t in tokens:
        tl = t.lower()
        if _PUSHPULL.match(t):
            push += t.count("+") - t.count("-")
        elif tl in _VARIANT_TOKENS:
            variants.append(tl)
        elif len(t) == 1 and t.isdigit() and base:
            idx = int(t)
        else:
            base.append(t)
    base_name = " ".join(base)
    rec = {"base_name": base_name, "variant_tokens": variants, "push_pull_stops": push, "variant_index": idx}
    if slug == "filmlab_resolve_vision3":
        key = tokens[0].lower()
        rec.update({"film_stock": _VISION3.get(key), "film_brand": "kodak", "film_type": "colour_negative_cine",
                    "stage": "negative" if "film" in stem else "print_2383_rec709"})
        return rec
    claim = stock_claim(stem.replace("_", " "))
    if claim.get("film_stock") is None:  # hyphenated names (kodak-gold-200) need the hyphens as spaces
        claim = stock_claim(stem.replace("_", " ").replace("-", " "))
    stock, brand, ftype = claim.get("film_stock"), claim.get("film_brand"), claim.get("film_type")
    # camera-simulation packs and names the project canonicaliser misses take the override table
    if stock is None or slug == "sguyader_filmsim":
        for rx, s, b, ft in _OVERRIDES:
            if rx.search(stem.replace("_", " ")):
                if s is None:
                    slugged = re.sub(r"[^a-z0-9]+", "_", base_name.lower()).strip("_")
                    s = slugged if slugged.startswith(b) else f"{b}_{slugged}"
                stock, brand, ftype = s, b, ft
                break
    rec.update({"film_stock": stock, "film_brand": brand, "film_type": ftype, "film_stock_speed": claim.get("film_stock_speed")})
    return rec


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_lut(path: Path) -> bool:
    return path.suffix.lower() in (".cube", ".png", ".tif", ".tiff")


def build() -> dict:
    luts = []
    for slug, src in SOURCES.items():
        base = LIB / src["local_dir"]
        if not base.exists():
            print(f"WARNING: {base} missing", file=sys.stderr)
            continue
        files = sorted(p for p in base.rglob("*") if p.is_file() and (is_lut(p) or p.suffix.lower() == ".pp3"))
        for p in files:
            rel = p.relative_to(LIB).as_posix()
            stem = p.stem
            rec = {
                "id": f"{slug}/{re.sub(r'[^A-Za-z0-9]+', '_', p.relative_to(base).with_suffix('').as_posix()).strip('_').lower()}",
                "name": stem,
                "source_slug": slug,
                "source_name": src["name"],
                "source_url": src["source_url"],
                "author": src["author"],
                "license": src["license"],
                "license_url": src["license_url"],
                "license_quote": src["license_quote"],
                "commercial_permitted": src["commercial_permitted"],
                "attribution_text": src["attribution_text"],
                "approved_for_training": src["approved_for_training"],
                "path": f"data/lut_library/{rel}",
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
                "download_date": src["download_date"],
                "format": {".cube": "cube", ".png": "haldclut_png", ".tif": "haldclut_tif", ".tiff": "haldclut_tif", ".pp3": "rawtherapee_pp3"}[p.suffix.lower()],
                "input_color_space": src["input_color_space"],
                "output_color_space": src["output_color_space"],
                "technical_notes": src["technical_notes"],
            }
            category = p.parent.relative_to(base).as_posix() if p.parent != base else ""
            rec["category_path"] = category
            not_film = slug == "rawtherapee_film_simulation" and (category.startswith("Color/CreativePack-1") or stem in ("Negative", "Hald_CLUT_Identity_12"))
            rec.update(parse_name(stem, slug))
            if not_film:
                rec.update({"film_stock": None, "film_brand": None, "film_type": "not_a_film_stock",
                            "role": "identity" if "Identity" in stem else ("negative_inversion" if stem == "Negative" else "creative_look")})
            else:
                rec["role"] = "film_look"
            if rec["format"] == "rawtherapee_pp3":
                rec["validation"] = {"parsed": False, "ok": None, "note": "RawTherapee processing profile, not a LUT; kept with its source"}
            else:
                v = asdict(validate(p))
                v.pop("path", None)
                v.pop("header_comments", None)
                rec["lut_size"] = v.get("size")
                rec["validation"] = v
            luts.append(rec)

    # summary
    per_stock: dict[str, dict] = {}
    for r in luts:
        if r["role"] != "film_look":
            continue
        key = r["film_stock"] or f"unresolved:{r['film_brand']}:{r['base_name']}"
        d = per_stock.setdefault(key, {"count": 0, "sources": set(), "approved": 0})
        d["count"] += 1
        d["sources"].add(r["source_slug"])
        d["approved"] += int(bool(r["approved_for_training"]))
    for d in per_stock.values():
        d["sources"] = sorted(d["sources"])
    total_bytes = sum(r["bytes"] for r in luts)
    manifest = {
        "schema_version": 1,
        "generated": TODAY,
        "generated_by": "research/lut_library/build_manifest.py",
        "library_root": "data/lut_library (git-ignored)",
        "sources": SOURCES,
        "candidates": CANDIDATES,
        "summary": {
            "n_records": len(luts),
            "n_lut_files": sum(1 for r in luts if r["format"] != "rawtherapee_pp3"),
            "n_film_look_luts": sum(1 for r in luts if r["role"] == "film_look"),
            "n_approved_for_training": sum(1 for r in luts if r["approved_for_training"]),
            "n_validation_ok": sum(1 for r in luts if r.get("validation", {}).get("ok") is True),
            "n_validation_warn": sum(1 for r in luts if r.get("validation", {}).get("parsed") and r["validation"].get("ok") is False),
            "n_validation_unparsed": sum(1 for r in luts if r["format"] != "rawtherapee_pp3" and not r.get("validation", {}).get("parsed")),
            "total_bytes_on_disk": total_bytes,
            "candidates_total": len(CANDIDATES),
            "candidates_approved": sum(1 for c in CANDIDATES if c["status"] == "approved"),
            "candidates_rejected": sum(1 for c in CANDIDATES if c["status"] == "rejected"),
            "candidates_reference_only": sum(1 for c in CANDIDATES if c["status"] == "reference"),
            "per_stock": dict(sorted(per_stock.items())),
        },
        "luts": luts,
    }
    return manifest


if __name__ == "__main__":
    m = build()
    (OUT_DIR / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    (OUT_DIR / "candidates.json").write_text(json.dumps(CANDIDATES, indent=1, ensure_ascii=False))
    s = m["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "per_stock"}, indent=1))
    for k, v in s["per_stock"].items():
        print(f"{k:40s} {v['count']:3d}  {','.join(v['sources'])}")
