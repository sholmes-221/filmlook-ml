# LUT library work log (research/lut_library)

Agent: lut-library agent (autonomous session 2026-11-10). Scope: discovery, licence verification, download of
approved film-emulation LUTs into `data/lut_library/`, manifest + validator, source-photo selection for synthetic data.
Rules: <=1 req/s per host, UA `FilmAnalysisResearch/1.0 (LUT licence survey)`, no e-mail gates, no paywalls, <=3 GB,
disk at 92 % (71 GB free at start). Never edit `research/lut/`, never push, never commit `data/`.

## 2026-11-10
- 00 Read CLAUDE.md, DATASET_SOURCES.md, configs/datasets.yaml, docs/AUTONOMOUS_RESEARCH_STATE.md, film_physics/README.md
  (spektrafilm profiles/LUTs CC BY-SA 4.0 already reviewed there). Created research/lut_library and data/lut_library.
- 01 Web discovery started (EN/JA/ZH). Leads: G'MIC film emulation (Pat David), RawTherapee HaldCLUT pack,
  lucasbuick/Film-Luts (MIT claim), spektrafilm, janlohse/spectral_film_lut, bean-mhm/flim, t3mujinpack, Q-DDL 800 LUTs
  (CC BY claim), PictureFX CineStill 800T (itch.io), Fujifilm official LUTs, freevisuals.net, cinecolor.io, presetpro.
- 02 Licence findings so far:
  * RawTherapee Film Simulation Collection (rawtherapee.com/shared/HaldCLUT.zip, 421,602,289 B, sha256
    0ffca81f30c72d7bbb85adab0ed98bbdeb0e1034cd6f5986f9f993bb66999dc2): README.txt inside the archive states
    "CC BY-SA 4.0"; credits Pat David, Pavlov Dmitry, Michael Ezra. RawPedia page is 404 to our UA (vary: User-Agent);
    Wayback copy (2026-08-03) carries the same README text. APPROVED (share-alike caveat recorded). Extracted to
    data/lut_library/rawtherapee_film_simulation/HaldCLUT (296 files, 403 MB).
  * TrondW/glassmeter-film-looks (commit 6a65ee0): LICENSE = CC BY 4.0, README "Everything here is licensed CC BY 4.0".
    30 sRGB 33^3 .cube downloaded + README/LICENSE/AGENT_GUIDE/profiles json; git blob SHA-1 verified. APPROVED.
  * peterskotte123/filmlab-resolve (commit 7f0ca0f): LUTs CC BY-SA 4.0 (derived from spektrafilm profiles, SPEKTRAFILM_LICENSE.txt),
    input DaVinci Intermediate / DWG, output Rec.709. 8 of 28 LUTs downloaded (4 Vision3 stocks x film + 2383 rec709 print;
    CineStill 800T/50D are respooled Vision3 500T/50D). APPROVED with colour-space caveat.
  * PictureFX (Marc Roovers): base licence CC BY-NC 4.0 ("may not use the Luts ... for commercial purposes") -> REJECTED, not downloaded.
  * Fujifilm official LUTs: site terms (global.fujifilm.com/en/terms) personal non-commercial, no modification -> REJECTED.
  * Stuart Sowerby Fuji X-Trans III HaldCLUTs: no licence stated (pixls thread 6333) -> unverified, REJECTED.
  * G'MIC color_presets film packs: re-encodings of the RawTherapee collection + PictureFX; no CLUT licence on gmic.eu -> not downloaded (originals taken from RawTherapee instead).
  * YahiaAngelo/Film-Luts (MIT claim on RawTherapee/G'MIC CLUTs, disclaims ownership) -> relicensing not valid, REJECTED as a source (duplicate anyway).
  * Jack4B/Retro-Film-LUTs, MokshC/Film-Emulation-LUTs, Lts1sds/filmcam, cedeber/hald-clut: RawTherapee derivatives -> duplicates, not downloaded.
  * DocM88/Free-Lut-Pack (.xmp, no licence), cschmertz/rawtherapee-film-looks (no licence), sohan-shingade/fujiLUT (no licence) -> unverified, REJECTED.
  * Q-DDL 800 LUTs (CC BY claim via cgchannel): quickddl.net does not resolve -> unreachable, not verified.
  * HF NeoGraph/Luts_Cube (license "other", third-party commercial LUT names), liguanwei/luts (apache-2.0 tag, unknown origin), cgauss/comfy_luts (no licence) -> REJECTED.
  * Generators (no shipped LUTs): JanLohse/spectral_film_lut (MIT), andreavolpato/spektrafilm (GPL code, CC BY-SA profiles), bean-mhm/flim (AGPL, not stock-specific), ryancara/Spektrafilm-LUT-Generator (GPL) -> recorded as candidates, nothing downloaded.
- 03 validate.py written; validation run over all downloaded files.
- 04 sguyader/FilmSim (commit 1453b2b): LICENSE = CC0 1.0 (author: "I release them for Public Domain" on pixls.us thread 1797).
  4 HaldCLUT TIFFs (Classic Chrome / Provia Std / Velvia Vivid v2 / Acros, Fujifilm camera looks) + 4 pp3 downloaded,
  git blob SHA-1 verified. APPROVED.
  Other checks: cyangraphics/clut (CC BY-SA mirror incl. PictureFX + Juan Melara -> invalid relicensing, duplicate);
  Juan Melara print LUTs (no licence, log-scan input) -> unverified; abpy/FujifilmCameraProfiles (no licence) -> rejected.
- 05 validate.py: grey-axis monotonic tolerance set to 1 % of range (8-bit HaldCLUTs carry +-1 code noise); Negative.png
  (inverting) and a few Polaroid CLUTs with heavy clipping are flagged, nothing is modified.
- 06 build_manifest.py + source_photos.py written and run. Source-photo pool: commoncatalog_ccby (CC-BY-2.0, test split and
  quarantine excluded; digital/smartphone = tier 1, film scans/filtered = tier 2), pd12m_pilot (CC0), PD12M iNaturalist
  photo-gate images (CC0, only local birds/flowers source; photo_gate_split kept). Categories are heuristics (tags + pixel stats).
- 07 README.md (library layout, licences, coverage, loader, source photos) and docs/LUT_LIBRARY.md (owner summary) written.
  RawTherapee archive deleted after extraction (sha256 recorded; disk at 93 %). Totals: 341 records (337 LUT files + 4 pp3),
  572,394,151 B on disk, 334/337 validation OK (3 intentional inversions), 34 candidates (4 approved / 24 rejected / 6 reference).
  Source photos: 1,062 records (669 tier 1, 393 tier 2). Committing locally; never pushing.
