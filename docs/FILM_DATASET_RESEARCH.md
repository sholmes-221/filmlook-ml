# Public film-photography datasets: research for FilmLook-ML

Compiled 2026-11-07. Metadata and cards were fetched read-only from the Hugging Face API; licence files were read
where present. Machine-readable registry: `data_registry/film_datasets.json`. Nothing was downloaded unless its
status is `eligible`.

Rule applied throughout: a Hugging Face licence tag is a claim by the uploader, not evidence of rights. A dataset
is eligible only if the upstream rights of the *images* are documented.

## The four starting datasets

### yangkun2197/DigitalFilm_dataset and Richards-Sheehy-sudo/DigitalFilm_dataset
Identical cards and file lists (5 ZIPs named `Kodak_Aerochrome`, `Kodak_E100`, `Kodak_gold_200`, ...; "1K<n<10K →
7K" images; MIT tag). They belong to the DigitalFilm CycleGAN project (github SongZihui-sudo/digitalFilm).
- Real film? The card says the **digital** side was shot on a Xiaomi 13 Ultra plus an HDR dataset, and the **film
  samples "are collected from the Internet"**. So the film-side images are third-party photographs of unknown rights.
- Film-stock labels: folder names only; unverified.
- Paired: no (unpaired CycleGAN data).
- **Status: not eligible.** The MIT tag cannot relicense scraped photographs. Not downloaded.

### ttgroup/blueneg-release (BlueNeg, ICCV 2025)
- **Real 35 mm colour-negative scans**: 491 negatives (16-bit DNG, 10,128 × 6,840) with 8-bit positive previews
  (1,322 × 892), 398 scanned prints (16-bit TIF, 2,994 × 1,920), and 247 pseudo-ground-truth images derived from
  the prints. 30 test items are withheld.
- **Film stock labels: yes, per negative, from the owner's own records** (`meta.json`, field `film_type`): Kodak Gold
  100-5 (226), Fuji NPH400 (53), Fuji 400 (45), Kodak Gold 400-5 (38), Konica Super XG 100 (30), Kodak Gold 100 (25),
  Fuji 100 (23), Kodak Gold 400 (19), Kodak GA 100 5095 (16), and 4 rarer Kodak Gold variants. Also date (1988-2002,
  mostly 1994-1999), roll id, location, indoor/outdoor, day/night.
- Caveats: 297 of 491 negatives are in the `blue-corrupted` partition (the blue layer has deteriorated), 194 are
  `blue-intact`. The dataset exists to study that deterioration, so colour on the corrupted half is not the film's
  original rendering. The stocks are consumer films of the 1990s; none is a modern Portra/Velvia/Tri-X.
- **Licence:** custom, dated 2024-06-12: all photographs are copyrighted by Tien-Tsin Wong; permission is granted
  automatically "for academic and commercial usages" provided the credit "Copyrighted by Tien-Tsin Wong" accompanies
  any publication, reproduction, redistribution or derivative; citation of the ICCV 2025 paper requested. The README
  says it shares the spirit of CC-BY 4.0 but is not CC-BY 4.0.
- Provenance: the owner shot and scanned the negatives himself; one photographer, mostly family and travel scenes.
- **Status: eligible** (commercial use with credit). Use: real-film reference and evaluation set with verified
  stock labels; the `blue-intact` previews as clean film references; the `blue-corrupted` ones as a real "archival
  deterioration" family. Downloaded: 8-bit previews, pseudo-GT and metadata only (about 1 GB); the 16-bit DNG/TIF
  originals were not fetched.

### joshuachin/openphoto-restore-dataset
- 5,000 pairs (pristine 1024 × 1024 vs synthetically damaged) for photo restoration. **Not film:** source images are
  modern digital photographs from the **Unsplash Lite** dataset; "film grain" is Gaussian noise; damage is procedural.
- Licence tag CC-BY-4.0, but the images are governed by the Unsplash Lite terms, which this project already classified
  as internal-use only (`configs/datasets.yaml`).
- **Status: not eligible** for FilmLook, and not useful (synthetic damage, no film).

## Other datasets found

| dataset | what it is | status |
|---|---|---|
| [Nbardy/film_stock](https://huggingface.co/datasets/Nbardy/film_stock) | 60 image+text rows, no card, no licence | not eligible (no licence) |
| [pslabs/35mm_32k](https://huggingface.co/datasets/pslabs/35mm_32k) | two ZIPs, no card, no licence; name suggests 32k scraped "35mm" images | not eligible |
| [msdkhairi/kodak](https://huggingface.co/datasets/msdkhairi/kodak) and mirrors | the 24-image Kodak PhotoCD test suite (classic compression benchmark; originally film, scanned by Kodak) | licence unclear on HF; 24 images; not used |
| FilmSet ([arXiv 2301.08880](https://arxiv.org/abs/2301.08880)) | 5,000+ images in three "film" styles produced with **Capture One film recipes**: synthetic, not film | not film; not pursued |
| Cinestill 800T paired set ([arXiv 2411.15967](https://arxiv.org/abs/2411.15967)) | 41 raw / 38 processed aligned digital-film pairs (Sony A7 + Nikon F3 on one tripod), one stock | **unobtainable**: the code repository named in the paper returns 404 and no dataset licence is published (checked 2026-11-07) |
| Film-damage scans ([arXiv 2302.10004](https://arxiv.org/abs/2302.10004)) | 44 4K damaged film scans with restorations | restoration data, licence not verified |
| ur2good/blueneg-release, yangkun2197/blueneg-release | mirrors of BlueNeg | use the original |

HF dataset searches for "film", "analog", "35mm", "negative", "filmstock", "lomography", "kodachrome" returned
mostly unrelated items (circuits, clocks, NLP negatives). There is no large, rights-documented, stock-labelled
collection of real film scans on the Hub.

## What FilmLook-ML already holds (real film, documented rights)

- **320 real film scans** from CommonCatalog CC-BY (Flickr), identified by scanner EXIF (Fuji Frontier, Noritsu,
  Pakon, Coolscan, ...), 28 of them with uploader-asserted stock tags; plus 439 digital-camera/phone originals as
  negatives. Rights: CC-BY 2.0 per record (`DATASET_SOURCES.md`).
- These remain the primary real-world transfer set. BlueNeg adds 491 scans from one photographer with owner-verified
  stock labels and metadata, a different scanner and era.

## Decision for the hybrid dataset

| source | role | labels |
|---|---|---|
| A. Real film: CommonCatalog film scans (320) + BlueNeg previews (491) | evaluation and real-photo adaptation; never synthetic-labelled | capture type (EXIF / owner record); BlueNeg stock labels (verified by the owner); CommonCatalog tags (asserted) |
| B. Generated: SDXL image-to-image treatments of FilmLook's own licensed originals | second generator family | treatment id only, `film_stock_claim: null` |
| C. Programmatic: `filmlook_ml.filmsynth` with knowledge-base ranges | first generator family | exact parameters and seed |

Splits: a generated or simulated image always inherits its source photo's uploader-grouped split. BlueNeg is a
single-photographer set, so it is used whole as an **external** evaluation set, never split into train/test.
