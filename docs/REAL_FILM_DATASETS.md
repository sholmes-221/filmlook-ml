# Real analog-film datasets: discovery, verification, acquisition

Last updated: 2026-11-08 (session 7: second CommonCatalog snapshot for the fresh holdout; earlier text from session 4). Machine-readable records: `data/manifests/discovery_registry.json`
(`python -m research.dataset_discovery report`), dataset gate: `configs/datasets.yaml`, readiness:
`python -m research.data_pipeline status`. This is an engineering record, not legal advice.

## What the research needs from real film

The selection instrument (`docs/BENCHMARK.md`) and the evaluation pipeline (`research/eval_pipeline.py`) need real
scans with (1) a stock label that is at least an uploader claim, (2) many independent uploaders so the query's uploader
can be excluded and confidence intervals can be bootstrapped over uploaders, (3) scanner or process metadata for the
leakage guards, and (4) a licence that permits local evaluation. Film authenticity needs real scans without stock labels
too. Nothing here requires human annotation, and nothing is labelled from pixels.

## Authenticity classes (evidence-based; never a visual classifier)

| class | meaning | real-film pool |
|---|---|---|
| A | verified real-film scans: documented negatives/slides, owner or institutional records | yes |
| B | probable: credible analog metadata (scanner EXIF, documented stock, consistent provenance) | yes |
| C | unverified analog claims | quarantined |
| D | film simulations (digital captures processed to look like film) | synthetic experiments only |
| E | synthetic / AI-generated | synthetic experiments only |

Licence: CC0/PD/CC-BY may be ingested (training per `configs/datasets.yaml`); CC-BY-SA is evaluation-only (as for
`commons_film`); NC/ND/unknown are never ingested (metadata and links only). Classes C/D/E never enter the real pool.

## Discovery (multilingual, API-based)

`research/dataset_discovery.py` searched Hugging Face Hub, Zenodo, figshare and GitHub with the English, Chinese and
Japanese query lists from the brief (117 platform/query pairs plus 15 Hub tokens), serial requests with spacing, no
HTML scraping, no logins. Web searches in the three languages were run by the agent for institutional archives and
Kaggle. Results: 187 sources recorded; the bulk are keyword false positives (thin films, grain crops, cameras as 3-D
objects, cinema, audio) and were rejected after review. Chinese and Japanese queries returned no film-scan datasets on
any platform (Zenodo returns Japanese 3-D scans tagged "フィルム" by camera brand; Chinese results were generic
deep-learning dataset round-ups). The registry prevents re-searching a query within 7 days and keeps every decision.

## Verified sources

| source | class | licence | size | status | use |
|---|---|---|---|---|---|
| **CommonCatalog CC-BY film scans with uploader stock tags** (`commoncatalog_film`) — Flickr/YFCC100M rows whose EXIF capture device is a film scanner or minilab and whose uploader tagged a stock | B | CC-BY-2.0 per row (checked) | all 1,076 shards of the 1024-2048 px range scanned (5,130 qualifying rows); caps 20/uploader, 6/uploader×stock, 150/stock | **acquired: 905 scans, 252 uploaders, 88 uploader-claimed stocks** (two snapshots: 487 on 11-08 00:15, 905 on 02:23); evaluated twice (E27b) | evaluation-only; uploaders in the training manifest excluded; analysis/test halves by uploader hash |
| **ARTeFACT** damaged & manually restored 35mm scans, Ivanova et al., Eurographics 2023 (`artefact_film`) | A | CC BY 4.0 (figshare 21803304) | 44 originals + 44 restored, 11 emulsions (Portra 160/400, Ektar 100, Gold 200, Ultramax 400, Ektachrome 100, Provia 100, Velvia 50, Superia 400, CineStill 50D/800T), one photographer | **acquired 88/88** (md5 + sha256 verified) | authenticity and per-emulsion sanity checks (within-photographer only) |
| **BlueNeg** (ICCV 2025), 491 colour negatives with roll records | A | CC-BY derivative, credit Tien-Tsin Wong | 491 | acquired (session 1) | benchmark v1 test; blue-corrupted partition retired for selection (E26) |
| **CommonCatalog CC-BY film scans, second snapshot** (`commoncatalog_film_hd`, session 7) — the 2048-4096 px shards (never scanned before), uploaders of the first snapshot (both halves) and of the training manifest excluded, caps 30/uploader, 10/uploader×stock, 150/stock | B | CC-BY-2.0 per row (checked) | scan-limited: the large shards yield about two qualifying rows each and one in seven from a new uploader | **acquiring** (session 7); see `docs/EXPERIMENT_LOG.md` session 7 for the final count | fresh held-out evaluation only (E53, read once); `training_allowed: false` |
| **Wikimedia Commons film-brand categories** (`commons_film`) | B | per file CC0/PD/CC-BY/CC-BY-SA | 5,743 items, 266 stored | acquiring; Wikimedia throttles this client (HTTP 429, Retry-After honoured); deprioritised, retries with doubling backoff | evaluation-only |

## Recorded but not ingested

| source | class | reason |
|---|---|---|
| GitHub DhaliwalX/gold200-dataset (Kodak Gold 200 frames scanned per LED + matching digital captures, CC BY-SA 4.0) | A | 3 frames only; kept as a paired film-vs-digital reference |
| Zenodo 22170495 grainifier frame statistics (197 Double-X frames, Coolscan 4000 dpi, CC BY 4.0) | A | statistics (npz), no pixels |
| Zenodo 14283642 Robert Breitinger estate 1886–1910 (Zentralbibliothek Zürich, public domain, 3.3 GB) | A | deferred: plate era, no stock labels; candidate for cross-era authenticity |
| figshare 21803292 empty Lomochrome Purple frames | A | artefact-only frames, no scenes |
| Kodak PhotoCD test suite mirrors on the Hub (24 images) | B | legacy "unrestricted usage" wording, no formal licence; too small |
| GitHub dstrub/DataSetFilmCorection (QCAV 2019 film scans for artefact removal) | C | no licence file; quarantined for owner follow-up |
| Hub `pslabs/35mm_32k`, `Nbardy/film_stock`, `wornleto/grain` | C | no licence, no card, unknown provenance |
| FilmSet (IJCAI 2023; Capture One film recipes on digital photos) | D | simulation; licence unclear; never mixed into real-film evaluation |
| HISTORIAN / HistShot (digitised cinema reels) | A | motion-picture frames, out of scope |
| Hub Flickr/Pexels portrait re-uploads with permissive card tags | C | a card tag cannot relicense third-party photographs |
| Smithsonian Open Access (CC0 subset; public S3 bulk metadata) | A | deferred: needs per-unit filtering for photographic negatives; large |
| The Met Photographs department (CC0) | A | deferred: search endpoint returned HTTP 410 on 2026-11-07 |

## Standardised manifest fields (new datasets)

`image_id, source_dataset, source_url, original_image_url, photographer_id, uploader_id, film_stock, film_brand,
film_type, film_format, camera, scanner, scanner_category, scan_process, development_process, authenticity_class,
authenticity_evidence, license, license_url, commercial_use_allowed, training_use_allowed, image_width, image_height,
sha256, perceptual_hash, split` plus the fields the evaluation script reads (`uploader`, `kind`, `film_stock_claim`).
Unknown stays `null`; a brand-only tag gives `film_brand` without `film_stock`; a missing stock is never a negative label.
Originals are stored as received (no recompression, no colour correction); derived versions (ARTeFACT restorations) carry
`kind: film_restored` and `derived_from`.

## Shortcut prevention

Uploader-excluded retrieval and uploader bootstraps everywhere; per-uploader and per-stock caps; training uploaders
excluded from `commoncatalog_film`; scanner family recorded for the leakage guard; sha256 and perceptual-hash
deduplication at ingest (near-duplicates rejected); splits by uploader hash so no photographer spans both halves; the
test half is evaluated once (`--final`). Film authenticity (film vs digital) and stock retrieval stay separate metrics.

## Evaluation readiness and results (2026-11-08 02:40)

- `commoncatalog_film`: ready (905 rows); evaluated on the analysis half twice (`reports/eval_pipeline/history.jsonl`).
  Same-stock p@5 is at chance for every candidate on both snapshots (teacher 0.051 [0.036, 0.069] vs chance 0.044 on
  the second); E27b inconclusive twice, no student promoted, the L3 target closed (`docs/EXPERIMENT_LOG.md`).
- `artefact_film`: ready (88 rows); single photographer, so it feeds authenticity checks, not selection.
- `aic_photographs` (curated group): ready (593 rows); E29 run (`docs/CURATED_PHOTOGRAPHY_SOURCES.md`).
- `commons_film`: 266/5,743; stays partial while Wikimedia throttles (four attempts, all HTTP 429).
- Test halves of all new sets untouched.

## Limitations

- Stock labels are uploader claims (CommonCatalog, Commons) or owner records of one photographer (ARTeFACT, BlueNeg).
- No source offers many photographers *and* owner-verified stocks; the two are traded off deliberately.
- Chinese- and Japanese-language platforms yielded nothing reusable in this pass; the queries are kept for re-runs.
- Wikimedia Commons remains throttled; the project honours its Retry-After and does not change identities.
