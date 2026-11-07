# Dataset sources, licensing and teacher authorization (v1)

Last verified: **2026-11-06**. Machine-readable registry: `configs/datasets.yaml`. Only entries with
`training_allowed: true` *and* a per-record license on the allowlist can enter training;
`scripts/prepare_dataset.py` enforces this per record and writes rejects to `data/quarantine/rejected.jsonl`.
A Hugging Face dataset tag is never treated as overriding the underlying image copyright.

This is an engineering record, not legal advice. Items marked ⚠ should be confirmed by counsel before a
public release.

## Used for v1 training (1,008 source photos, 9,072 images incl. edits)

| Dataset | Source | License (checked per record) | Commercial training | Originals used | Labels / metadata | Access method | Attribution |
|---|---|---|---|---|---|---|---|
| CommonCatalog CC-BY | https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by (YFCC100M Flickr photos) | **CC-BY 2.0** per row (`licenseurl` must be `creativecommons.org/licenses/by/…`) | yes (CC-BY allows commercial use) | **819** (320 film scans, 260 digital cameras, 120 smartphones, 59 smartphone + filter tags, 60 digital + filter tags) | EXIF make/model, user tags, BLIP-2 caption, uploader | metadata columns, then per-image parquet *page* range reads through `huggingface_hub` (no Flickr scraping; about 800 MB fetched) | required: author, page URL and license URL kept per image in `data/raw/commoncatalog/ATTRIBUTION.jsonl`, shipped as `TRAINING_DATA_ATTRIBUTION.jsonl` |
| PD12M | https://huggingface.co/datasets/Spawning/PD12M | CC0 / Public Domain Mark per row | yes | 149 | captions | metadata parquet + S3 image URLs, 1 req/s | not required (kept anyway) |
| Synthetic scenes | generated in-repo | n/a | yes | 40 | edit family | generated | none |

**Capture medium** (CommonCatalog only) is inferred from metadata:
- **film scan**: the EXIF device is a film scanner or minilab (Fuji Frontier SP-2000/3000, Noritsu QSS, Pakon, Coolscan, Epson Perfection, Plustek OpticFilm …);
- **digital camera / smartphone**: camera EXIF with a capture timestamp;
- **filtered digital**: owner tags such as vsco, instagram, hipstamatic, vintage, faux;
- **ambiguous**: film tags on digital-camera EXIF; excluded.

It is metadata, not a judgement of aesthetics, but it is independent of every teacher.

⚠ Risks recorded in the registry:
- Some Flickr uploads may not be the uploader's own work (uploader-asserted licenses).
- Licenses are as of the YFCC100M snapshot. CC licenses are irrevocable for copies obtained under them.
- BLIP-captioned non-photos (screens, drawings, posters) are filtered by caption, but not perfectly.

## Session 4 additions (2026-11-07/08)

Real-film discovery and acquisition are recorded in `docs/REAL_FILM_DATASETS.md` (registry: `data/manifests/discovery_registry.json`); curated-photography platforms and their terms in `docs/CURATED_PHOTOGRAPHY_SOURCES.md`. New registry entries: `commoncatalog_film` (CC-BY film scans with uploader stock tags, evaluation-only), `artefact_film` (CC BY 4.0, evaluation-only), `aic_photographs` (CC0, curated group). PhotoVogue, DEHAZED and Iconic Artist: metadata only, never crawled (terms prohibit automated access / ML use).

## Rejected or not used

| Source | Reason |
|---|---|
| Wikimedia Commons API | robots.txt disallows `/w/` and `/api/` for generic agents. Not used (one `robots.txt` read only). |
| Openverse API | robots.txt disallows `/v1/images/` and explicitly blocks AI crawlers. One search request was sent before reading its robots.txt; nothing was downloaded, and it is not used. |
| CommonCatalog CC-BY-SA | Share-alike scope for trained models unresolved; not needed. |
| Unsplash Lite | Terms allow ML training only for "internal business purposes"; v1 serves a public product. `training_allowed: false`. |
| Unsplash Full | Non-commercial. |
| AVA | No license grant; third-party uploads via unofficial mirrors. |
| HF re-uploads of Unsplash, `*/photo-aesthetics`, TAD66K mirrors | Card licenses cannot relicense third-party photos. |
| PhotoVogue, Instagram, Pinterest, photographer portfolios | Never scraped. |
| Film-stock and photographer reference sets | None licensed. v1 therefore **removed** photographer-name matching and film-stock similarity. |

## Claude API as an offline teacher: authorization provenance

Governed by `docs/AI_DATA_POLICY.md` (mandatory; see `CLAUDE.md`). Status on 2026-11-07: **uncertain → blocked**.

- **What the project owner stated** (Claude Code session, 2026-11-06): they *"have confirmed with Anthropic that Claude API outputs may be used to train this specialized FilmLook model."*
- **Reference:** no written reference or ticket number was supplied, and none is invented here.
- **Where it is recorded:** `configs/labeling.yaml` → `claude.training_authorization`, as `status: confirmed_by_project_owner`, `reference: null`, `scope: offline teacher labeling and evaluation assistance for the FilmLook film-aesthetics model only`.
- **How it is applied:** `filmlook_ml.compliance.policy.claude_training_permission` classifies this record as **uncertain** because no written reference exists. Claude labels are stored with `usable_for_training: false` and a structured `provenance` (`training_permission: uncertain`); the training preflight (`scripts/preflight_check.py`, also run by every training/evaluation entry point) refuses to let them carry weight, and paid Claude runs declared `--intended-use training` are refused. Evaluation-only runs additionally need `claude.evaluation_use.approved` with a name and date.
- **If a written reference is later issued:** set `FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF` (or `training_authorization.status: written_reference` + `reference`); the status becomes `permitted`.
- ⚠ **Background:** Anthropic's Usage Policy prohibits using outputs to train an AI model *without prior authorization*; the Commercial Terms prohibit training *competing* models; the Help Center lists "using Outputs as training targets" as prohibited while allowing non-competing classifiers. Quotes, dates and URLs are in the policy, section 2.
- **Not covered:** permission to train on Claude outputs does **not** cover the photographs. Only images from registry-approved datasets (`claude.consented_datasets`) may be uploaded to the API.
- **Runtime:** Claude is never used at inference.
- **Status in v1:** no Claude labels exist yet. `ANTHROPIC_API_KEY` was not available in the session, and the project owner chose to proceed without Claude. The budget cap is US$5, cumulative.

## Photographer-style module (js/filmlook-style)

- **No photographs by Saul Leiter, William Eggleston, Daido Moriyama, Alex Webb, Fan Ho, Stephen Shore, Fred
  Herzog, Rinko Kawauchi or Ernst Haas were downloaded, used for training or calibration, or shipped.** Their
  works are under copyright and no licensed reference set exists.
- **Profiles:** short textual descriptions of widely published visual characteristics, expressed as targets on
  measurable dimensions (`src/filmlook_ml/style/profiles.py`). Names are used descriptively, and every output carries
  a "not authorship / not affiliation" disclaimer.
- **Rarity calibration:** score quantiles from 601 CommonCatalog CC-BY photos by ordinary uploaders (train
  split). Only aggregate numbers ship in `model-data.js`.
- **Flickr photographer-name tags** were scanned as a possible weak label (metadata only, 161 shards). There was
  1 hit in about 800k photos, so the scan was stopped and nothing was downloaded. Tagged photos of exhibitions,
  prints or books would have been excluded as reproductions of the photographer's work.
- **Test fixtures:** CC0 (PD12M) and synthetic images only, generated locally and git-ignored.

## Quarantine (configs/quarantine.yaml)

| image | reason | effect |
|---|---|---|
| `f993c25b62208870_original` (Flickr 9733492355, CC-BY-2.0) | phone photo of a printed 1967 school-yearbook page; the uploader's CC-BY may not cover the underlying photograph | the image and its 8 edited derivatives are excluded from annotation (`discover`) and from future training (`training.data.load_manifest`); the file stays on disk |

Found in categorization pilot 1 (2026-11-07). Models built before that date (the film-score model in `checkpoints/best`
and the Mini Tool's rarity quantiles) included it in their training data or calibration; they were not rebuilt. CC-BY or
public-domain status alone does not settle the rights to reproductions of prints, artworks or other people's
photographs.
