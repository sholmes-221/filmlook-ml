# Production migration plan: `ANALYSIS_BACKEND=retrieval` (2026-11-08)

State: production runs the provider backend (Gemini). The website already has an `ANALYSIS_BACKEND` switch with a
`filmlook_ml` mode that calls no external model and abstains from the score (web repo, `docs/ml-backend.md`). The
MVP in this repository (`filmlook_infer.retrieval_mvp`) is the thing that mode should run. Nothing below has been
deployed; the switch stays `provider` until the gate passes.

## Integration steps (website repository)

1. **Runtime dependency.** Add `onnxruntime` (CPU wheel) and `scipy` to the web image; vendor `filmlook_infer`
   (built by `scripts/build_package.py --verify` here: `dist/filmlook-infer-<ver>/`) into the web repository as a
   pinned package, together with the three artifact directories (`best`, `student_sup`, `retrieval_mvp`; about
   5.2 MB in all). Artifact paths come from one setting (`FILMLOOK_ML_MODEL_DIR`), never from a request.
2. **Adapter.** A `RetrievalFilmLookAnalyzer` in `filmlook/analyzer.py` that maps the v2 result onto the existing
   `FilmLookAnalysis` model the pages already render: `film_look_score` ← the film-likeness estimate × 100 with
   `score_available = True` **and a new `score_kind = "film_likeness_estimate"`**, so the UI labels it
   "胶片感估计（实验）" rather than the provider's "Film Look Score"; `film_matches` ← the one look suggestion
   (score = the estimate band, reason = the basis sentence) or an empty list on abstention; `digital_signals` ←
   weaknesses; `film_like_strengths` ← strengths; `summary` ← summary; `film_character` ← the six rendering
   attributes mapped to the existing character fields; `recommended_edit` ← none (no template for edits yet).
   The DNA side keeps the measured ranking (already what `filmlook_ml` does).
3. **Library rules.** `allowed_looks` = the tier's `library.film_ids(scope)`; the rules can only name those.
4. **Switch.** `ANALYSIS_BACKEND=retrieval` selects the adapter; `filmlook_ml` keeps meaning "measured only,
   abstain"; `provider` unchanged. A missing artifact makes the API refuse to start in production (like a missing
   key does today), never fall back to a provider.
5. **Schema and history.** Stored results keep `provider = "retrieval"`, `model = "<v1>:<hash>+<sup>:<hash>"`,
   prompt version `"mvp-2.0"`; the cache key includes them, so nothing old is reused or rewritten.
6. **Tests.** Adapter mapping, zero external requests in retrieval mode, allowlist enforcement, abstention
   rendering, badge and card flows on a retrieval result, quotas unchanged, test-account privacy unchanged.

## Gate (all must hold before the default changes)

| item | status 2026-11-08 |
|---|---|
| film-likeness AUROC ≥ 0.75, lower CI ≥ 0.65, ECE ≤ 0.10 on independent data | **met** on the E58 READ half (0.86 [0.79, 0.92], ECE 0.07) with the caveat that negatives are 59 test originals; a fresh digital set would tighten the CI |
| robustness flips ≤ 5 % under JPEG q50 and 0.5x rescale | **met** (about 5 %); the p95 shift of the estimate is up to ~24 points under rescale: show bands, not precise percentages |
| characteristics validated | met (0.81 to 0.89 on held-out originals); grain from statistics |
| look suggestion coverage and monochrome correctness | rules; tested for monochrome and abstention; coverage to be measured on the held-out photo set |
| instance benchmark p95 < 3 s, RSS < 200 MB | **not yet measured on the instance** (dev host: 0.3 s p95, 165 MB) |
| application tests incl. no-external-request | **not yet** (adapter not written) |
| comparison with provider output for completeness and consistency | **not yet** |

## Rollback

The switch (`ANALYSIS_BACKEND=provider`) and a container restart. Results made in retrieval mode stay readable
(they carry their own versions); nothing in the database changes shape.

## What the user will see differently

A film-likeness *estimate* with a band of wording instead of a provider-written score with sub-scores and
prose; measured characteristics instead of observations about mood or subject; one look suggestion worded as
resemblance (or none) instead of three ranked film matches with reasons; strengths and weaknesses from templates;
no edit recipe until a template set exists. The About and privacy pages change accordingly before the switch.
