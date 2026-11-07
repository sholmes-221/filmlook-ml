# Curated photography sources (dataset group B)

Last updated: 2026-11-08. Group B is kept separate from group A (real analog film, `docs/REAL_FILM_DATASETS.md`) and
group C (local synthetic film simulations). Nothing in this group is assumed to be film, and editorial selection is
never treated as ground truth of photographic quality.

## Platforms investigated (terms read before any access; no crawling, no pixels, no embeddings)

| platform | what it is | terms found | decision |
|---|---|---|---|
| **PhotoVogue** (vogue.com/photovogue) | Condé Nast's curated photography platform (Pic of the Day, Best of PhotoVogue, open calls) | Condé Nast User Agreement: no automated access; no use of the Service "in connection with the development, training, fine tuning, grounding ... of any large language model, foundation model, deep machine learning" without prior written consent; robots.txt disallows query URLs and `*/c_limit/` image paths. Photographers keep copyright; open-call terms grant Condé Nast publication rights only. | **metadata only**: public landing URL recorded in the discovery registry; no pages crawled, no images, thumbnails or embeddings. A publication licence to a gallery is not a training licence for FilmLook. |
| **DEHAZED** (dehazed.eu) | German online gallery / print magazine run by ICONIC ARTIST (Munich); paid submission tiers; annual photography grant | Squarespace site whose robots.txt lists AI crawlers; submitting photographers retain rights; no licence covering ML use anywhere on the public pages | **metadata only**; not crawled |
| **Iconic Artist** (iconicartist.eu) | print-sales gallery of the same operator | same as above | **metadata only**; not crawled |
| 1x.com, VSCO, Unsplash, Pexels | curated or community platforms found during search | Unsplash Lite terms: internal-use only (registry); Unsplash Full: non-commercial; Pexels/1x/VSCO: not examined in this pass | not used |

Searches in Chinese (策展 摄影 平台 开放授权 / 摄影作品 CC 授权) and Japanese (キュレーション 写真 プラットフォーム / 写真 CC ライセンス)
surfaced no openly licensed editorial collections; stock-image sites (写真AC, PIXTA) have their own licences and were not used.

## Openly licensed alternatives

| source | licence | content | status |
|---|---|---|---|
| **Art Institute of Chicago, Photography and Media, public domain** (`aic_photographs`) | CC0 (AIC Open Access; `is_public_domain` per record) | 3,777 works with images; museum records of artist, date, process (gelatin silver, albumen, salted paper, platinum, chromogenic ...); mostly 1840s–1920s (Stieglitz, Atget, Emerson, Fenton, Cameron, Talbot, Hine ...) | **acquired 2026-11-08: 593 images, 207 artists (max artist share 1.3 %), 1830s-1920s**, capped 8 per artist via the public API + IIIF 843 px renditions (`scripts/ingest_aic_photographs.py`); the API caps a query at 1,000 results, so the listing is partitioned by decade |
| The Met, Photographs department | CC0 for public-domain works | similar era; objects endpoint works, search endpoint returned HTTP 410 on 2026-11-07 | deferred (open-access CSV route) |
| Smithsonian Open Access | CC0 subset | 2.8 M items incl. photographic collections; bulk S3 metadata | deferred (per-unit filtering) |
| Wikimedia Commons Featured/Quality pictures | per file CC | community-curated modern photography | deferred: this client is throttled by Wikimedia (HTTP 429) |

## What group B can and cannot support

- Can: cross-source robustness of the style embeddings (museum digitisation is a nuisance that differs from Flickr
  scans and from synthetic renders); process retrieval with artist-excluded queries (owner records of the physical
  print process); unsupervised clustering by interpretable colour/tone/texture features; a distillation-input domain
  for students (E23-style), CC0 permitting training.
- Cannot: modern editorial aesthetics (the CC0 pool is historic), famous-photographer "style" labels (an artist name is
  not a style label and is never auto-assigned to a cluster), any claim about beauty or objective quality.

## Planned experiment (pre-registered before any group-B result is read)

E29 `curated_cross_source_robustness`: on the `aic_photographs` analysis half, (a) process retrieval p@5 with the
artist excluded for the fixed candidates and eligible students (chance = class prior), (b) artist-share top-5 as the
photographer-leakage guard, (c) agreement of student and teacher embeddings on these images (Spearman of pairwise
cosines, as E22). Falsified for a candidate if its artist-share excess over the hybrid baseline exceeds 0.02 or its
process p@5 lower CI is below chance. No promotion follows from this group alone.

## E29 result (2026-11-08)

Process retrieval (artist excluded) is above chance for the hybrid teacher (0.532 [0.453, 0.596] vs 0.389) and the synthetic-only students (0.48-0.51) with low artist leakage (0.04-0.10 vs prior 0.017); raw DINOv2-global and colour statistics score higher but leak the artist (0.21-0.22). The three real-distilled students fall to 0.445-0.468 with lower CIs below chance and are falsified under the rule. Mixed result; the set stays evaluation-only; see `docs/EXPERIMENT_LOG.md` (E29).

## Dataset limitations

Historic processes only; museum photography of prints (not negatives); strong period confounds (subject matter and
process are correlated with decade); artist caps keep any single photographer below 2 % of the pilot.
