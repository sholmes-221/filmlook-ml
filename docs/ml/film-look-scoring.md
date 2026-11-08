# Film-look scoring: what can be justified, and the narrower definition the MVP uses (2026-11-08)

## The existing score cannot be called film-likeness

The website's provider backend shows a "Film Look Score" written by a vision-language model against a rubric;
no ground truth exists for it. This repository's v1 model (`checkpoints/best`, `film_score` 0 to 100) was trained
on a heuristic teacher and on synthetic edit orderings; it ranks edits of the same photograph consistently
(0.99 within-photo agreement) but **does not separate real film scans from digital captures**: AUROC 0.49
[0.35, 0.65] on the independent E58 READ half, 0.62 on the within-source test originals (REPORT.md already
warned that the heuristic teacher tracks film capture at chance). A number that cannot tell film from digital
must not be presented as how film-like a photograph is.

## What can be justified today

The supervised student's capture head gives P(film scan). On independent data it separates film scans from
digital captures at AUROC 0.86 [0.79, 0.92]; after a two-parameter Platt map fitted on a disjoint half it is
calibrated to ECE 0.07; under JPEG q50 and 0.5x rescale the 50 % decision never flipped on the READ half and the
estimate moved by 0.8 and 2.7 points on average (p95 3.7 and 11.9). Its training labels are EXIF capture medium
and owners' records: weak, but independent of any model and of any human rating made for this project.

## The narrower, honest definition

**Film-likeness estimate (胶片感估计, experimental): the calibrated probability that the photograph's rendering
resembles real film scans more than digital captures, read by a local model from the image alone.**

Presentation rules (`RetrievalMVP`):

- a rounded percentage **and** a band sentence (很像真实胶片扫描的成像 / 比较接近 / 介于胶片与数码之间 / 更接近数码 /
  明显是数码成像), because the p95 shift under heavy resampling is about 12 points;
- the reference figures (AUROC, CI, ECE, label type) travel with every result (`film_likeness.reference`);
- the disclaimer says it is a resemblance, not proof of the medium, and that no camera, stock or photographer is
  identified;
- the six v1 rendering attributes are kept as *observations* that drive the strengths and weaknesses text
  (clipped highlights, crushed shadows, sharpening halos, uniform overlay grain), not as a score.

## What is deliberately not done

- No cosine-to-score conversion; no similarity-based "score".
- No learned "aesthetic" head: public aesthetic predictors measure preference, not film rendering, and no
  film-look label set exists.
- No re-use of the provider's score as ground truth: its distribution on the same photographs may be compared for
  completeness and consistency only.

## Gate for showing the estimate (met on E58, with caveats)

AUROC ≥ 0.75 with lower CI ≥ 0.65 (0.86, 0.79); ECE ≤ 0.10 (0.07); flips ≤ 5 % under JPEG q50 and 0.5x rescale
(0 %). Caveats: 30 negatives in the READ half; positives dominated by one fresh snapshot; a fresh digital set is
the next evaluation. Until then the UI label keeps the word "estimate" and the About page explains it.
