# Retrieval versus classification for FilmLook (2026-11-08)

The question: should the Gemini-free analysis be a **retrieval** system (nearest reference images, aggregate
their metadata) or a **classifier** (a head that outputs a label or a score), and over which labels?

## What the data allow

| target | evidence | verdict |
|---|---|---|
| rendering / process family (4 classes: albumen, gelatin silver, chromogenic, other) | 5-NN on the 28 statistics 0.85 balanced accuracy on fresh museum data (E57); hybrids equal; learned alone 0.58 to 0.69 | **supported** as a coarse "rendering family" signal; it is not a photographic-style label a user would recognise, but it proves the pipeline |
| film-stock line (23 classes) | best 0.12 balanced accuracy vs 0.043 chance, CIs touching chance (E57); p@5 0.11 to 0.15 vs 0.08 (E53) | **not supported** for identification; within one photographer's work the rendering signal exists (AUC 0.68) |
| film scan vs digital capture | `student_sup` head: AUROC 0.78 [0.69, 0.88] within source, ECE 0.20, scanner-dependent (reports/film_authenticity_eval.json) | **weakly supported**; the only score-like signal; needs calibration and a fresh read |
| edit attributes (temperature, saturation, contrast, key, sharpness) | heads 0.81 to 0.89 accuracy on held-out originals against known edits; grain 0.59 | **supported as measured characteristics** (they describe the rendering, validated by construction) |
| photographer style | no legal reference photographs; the site's profiles are text vectors; artist share shows embeddings *can* retrieve the same artist but that is leakage, not style | **not supported** as a learned task; stays the measured 17-axis ranking |
| aesthetic quality | no label; public aesthetic predictors answer preference, not film-likeness | **out of scope** |

## Retrieval or classification?

- For the **supported coarse labels** (rendering family, film vs digital), a *classifier* over the statistics
  (linear or 5-NN) is simpler, faster and as accurate as retrieval, and its evidence is explainable ("the
  highlight roll-off and the grain statistics sit with gelatin-silver prints").
- For **"looks like" without a label**, retrieval is the only honest tool: it returns neighbours and their
  *licensed* metadata, and abstains when the neighbourhood disagrees (the E55 gate: 3 of 5 agree and a cosine
  threshold). But retrieval is only as good as the reference bank's labels and licences, and FilmLook's bank today
  is small (the fresh scan snapshot, 324 images, is evaluation data, not a bank).
- For **film-stock resemblance**, neither works across photographers today. The honest product output is a
  *rule-based* suggestion within a small set of defined looks, built from the measured characteristics
  (monochrome → Tri-X look; warm saturated negative → Gold look; neutral low-contrast colour → Portra look;
  saturated contrasty → Velvia look; low key with tungsten cast → CineStill look), worded as "the look closest
  to this photo's measured character" and never as the film used. This is a deterministic mapping, not a model,
  and it is the same five directions the free tier already uses.

## Decision

Hybrid, in this order of trust:

1. **Measured characteristics** (28 statistics + the validated attribute heads) are the backbone of every
   statement.
2. **Classification heads** only where evidence exists: rendering family (statistics, 5-NN or linear) and film
   vs digital (student_sup, after Platt calibration on a fresh fit set). Both labelled with their accuracy.
3. **Retrieval** as a *similar-look* feature over a licensed bank, with abstention, surfaced as "looks similar
   to" neighbours' metadata only when the gate passes; never as identification.
4. **Rules** for film-look style suggestions within the library of defined directions.
5. **Templates** for text, filled from 1 to 4.

What is rejected: a learned film-look *score* head (no label exists), cosine-to-score conversions, stock
identification, photographer identification, and any text generation model in production.

## K and weighting (from E57)

- Rendering family: K = 5 plain vote is best (0.85); class-balancing lowers accuracy on this catalogue-skewed
  set and should be used only if the product needs equal treatment of rare families.
- Stock: no K rescues it; do not ship.
- Prototypes (class centroids) are within 0.01 to 0.04 of kNN on the rendering task and need no bank at serving
  time (4 vectors): the cheapest deployable form of the classifier.
- Confidence: vote shares are over-confident (ECE 0.17 to 0.28). Any displayed confidence needs a calibration
  map fitted on a separate set; until then show the label with its *measured accuracy*, not a per-image
  probability.
