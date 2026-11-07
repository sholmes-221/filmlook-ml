# Shadow-mode integration proposal (research interface; nothing deployed)

`filmlook_infer.shadow.ShadowAnalyzer` produces one JSON record per photo with every concept in its own field.
The FilmLook app would call it **beside** its current pipeline, log the record, and show nothing new until the
evidence justifies it. The app, its database, cards and badge progress are untouched.

## Record (`shadow-1.0`)

| field | source | status | never used as |
|---|---|---|---|
| `film_aesthetics` | v1 model (`checkpoints/best`): film score 0-100, six attributes, style tags, bilingual templates | shipped v1; unverified against humans | authenticity |
| `film_authenticity` | MobileNetV3 capture-type head (`student_sup`), `p_film_scan` | experimental; within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 (`reports/film_authenticity_eval.json`) | the aesthetic score |
| `visual_attributes` | deterministic measurements, style tags, `badge_candidates` (rules in `docs/BADGE_ATTRIBUTE_MAPPING.md`) with the raw values | deterministic; thresholds not human-validated | photographer or stock labels |
| `style_embedding` | synthetic-trained student (`student_style_fs`), unit vector | experimental; retrieval only | any label |
| `uncertainty` | per-field statements (`authenticity_calibrated: false`, `human_validation: none`) | always present | — |
| `model_versions`, `timing_ms` | provenance and cost | — | — |

Cost on the dev CPU, one thread: about 0.3 s for the v1 model plus 2-3 ms per MobileNetV3 head; peak RSS about
110-160 MB (`deployment/E2_MICRO_PLAN.md`). Fits the E2 Micro inside the existing app process; DINOv2 is not
involved.

## Rollout gates (none passed yet)

1. **Log only.** Store shadow records next to the app's analysis id; compare `film_authenticity.p_film_scan`
   with the user's self-reported capture type from the app's `training_labels` table as it accumulates (this is the
   first real, independent authenticity label the project would have; it is self-reported, not verified).
2. **Attribute badges** may be enabled when their rarity on real traffic is known and the measured values have been
   audited; composition badges wait for the spatial branch.
3. **Replace nothing.** The v1 film score stays; the authenticity head is shown only if its within-source AUROC and
   calibration improve on independent data; the style embedding stays internal until same-stock retrieval is
   validated on a second real-film source.

## What the record does not do

- It does not identify photographers or film stocks.
- It does not combine authenticity and aesthetics into one number.
- It carries no learned "aesthetic quality" score: there are no human preference labels.
