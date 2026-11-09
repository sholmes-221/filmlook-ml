# Independent check of the film-likeness estimate (2026-11-09)

Model: `filmlook_infer.retrieval_mvp` (student_sup capture head, Platt calibration from E58), unchanged.
Data: `commons_film_digital_manifest.json`, 183 Wikimedia Commons photographs: 92 film scans in ten declared-stock
categories (Portra 160/400, Gold 200, Ektar 100, Superia 200 / X-TRA 400 / X-TRA 800 / Reala, HP5 Plus, Tri-X) and 91
digital captures from eight camera categories; ≤ 4 files per uploader per category; 1024 px thumbnails for both classes.
Labels come from the categories (uploader-declared), never from appearance. Results: `commons_film_likeness_results.json`.

| | AUROC | 95 % bootstrap CI |
|---|---|---|
| MVP film-likeness estimate, all 183 | 0.783 | 0.717 – 0.849 |
| MVP estimate, colour only (159) | 0.729 | 0.652 – 0.807 |
| Baseline: monochrome flag | 0.620 | 0.575 – 0.668 |
| Baseline: fine-noise level | 0.613 | 0.535 – 0.690 |

Calibration does **not** transfer: every digital photograph scored ≥ 0.60 (the band "renders fairly close to a film
scan"); mean estimate 0.86 for film, 0.75 for digital. The migration gate's "ECE ≤ 0.10 on independent data" fails here.

Reading: the capture head carries real signal (above both baselines, also without black-and-white), weaker than E58's
0.86, and its absolute values are not usable as a user-facing number. Do not ship the estimate as a score or band; a
rank-only use (for example "more film-like than most of your photos") or a recalibration on a separate,
uploader-disjoint set would have to be validated first.

Caveats: small set; Commons film scans and digital captures differ in subject and processing; thumbnails are
re-encoded by Commons; uploader categories can be wrong. This is evidence against deployment, not a full evaluation.
