# FilmLook-ML research audit (2026-11-08)

Everything below was read from the repository and its execution environment on 2026-11-08 (ledger, registry,
backlog, checkpoints, manifests, reports). Where a number comes from a document rather than a file I re-read,
the document is named. Nothing was trained or re-run for this audit.

## 1. What was actually trained

**DINOv2-Small was never trained, fine-tuned or interrupted.** It is a frozen teacher (`facebook/dinov2-small`,
22.1 M parameters, 88 MB): its CLS++mean token and the mean/std statistics of layers 4 and 8 are extracted once
and cached (`data/embeddings/*_features_v1.npz`). The question "is DINOv2 training completed" has the answer
"there was no such training by design" (CLAUDE.md: the backbone stays frozen; Stage B off).

What was trained, all on the local RTX 4090, all MobileNetV3-Small students of 4.4 MB unless noted:

| checkpoint family | count | recipe | status |
|---|---|---|---|
| `student_style_fs_v2_real_f576{,_s1,_s2}` (E23b, session 5) | 3 | synthetic family + treated heads, SupCon-teacher distillation (64-d) from the frozen DINOv2 hybrid, 601 real CC-BY originals as unlabeled distillation inputs, 25 epochs | complete; held-out read E44 |
| `student_style_fs_v2_real_f576_deg{,_s1,_s2}` (E52, session 6) | 3 | the same plus re-encoding of half the crops (JPEG q40-70 / 0.5x rescale) | complete; **frozen** in `checkpoints/frozen_session7/` with SHA-256s; fresh-holdout read E53 |
| `_nokd`, `_nvic1`, `_svic1`, `_vic3`, `_pcaw`, `_kdsdxl`, `_sel`, `_l3`, `_s1/_s2` variants | 21 | ablations of the target (no teacher, VICReg on the pooled layer, whitened target, SDXL inputs, layer-3 teacher) | complete; all rejected (E30, E38/39, E43-E51) |
| `student_style_fs`, `student_style_both`, `student_style_fs_v2`, `student_kd` (sessions 1-3) | 4 | earlier synthetic-only students | superseded |
| `student_sup` | 1 | EXIF capture-type head (film vs digital), the authenticity baseline | kept, unchanged since session 4 |
| `student_imagenet_init` | 1 | the untrained ImageNet backbone exported for control (E48) | control only |
| `best` (1.1 MB), `dinov2_baseline_v0` (113 MB), `fused_224` (133 MB), `spatial_student_v0`, `stage_a`, `texture_ms1280` | 6 | the attributes classifier v1 (Claude categorisation labels under the recorded `user_confirmed` permission), frozen-DINOv2 linear heads, a fused baseline, the spatial probe, early stage-A artifacts | historical; not candidates |

39 checkpoint directories, 413 MB in all. Every student directory carries `meta.json` with the recipe
(`kd_target`, `teacher_layers`, `train_generators`, `aug_degrade`, `vic_pooled`, outputs, input size) and a
training report (`reports/student_<tag>.json`) with seed, epochs, loss curve, peak GPU memory and the synthetic
test readings; the three frozen candidates and the four comparators have their hashes in
`checkpoints/frozen_session7/freeze.json`. Reproducibility: the trainer command lines are recorded in
`research/backlog.yaml` (`command:`) and the registry; seeds 0-2 differ by name suffix.

## 2. Datasets, provenance, licences

| set | rows | real or synthetic | licence / provenance | use |
|---|---|---|---|---|
| `commoncatalog_ccby` | 7,371 manifest rows (819 originals + on-disk edits; **601 train originals**) | real Flickr photographs (CommonCatalog / YFCC100M) | CC-BY-2.0 checked per row | distillation inputs (no labels), PCA fit set, validation originals for probes |
| `pd12m_pilot` | 1,341 rows (149 originals) | real, mostly archival | CC0 / PDM per row | earlier pilots |
| `film_synth_v2` / `sdxl_i2i_v1` / `zimage_i2i_v1` / `synthetic_fixture` | 800 / 400 / 400 / 360 | synthetic treatments of CC-BY sources (deterministic engine; SDXL and Z-Image img2img, research-only) | derived from CC-BY sources; generator outputs not shipped | the students' only labels (family, treated) |
| `blueneg` | 491 | real 35 mm negatives, owner-recorded stocks, 27 rolls | CC-BY derivative, credit "Copyrighted by Tien-Tsin Wong" | evaluation only (benchmark D, read once E44) |
| `commoncatalog_film` | 905 | real film scans, uploader-claimed stocks, 252 uploaders | CC-BY-2.0 per row | evaluation only; both halves spent |
| `aic_photographs` | 593 | museum prints, process records | CC0 | evaluation only; both halves spent |
| `artefact_film` | 88 (44 originals) | real scans, one photographer, owner-recorded emulsions | CC BY 4.0 | read once (E53) |
| `met_photographs` | 698 | museum prints | CC0 | analysis half read once (E53); **test half (315) unread** |
| `commoncatalog_film_hd` | 474 | real film scans, 133 new uploaders | CC-BY-2.0 | read once (E53) |
| `commons_film` | 266 of 5,743 | Wikimedia Commons | per file | `permanent_failure` (HTTP 429/503), never used |

Real photographs on disk: about 3,700 originals across these sets; real photographs used for training: 601 (as
unlabeled inputs). Synthetic images: 1,960. No human annotation was requested anywhere; no Claude or Gemini output
is a training target for the style students (the only Claude-labelled model is the attributes classifier `best`,
under the recorded permission, and it plays no part in the style work). Unsplash Lite was never used; one yearbook
image is quarantined.

## 3. Objectives, pseudo-labels, augmentation

Student loss = cross-entropy on the synthetic treatment family (9 classes) + BCE on "treated" + distillation of
the teacher's 64-d SupCon projection (`kd_target supcon`, weight 1.0); real CC-BY originals enter only the
distillation term. Augmentations: random crops, flips, and since E52 re-encoding of half the crops. No pseudo-labels
on real photographs. Rank-control variants (VICReg on the pooled layer, whitened targets) were tried and rejected
because they re-admit scene content (E46, E47, E51).

## 4. Losses and in-training metrics

Training loss (sum of the three terms) over 25 epochs: 3.99 → 0.52 for the frozen seed 0 (`_deg`), 3.99 → 0.35
for the E23b seed; no separate validation loss is logged. Selection never used the training loss: it used the
synthetic validation split (family macro-F1 0.63 / 0.65, treated AUROC 0.85 / 0.81 on the synthetic test split)
and the real analysis halves. The low loss is a fit to synthetic labels and says nothing about real film; the
report states this because the owner asked that a low loss not be read as usefulness.

## 5. Evaluation metrics measured (where the evidence is)

`docs/FINDINGS.md` claims 1-40, `docs/EXPERIMENT_LOG.md`, `reports/e*.json`. The independent readings are in
`reports/MODEL_EVALUATION.md`. In one line: on fresh real photographs the frozen student's only statistically
supported gain over the untrained ImageNet backbone is within-photographer stock separation (+0.13 AUC); it ties
the 28 handcrafted statistics there and trails them on museum process; stock identification is at chance for every
representation.

## 6. Leakage, shortcuts, overfitting: what was checked

- Splits by uploader or artist hash; uploaders in the training manifest excluded from evaluation sets; near-duplicate
  rejection by perceptual hash at ingest; the test halves read once with pre-registered rules (E44, E53).
- Scanner confound found (E24) and audited (E56: the stock signal does not need the scanner); photographer leakage
  measured on every real set (artist/uploader share); content leakage measured as own-source rank-1 and uploader
  share; the untrained backbone used as the control for "what training buys" (E48).
- Overfitting to synthetic labels: synthetic benchmark gains did not transfer to real scans (claims 5-8, E29), and
  the session-5/6 selection on the analysis halves was followed by a fresh read that falsified two of six guards.
- Not checked: human judgement of "film look" (none exists); Gemini's analysis quality against ground truth (none exists).

## 7. Compute and disk

Ledger (`research/ledger.json`, metered runs since session 2): 7,335 GPU-seconds = **2.04 GPU-hours** over 57 runs,
largest single run 36 minutes (Z-Image generation); sessions 6 and 7 used about 17 GPU-minutes together. Disk:
`data/` 50 GB, of which 39 GB is the Hugging Face parquet cache (`data/hf_cache`), 6.7 GB processed images, 3 GB raw
acquisitions, 0.6 GB candidates; `checkpoints/` 413 MB; `reports/` 9.7 MB. No paid API or cloud GPU was ever used.
The GPU was idle at audit time (another user's job holds 5.7 GB).

## 8. Incomplete or unfinished items

- Backlog statuses not finalised although the work is done and documented: E8, E11, E12, E16, E17, E18 (`pending`
  in `backlog.yaml`, decided in the registry); E9 superseded; E13 and E27 deferred; E20 (second real film source =
  Commons) `in_progress` but the acquisition is `permanent_failure`; E21 (abstention threshold in shadow mode)
  blocked on that source.
- The Met test half (315 images) is the only unread real holdout.
- The CPU service (`service/`) was measured on the development host, not on the E2 Micro.
- No human similarity judgements have been collected; every metric is catalogue metadata or uploader claims.

## 9. Audit verdict

The research line is finished and honestly documented: a small, cheap, frozen embedding with one supported
real-world advantage and several measured limits, and no evidence that further training of this recipe would
change the picture (every intervention since session 5 was rejected or neutral on fresh data). Nothing is deployed.
