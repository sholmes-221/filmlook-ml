# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`
and `research/ledger.json`. Conversation context is not the record.

Last updated: 2026-11-07, end of the first autonomous session (E8-E18).

## Current objective
A lightweight photographic-aesthetics representation that transfers to real photographs: explainable attributes,
cross-photo film-style similarity, badge attributes, later photographer-style retrieval and composition. Film
authenticity is a separate task with the EXIF student as its unchanged baseline.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session used 64 GPU-minutes of 240
and 221 MB of the 20 GB disk allowance (13 metered runs; the 33 GB Z-Image download preceded the budget).

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo style similarity (research) | hybrid (explicit + DINOv2 L4+L8) + family SupCon | BlueNeg same-stock p@5 0.25 [0.15, 0.31] (chance 0.12); E8b 0.26 [0.16, 0.34] | `synth_experiment_v3.json`, `synth_experiment_v4b.json` |
| composition | frozen DINOv2 2x2 patch-grid probe | signed offset R^2 ~0.23; mirror 0.18 | `spatial_probe.json` |
| lightweight student | `student_style_fs_v2` (4.4 MB, 1.2 ms) | BlueNeg 0.77 / same-stock 0.21; not promoted | `student_style_fs_v2.json` |

## What this session established
- The cross-generator "treated vs untreated" signal is not aesthetics: within a generator type it is a generator
  signature (E2-E4); across types it reduces to one statistic, saturation clipping (E15, E18).
- A third generator (Z-Image) generalises at 0.63 [0.52, 0.75] but improves nothing real (E8); not a VAE artifact (E14).
- Authenticity lives in texture (DINOv2 mid layers); colour/tone cannot tell film from digital (E15).
- Richer or deeper descriptors re-encode scene identity and transfer worse (E16, E17); fine parameter-similarity
  objectives overfit the engine (E11); a geometry-regressing spatial student fails (E12).
- The same-stock benchmark has wide CIs (27 rolls); the Cinestill paired dataset is unobtainable (repo 404).

## Known limitations
- One real-film retrieval source (BlueNeg: one photographer, 1990s consumer stocks). No human aesthetic judgement.
- Composition barely decodable. Students not promoted. Budget for Z-Image generation is high (5 s/image).

## Active hypotheses / pending
E10 SDXL strength 0.5 (value lowered by E18), E13 scaling (deferred). New candidates belong in `research/backlog.yaml`.

## Rejected approaches (do not repeat without a new hypothesis)
- E3 both-generator pristine-control training; E2/E4 label-tied grain; E4 C1/C2 standardisation or VAE control alone.
- E9/E9b single best DINOv2 layer; E11 parameter-similarity SupCon; E12 spatial-student geometry regression;
  E16 texture-neutralised DINOv2; E17 rich colour-rendition descriptor; "treated vs untreated" as a target (E18).

## Next recommended action
Obtain a second rights-cleared real-film source with stock records (or self-reported stock via the app's existing
training_labels flow) before further embedding work; then re-run same-stock retrieval with the hybrid SupCon
embedding and the student on both sources. Until then, product work should use the deterministic attribute layer
and the shadow-mode record.
