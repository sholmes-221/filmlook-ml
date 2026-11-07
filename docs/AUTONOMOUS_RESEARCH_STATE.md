# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`
and `research/ledger.json`. Conversation context is not the record.

Last updated: 2026-11-07 (autonomous loop, after E14).

## Current objective
A lightweight (MobileNetV3-class) photographic-aesthetics representation that transfers to real photographs:
explainable attributes, cross-photo film-style similarity, badge attributes, later photographer-style retrieval
and composition. Film authenticity stays a separate task with the EXIF student as its unchanged baseline.

## Hardware and budget (this session)
RTX 4090 24 GB shared with another job (~5.7 GB taken); 32 cores; 61 GB RAM; 83 GB free disk.
Session budget 4 GPU-hours and 20 GB disk, tracked in `research/ledger.json` by `research/experiment_runner.py`.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20; robust to JPEG/rescale/crop | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best` | shipped; edit orderings 0.994 | `REPORT.md` |
| style embedding (research) | SupCon on hybrid features / `student_style_fs_v2` | BlueNeg same-stock p@5 0.25 / 0.21 (chance 0.12) | `synth_experiment_v3.json`, `student_style_fs_v2.json` |
| cross-generator treated probe | hybrid features, filmsynth v2 | fs→SDXL 0.58 [0.52, 0.67], SDXL→fs 0.48 | `synth_experiment_v3.json` C3 |
| composition | DINOv2 2×2 patch grid probe | signed offset R² ≈ 0.23, mirror 0.18 | `spatial_probe.json` |

## Known limitations
- Cross-generator transfer is weak and asymmetric; only two generator families so far (third in progress).
- Real-film evaluation = BlueNeg (one photographer, 1990s stocks) + CommonCatalog EXIF scans; both weak labels.
- Grain/texture carried the real-film separation; colour/tone is the transferable aesthetics signal and is weak.
- Composition is barely decodable; badges for it stay off.
- Student models are not promoted; the EXIF student remains production baseline.

## Active hypotheses (see backlog for criteria)
E8 third generator (data generating at session start), E9 layer ablation, E10 SDXL strength, E11 parameter-similarity
embedding, E12 spatial student, E13 scale to 500 sources (blocked by E8).

## Completed this session
- E9/E9b layer ablation: neutral (depth trade-off documented; L4+L8 kept).
- E11 parameter-similarity embedding: rejected (real same-stock regressed 0.25 -> 0.21).
- E12 spatial student: rejected (framing R^2 < 0.1; frozen DINOv2 grid probe stays the reference).
- E8/E8b third generator: held-out Z-Image 0.63 [0.52, 0.75]; gains are diffusion<->diffusion; real film unimproved; E13 scaling deferred.
- E14 artifact probe: diffusion transfer is not a VAE signature (B 0.52).
- E15 colour/tone vs texture: 16 coarse colour/tone statistics transfer across generator TYPES (SDXL->fs 0.74, Z-Image->fs 0.71); texture = authenticity axis.
- E16 texture-neutral DINOv2: rejected (content-dominated, transfer worse).
- E17 80-d colour-rendition descriptor: rejected (scene fingerprint, transfer inverted).

## Rejected approaches (do not repeat without a new hypothesis)
- Training "treated vs untreated" on both generators with pristine-only controls: real-film regression (E3 `style_both`).
- Label-tied grain (film-synth-v1) as a transfer signal: generator fingerprint (E2/E4).
- Resolution/JPEG standardisation or VAE controls as the leakage fix alone: little effect (E4 C1/C2).
- Fine parameter-similarity contrastive objective (E11): fits synthetic geometry, hurts real-scan retrieval.
- Searching DINOv2 layers for a single best style layer (E9/E9b): within CI; mid layers = real texture, deep = generator-invariant.
- Deep or rich descriptors for generator-agnostic aesthetics (E16, E17): they re-encode scene identity; only coarse global statistics transfer.
- Direct geometry regression into a MobileNetV3 spatial student (E12).

## Resource usage
See `research/ledger.json`.

## Next recommended action
Run E9 while E8's data finishes; then E8 (v4 benchmark); decide on E13 scaling from E8's CI.
