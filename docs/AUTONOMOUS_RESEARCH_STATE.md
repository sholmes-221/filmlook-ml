# Autonomous research state

Resume from this file, `docs/RESEARCH_BACKLOG.md` (rendered from `research/backlog.yaml`), `research/registry.jsonl`
and `research/ledger.json`. Conversation context is not the record. Validated claims with their evidence are in
`docs/FINDINGS.md`; the research model card is `docs/MODEL_CARD.md`; the selection instrument is `docs/BENCHMARK.md`.

Last updated: 2026-11-07, end of the second autonomous session (benchmark v1, E19, B1, E10).

## Current objective
A lightweight photographic-aesthetics representation that transfers to real photographs: explainable attributes,
cross-photo film-style similarity, badge attributes, later photographer-style retrieval and composition. Film
authenticity is a separate task with the EXIF student as its unchanged baseline. The standing question is whether
the representation captures treatment rather than content, generator fingerprints or dataset shortcuts; benchmark
v1 answers it per candidate.

## Hardware and budget
RTX 4090 24 GB shared (another job holds ~5.7 GB); 32 cores; 61 GB RAM. Session 2 used about 22 GPU-minutes
(17 metered) and 0.11 GB of the 20 GB allowance; both sessions together 81 metered GPU-minutes, 0.33 GB
(`docs/RESOURCE_USAGE.md`). Nothing paid.

## Best verified models (all proxies; no human labels anywhere)
| task | model | number to quote | source |
|---|---|---|---|
| film authenticity (EXIF labels) | `checkpoints/student_sup` capture head | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20 | `reports/film_authenticity_eval.json` |
| film-aesthetics attributes | v1 measured-statistics model `checkpoints/best`; deterministic attributes in `filmlook_infer` | shipped; edit orderings 0.994 | `REPORT.md` |
| cross-photo style similarity (research) | hybrid (explicit + DINOv2 L4+L8) + family SupCon | benchmark v1 test: B fs 0.44 [0.39, 0.48], rank-1 0.33, gen share 0.59; BlueNeg same-stock p@5 0.25 [0.16, 0.31] (chance 0.12) | `reports/style_benchmark_test.json` |
| abstention for the above | top-1 cosine similarity, answer the 50 % most confident | BlueNeg rank-1 0.22 → 0.37 at 50 % coverage; AURC 0.59 vs 0.78 | `reports/abstention_test.json` |
| composition | frozen DINOv2 2x2 patch-grid probe | signed offset R^2 ~0.23; mirror 0.18 | `spatial_probe.json` |
| lightweight student | `student_style_fs_v2` (4.4 MB, 1.2 ms) | best on synthetic (B fs 0.55, rank-1 0.16) but BlueNeg 0.18 [0.13, 0.23] = DINOv2 reference; not promoted | `style_benchmark_test.json` |

## What session 2 established
- Benchmark v1 (Tests A-E, val selection / test once) with a cached feature pass: raw DINOv2 global is a content
  embedding (A2 0.06, rank-1 0.99); every retained candidate is style-dominated on synthetic data but keeps some
  content (rank-1 0.16-0.37) and generator signal (share 0.59-0.63 vs prior 0.40).
- Val and test agree within 0.05 on every column; the test split is now spent for this candidate set.
- Only the hybrid family-SupCon embedding is above the DINOv2 reference on real scans; synthetic rank does not
  predict real rank (the student wins on synthetic and ties the reference on real).
- Loss-side content removal (E19: all-generator labels, same-source hard negatives, content decorrelation) does
  not help: generator share and subject leakage rise. Rejected.
- Diffusion img2img treatments are not style labels: at strength 0.35 two treatments of one source differ by less
  than JPEG+rescale nuisance; at 0.5 (E10, 500 new images) each image changes more but same-treatment retrieval
  across scenes stays at chance and content preservation drops (0.97 → 0.82). Retired as labels at any strength.
- A label-free confidence (top-1 similarity) orders the research embedding's errors on real scans; the student's
  does not.

## Known limitations
- One real-film retrieval source (BlueNeg: one photographer, 1990s consumer stocks, 27 rolls; CI floor ±0.08).
  No human aesthetic judgement anywhere; no claim about beauty, stock identity or photographer identity.
- 70 training sources: the residual content/generator signal in the projection is a data limitation.
- Composition barely decodable; students not promoted; shadow-mode logging awaits a privacy review.

## Active hypotheses / pending
- E20 second real-film source (blocked on rights; highest value). E21 abstention threshold on shadow traffic
  (blocked on privacy review). E13 scaling synthesis stays deferred (E8, E10). See `docs/RESEARCH_BACKLOG.md`.

## Rejected approaches (do not repeat without a new hypothesis)
- E3 both-generator pristine-control training; E2/E4 label-tied grain; E4 C1/C2 standardisation or VAE control alone.
- E9/E9b single best DINOv2 layer; E11 parameter-similarity SupCon; E12 spatial-student geometry regression;
  E16 texture-neutralised DINOv2; E17 rich colour-rendition descriptor; "treated vs untreated" as a target (E18).
- E19 content-invariant projection by loss design; E10 diffusion treatment ids at higher strength.

## Stopping point and next recommended action
Three consecutive rejections (E17, E19, E10) and a spent test split: further embedding variants on the present
data are not expected to move the real-scan number, and the benchmark would need a fresh test set to evaluate
them honestly. Next, in order:
1. Obtain a second rights-cleared real-film source with stock records (or self-reported stock through the app's
   existing training_labels flow, after the owner decides) — E20. Re-run D on both sources with the hybrid SupCon
   embedding, the student and the colour/tone statistics; keep BlueNeg as a held-out check.
2. If a second source arrives, hold out fresh sources for a benchmark v2 test split (the v1 test is spent) and add
   a human-free perceptual check of "same treatment" (e.g. agreement between the three generators' identity
   controls) to the B definition.
3. Product work meanwhile uses the deterministic attribute layer plus the shadow-mode record with the abstention
   threshold from `docs/BENCHMARK.md`; nothing is deployed and the EXIF authenticity baseline is unchanged.
