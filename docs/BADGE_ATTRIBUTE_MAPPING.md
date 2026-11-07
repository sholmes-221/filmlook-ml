# Badge attribute mapping proposal (research, not deployed)

The FilmLook app credits badges from analysis records (`filmlook/collection/badges.py` in the app repository;
untouched here). This proposal maps candidate badge dimensions to **deterministic, explainable measurements** that
already exist in `filmlook_infer` and can be computed on the E2 host in milliseconds. It changes nothing in the app.

Principles:
- A badge rule must be reproducible from pixels alone, state its threshold, and never depend on a photographer or
  film-stock prediction (those are unvalidated).
- Thresholds come from `filmlook_infer.style.THRESHOLDS`, which were set on the training distribution; they are
  calibrated to population quantiles, not to human judgement. Rarity (how often an ordinary photo earns the badge)
  can be measured on the 967 originals and should be reported before any badge goes live.
- Composition badges are **not available yet**: pooled features do not encode layout (`docs/AESTHETICS_LEARNING_RESEARCH.md`
  section 3.7); they wait for the spatial branch.

| badge dimension | measurement (existing) | rule (deterministic) | status |
|---|---|---|---|
| Warm palette | `warmth` (R-B balance), colour images only | `warmth > THRESHOLDS.warm` (0.055) and not monochrome | ready |
| Cool palette | `warmth` | `warmth < THRESHOLDS.cool` (-0.015) | ready |
| Muted colours | `sat_mean` | `sat_mean < THRESHOLDS.muted` (0.17) and not monochrome | ready |
| Pastel tones | `sat_mean`, `y_mean`, `y_std` | muted **and** `y_mean > 0.55` **and** `y_std < 0.2` (bright, low contrast, low saturation) | ready; rarity to measure |
| Soft highlights | `highlight_shoulder`, `clip_high` | `clip_high < 0.002` and `highlight_shoulder` in the upper third of the training distribution | ready; threshold to set from quantiles |
| Deep shadows | `p01`, `clip_low` | `p01 < 0.03` with `clip_low < 0.01` (dense but not crushed) | ready; threshold to set |
| High contrast | `y_std` | `y_std > THRESHOLDS.high_contrast` (0.30) | ready |
| Fine grain | `noise_std`, `noise_autocorr1` | `THRESHOLDS.grain_none < noise_std < THRESHOLDS.grain_coarse` with `noise_autocorr1 > 0.4` (correlated, not sensor noise) | ready; the correlation criterion needs a rarity check |
| Strong geometric composition | spatial branch (not built) | — | **not available** |
| Layered street photography | spatial branch + genre | — | **not available** |
| Minimalist composition | spatial branch (negative space) | — | **not available** |

Evidence status: the measurements are deterministic and robust to JPEG re-encoding (`reports/style_stability.json`:
top-1 style tag unchanged 92-98 % under browser resampling and compression), but **no human has validated that
"warm" or "fine grain" badges match what users perceive**. Treat the rules as explainable proxies and expose the
measured value next to the badge in research outputs so the rule can be audited.

Interaction with existing badges: the app's film badges are credited from the current analysis; this mapping would
add attribute badges beside them, never replace or re-credit existing progress.

## Evidence update (research sessions 2-3, 2026-11-07)

What the benchmark and the real-film experiments say about each feature family the product could use. Levels follow
`docs/FINDINGS.md` (high / medium / low); "deployable" means it runs on the E2.1.Micro host (1 GB RAM, no DINOv2).

| feature family | product use | evidence | level | deployable | recommendation |
|---|---|---|---|---|---|
| 16 colour/tone statistics (`filmlook_infer.features`) | attribute badges above (warm, muted, pastel, contrast, highlights, shadows) | deterministic; JPEG-stable 92-98 %; lowest semantic content of all candidates (subject probe 0.25 vs 0.22 majority); **not** an authenticity signal (E15) and **not** a film-stock signal (BlueNeg same-stock 0.18 = DINOv2 reference) | high for "what it measures", none for perception | yes (ms) | use for attribute badges only; never label them "film" |
| 12 texture statistics (`noise_std`, `noise_autocorr1`, ...) | "fine grain" badge | deterministic; texture is the axis that separates film scans from digital (E15); explicit texture stats are inverted on real film vs the engine's grain | medium | yes | keep the grain badge rule, but report the measured value; re-check the rarity on real scans |
| EXIF student authenticity head (`checkpoints/student_sup`) | film-authenticity score beside aesthetics | within-source AUROC 0.78 [0.69, 0.88], ECE 0.20; not calibrated | medium | yes (4 MB) | keep as the authenticity baseline; show as a score band, not a probability |
| style student `student_fs_v2` (64-d) | "similar film look" retrieval, badge candidates from `family_logits` | best on synthetic (B 0.55) but at the DINOv2 reference on real scans (0.18 [0.13, 0.23]); E22: it does not reproduce its teacher on real photographs; E23b: real-image distillation raises real agreement (Spearman 0.53 -> 0.62) | low on real data | yes (4.4 MB, 1.2 ms) | experimental only; the E23b recipe is the path to a deployable similarity feature, pending the Commons evaluation |
| hybrid DINOv2 L4+L8 + family SupCon | research similarity | only candidate above the reference on real scans (0.25 [0.16, 0.31]); abstention at 50 % coverage lifts rank-1 0.22 -> 0.37 | medium | no (DINOv2 needs more than 1 GB) | shadow-mode / offline analysis only |
| film-stock or photographer identification | — | no evidence at any point; stock labels are one photographer's records or uploader categories | none | — | never shown as a claim |

Scoring: the app's film-aesthetic score should stay a function of the deterministic attributes plus the authenticity
score band; the learned similarity feature enters only as a "similar looks" suggestion with the abstention rule from
`docs/BENCHMARK.md`, and only after the real-film evaluation on a second source (`scripts/eval_commons_film.py`).
