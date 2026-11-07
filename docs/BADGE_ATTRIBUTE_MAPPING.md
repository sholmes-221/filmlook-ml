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
