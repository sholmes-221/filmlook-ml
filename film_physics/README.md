# film_physics: physics-informed film-stock research (started 2026-11-09)

A research hypothesis, not a product feature. Nothing here is deployed.

## 1. Engines reviewed (read-only; their code is not copied, imported or run here)

| Project | What it is | Licence | Linux / NVIDIA |
|---|---|---|---|
| andreavolpato/spektrafilm | Python spectral simulation: sensitivities, H&D curves, couplers, grain, halation, print, scan | code GPL-3.0; profiles and LUTs CC BY-SA 4.0 (`SPEKTRAFILM_LICENSE.txt`) | Python (numpy, numba, colour-science, OpenImageIO...): runs on Linux; a Vulkan port exists in vkdt |
| JamesQiu2005/spektralab-engine | C++ render engine with Metal kernels and an Objective-C++ GPU layer | GPL-3.0 | macOS / Metal only; would need a port |
| JamesQiu2005/SpektraLab | macOS SwiftUI app around the engine, a fork of spektrafilm | no licence detected by GitHub | macOS only |

Licence consequences for FilmAnalysis: GPL code may be used as a research tool on the workstation; linking it into
distributed software (an app, the mini tool) would make that software GPL. Profiles may be used commercially with
credit, and anything derived from them (LUTs, re-encodings) must stay CC BY-SA 4.0. Running spektrafilm itself is
blocked tonight by the session's permission policy (installing external code); the owner can allow it.

Measured versus adjustable: each profile's `log_sensitivity`, `density_curves`, `channel_density` and `base_density`
are processed from data sheets (the profiles say so); `density_curves_model` and `hanatos2025_*` are fitted models;
grain, halation, glare, couplers and scanner settings in the engines are rendering parameters, not measurements.

## 2. Forward model (`simulator/forward.py`)

Independent numpy implementation on the profile data (11 profiles in `profiles/spektrafilm/`, CC BY-SA 4.0, see
NOTICE.md): smooth spectral upsampling of the scene, exposure through the measured sensitivities, the H&D curves, dye
transmission, a three-band scan with per-channel log inversion for negatives, direct viewing for slides. No grain,
halation, couplers or print stage. The approximations are listed in the module docstring.

## 3. Identifiability (`inverse/identifiability.py`, result `experiments/identifiability_oracle.json`)

Oracle upper bound: 24 Commons digital scenes × 10 stocks, a random unknown edit and JPEG q85 per query; the scene is
known exactly and the same simulator made the query, so there is no model mismatch. Rank stocks by the residual after
fitting the edit class.

| Unknown edit | Top-1 | Top-3 | Chance | Median margin (RMSE) | JPEG noise (RMSE) | Negative vs slide |
|---|---|---|---|---|---|---|
| white balance + exposure (3 params) | 0.754 | 0.942 | 0.10 | 0.00069 | 0.0206 | 1.000 |
| + per-channel levels (6) | 0.633 | 0.892 | 0.10 | 0.00015 | 0.0206 | 0.975 |
| + 3x3 colour matrix (12) | 0.692 | 0.912 | 0.10 | 0.00008 | 0.0206 | 0.988 |

Reading: even in this best case the true stock wins by margins 30 to 250 times smaller than JPEG noise; it is found only
because every candidate sees the same noise and the same simulator. With the scene unknown, a real scanner and real
edits, these differences are far below what one image can carry. Exact stock identification from an arbitrary digital
image is not supported by this model; the family (colour negative versus slide) is robust. The defensible product
statement is resemblance to a film family or look, never the stock used.

## 4. Real photographs (`evaluation/`)

The existing film-likeness estimate on 183 independent Commons photographs: AUROC 0.78 [0.72, 0.85] (colour only 0.73),
above simple baselines (0.61 to 0.62) but miscalibrated (every digital photo ≥ 0.60). Not deployable as a score.

## 5. Next experiments (backlog)

1. Stock-level retrieval on real scans: the Commons manifest has 92 film scans in ten declared stocks; DINOv2-small
   kNN with uploader-disjoint folds vs chance, and whether simulator-derived response features (fitted curve slopes,
   cross-channel coupling) add information. Expected: near chance at stock level (section 3), maybe above at family level.
2. Recalibrate the film-likeness estimate on an uploader-disjoint split and test on a third set before any UI use.
3. Manufacturer data-sheet database (`datasheets/`): official PDFs' values with units and provenance, unknowns left
   unknown; only facts the profiles cannot already give.
