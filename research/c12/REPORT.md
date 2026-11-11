# C12 result: open-set look ranker — gate FAILED, not shipped (2026-11-12)

Protocol and gate: `PREREGISTRATION.md` (committed e1f864d before any test number). Data: `build.py` (every feature
through the website's own code path: `prepare_image` -> `load_image` -> `describe`, live `FilmStyleEngine.analyze`,
photo-gate crop for DINOv2). Analysis: `openset.py`. Numbers: `reports/c12/{select,test}.json` (git-ignored).

Data: 52,204 pair renders (615 C08 sources, same split), 325 unknown LUTs in 64 families (50/20/30 family split per
kind); unseen pool 60 Common Catalog scenes that are not C08 sources (3,648 images); 1,139 real test-split photos (959
colour) x 8 conditions; 110 edited uploads.

Finding while preparing: the C08 "unseen image pool" had 52 of its 81 scenes among C08 *training* sources, so the
published 0.51 / 0.24 are not unseen-image numbers. Here the pool is restricted to the 60 clean scenes.

Validation selection (rule: max val top-1 + val non-B&W AUROC, deployable inputs): production descriptor (205) +
DINOv2 blocks 4/8, MLP with an explicit `unknown` class, score 1 - p(unknown). Known-only heads with MSP / energy were
worse than chance for unknown detection (val AUROC 0.41-0.47; Mahalanobis 0.61-0.63); the explicit class reached
0.75-0.76. The 312 research colour values were not better than the production descriptor.

| Test | Candidate | Engine (method D) |
|---|---|---|
| Known top-1, test sources [CI] | 0.294 [0.272, 0.316] | 0.174 [0.155, 0.193] |
| paired diff | +0.120 [+0.097, +0.144] | |
| top-3 | 0.544 | 0.412 |
| Known top-1, clean pool (60 scenes) | 0.379 [0.327, 0.432] | 0.252 [0.221, 0.285] |
| paired diff | +0.127 [+0.079, +0.176] | |
| Unknown AUROC non-B&W, test sources | **0.727** | 0.569 (margin) |
| ... colour film / creative / B&W | 0.758 / 0.648 / 0.865 | 0.58 / 0.55 / 0.61 |
| Unknown AUROC non-B&W, pool | 0.897 | 0.616 |
| Edited uploads vs known (AUROC) | 0.405 | 0.672 |
| ECE (14 known looks, test sources) | 0.032 | - |
| Operating point (test, known + non-B&W unknown) | coverage 0.25, accuracy 0.26 | coverage 0.71, accuracy 0.085 |
| Real colour photos: abstain / largest look | 35.9 % / 16.9 % (Candido 800T) | 31.1 % / 28.1 % (`none`) |
| Size / latency | 10.39 MB / 3.3 ms | - |

Robustness (change of top-1, candidate vs engine): JPEG50 -0.032 vs -0.002 (fail), half -0.024 vs +0.005, crop60
-0.042 vs +0.005 (fail), WB+8 -0.023 vs -0.015, WB-8 -0.012 vs -0.001, EV+1 -0.013 vs +0.023 (fail), EV-1 -0.034 vs
-0.036. Real-photo top-1 agreement under perturbation is lower than the engine's in every condition (crop 0.67 vs
0.73, EV+1 0.63 vs 0.74).

Gate: 1 pass/pass; 2 FAIL (0.727 < 0.80); 3 pass; 4 FAIL (JPEG50, crop60, EV+1); 5 pass/pass; 6 size FAIL (10.39 MB,
float32 x 5 seeds; fp16 would fit but that is a post-hoc change), latency pass. Decision: DO NOT SHIP.

Other observations: on test sources the pre-registered abstention rule rejected 62 % of known-look renders (the val
threshold on 1 - p(unknown) targeted 10 %; the extra `argmax = unknown` rule and a val-to-test shift add the rest), so
the operating point would hide most correct answers. On real edited uploads the candidate is *more* confident than on
known looks (0.41), i.e. failure (a) persists outside LUT renders. Real film still has no look labels: abstention and
spread are plausible, which is necessary, not sufficient.
