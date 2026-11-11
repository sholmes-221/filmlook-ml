# C12 pre-registration: open-set look ranker (written before any test number was computed)

Date: 2026-11-12. Data: `research/c12/build.py` (features through the website's own code path). Analysis:
`research/c12/openset.py`. Nothing below is changed after test numbers are read; deviations are reported as such.

## Data and splits
- Known classes (14): the 13 owner presets + `none`. Sources and their split: C08 (`data/c08/pairs/sources.json`,
  399 / 106 / 110 sources, split by uploader before rendering, dHash guard). Same render seeds as C08.
- Unknown edits: the 325 other analysable library LUTs (227 colour film emulations, 66 B&W, 32 creative), grouped in
  families (film stock, or creative-look stem; e.g. all Portra 400 variants of every collection are one family) and
  split per kind by a fixed hash: 50 % train / 20 % val / 30 % test families. Train sources only see train families,
  val sources val families, test sources test families: every test-time unknown edit is a LUT family never trained on.
- Unseen image pool: Common Catalog preset renders (`data/lut/synth`) restricted to scenes that are not C08 sources and
  whose uploader is not a C08 source uploader (60 scenes). Finding made while preparing: the C08 "unseen pool" (81
  test scenes) contains 52 C08 training sources, so its 0.51 is not an unseen-image number. The engine is re-scored on
  the same 60-scene pool. Unknown edits on the pool: test-family LUTs applied to each scene's untouched render.
- Real photos: the film / digital set used by `research/claude_annotations/film_eval.py`, test split only (1,139 rows).

## Candidates and selection (validation only)
- Deployable inputs: production descriptor `filmlook.filmstyle.descriptors.describe` (205 values; computed per upload
  today), optionally + DINOv2 blocks 4 and 8 [CLS, mean] (`PhotoGate.embedding_layers()[:1536]`, each 384-chunk L2
  normalised). Ablation, not deployable without porting: the 312 research colour values.
- Heads: (i) MLP with an explicit 15th `unknown` class (trained on train-family unknown renders); (ii) MLP on the 14
  known classes only with max-softmax, energy or Mahalanobis (penultimate layer, shared covariance) rejection;
  (iii) the live engine (method D) with its own `uncertain` rule, as shipped.
- Selection rule: among deployable (input, head, score) combinations, maximise val known-look top-1 + val unknown AUROC
  (non-B&W unknown edits). Final model = 5-seed logit average of the selected configuration, temperature-scaled on val.
- Abstention rule of the candidate: abstain when the photo is near-monochrome (the engine's `chroma_p50 < 0.04` rule),
  or the known-look score is below the threshold that accepts 90 % of val known-look renders, or (head i) when
  `unknown` is the argmax.

## Test metrics
- Known-look top-1 / top-3 (argmax over the 14 known classes, no abstention) on test-source varied renders and on the
  unseen pool; 95 % bootstrap CI over sources / scenes; paired bootstrap difference to the engine on the same images.
- Unknown-edit AUROC: known-look renders vs test-family unknown renders on the same test sources; primary = non-B&W
  unknowns (B&W, creative, colour-film and pool subsets reported). Engine score: the better on TEST of -(best
  distance) and margin (generous to the engine).
- Accuracy-coverage curve on test known + unknown renders (an accepted unknown counts as wrong).
- ECE (15 bins) of the top-1 confidence over the 14 known looks on test-source known renders.
- Robustness: change of known-look top-1 under JPEG q50, 0.5x resize, centre crop 60 % (of each side), white balance
  +8 % / -8 % (R up, B down in linear light, and the reverse), +1 / -1 EV, applied to the uploaded pixels.
- Real photos (no labels): abstention rate, distribution of the top look among non-abstained photos (`none` counts as
  a look), rank stability (top-1 agreement and abstain agreement) under the same perturbations.

## Deployment gate (all must pass)
1. Known-look top-1: paired bootstrap difference (candidate - engine) 95 % CI lower bound > 0 on test sources AND on
   the unseen pool.
2. Unknown-edit AUROC (non-B&W, test sources) >= 0.80 and greater than the engine's.
3. ECE <= 0.05.
4. For each of the 7 perturbations: candidate's change of top-1 >= engine's change - 0.03.
5. Real photos, colour rows of the test split: abstention between 20 % and 90 %; no single look > 50 % of the
   non-abstained photos.
6. Model file < 10 MB; CPU latency < 50 ms per image single-threaded (features already computed).
Decision: ship only if 1-6 all pass. Otherwise nothing changes in production.
