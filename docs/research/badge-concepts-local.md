# Local badge-concept recogniser: linear heads on the photo-gate embedding (E60, 2026-11-10)

Question: can validated local image analysis replace the vision provider's scene words (`_WORDS` in the website's
`filmlook/collection/context.py`) as the trigger for scene badges, precision first (a false unlock is worse than a miss),
at no extra model memory on the 1 GB Oracle VM?

Answer: for **22 heads feeding 21 badge keys, yes** on author-disjoint held-out Open Images photographs; for the other 25
heads plus 9 concepts without legal source labels, **no**: they are marked unsupported and must not unlock badges from
local vision. SigLIP-base zero-shot was clearly worse (5 concepts would pass the same rule) and is not recommended.

Artefact: `dist/concepts/concept-heads-20261110-8e3bfd84.json`
(sha256 `62df8315921fd0c34b143c599f80913a64632e09b7989a2bba099f56beeaab92`, 415 KB JSON, 22 x 768 weights), plus
`dist/concepts/TRAINING_DATA_ATTRIBUTION.jsonl` (27,068 training images: author, licence, source page). Both are
git-ignored. The website's `filmlook/local_vision.py` (`Heads.load`) loads the artefact unchanged; on 500 test vectors its
signals equal the NumPy reference (`research/concepts/predict.py`). Note: the website looks for `heads-*.json`, so it
has to be copied into `models/local_vision/` under a name starting with `heads-`, e.g. `heads-concepts-20261110-8e3bfd84.json`.

## Embedding the heads expect (identical to the photo gate, `dinov2s-cls-patch-768`)

`models/photo_gate/dinov2_small_cls_patch.onnx` (facebook/dinov2-small rev ed25f3a31f01, fp32) on
`dinov2_pixels(prepared JPEG)`: RGB, shorter side to 256 (bicubic), centre crop 224, /255, ImageNet mean/std, NCHW
float32; output [L2norm(CLS), L2norm(mean of 256 patch tokens)], 768 float32, used as is (not renormalised as a whole).
That is the vector `PhotoGate.embedding()` already returns. Each head: `p = sigmoid(w · v + b)` (Platt scaling from
validation folded into w, b); a head fires at `p >= threshold`; its output `signal` is the exact `_WORDS` key.
Training features were computed with the torch model in fp32 (TF32 off); against the deployed ONNX file on 200 images the
cosine is 1.00000 (max abs difference 2.4e-6).

## Data (labels from the source only)

- **Labels**: Open Images V7 human-verified image-level labels (CC BY 4.0, Google LLC). Positive = Confidence 1 for any
  of the concept's classes (`research/concepts/concept_map.py`); verified hard negative = Confidence 0. No LLM, no
  model, no eyeballing produced any label.
- **Images**: Open Images Flickr photographs whose listed licence is CC BY 2.0 (all selected rows), author / profile /
  source page / licence URL kept per image (`data/concepts/manifest.jsonl`). Fetched from the CVDF public S3 copy
  (27,713 images); 21,611 train-subset ids are not on that copy, for which the 640 px Flickr thumbnail listed in the
  Open Images metadata was used (17,946 fetched, 2,505 gone: HTTP 404/410). Every image was reduced locally to a 320 px
  short side; originals were not stored. Images that need rotation (about 1 %) were dropped. Generic project
  User-Agent, bounded worker pools (32 on the CVDF S3 copy, 6–8 on Flickr), Retry-After honoured on 429/503; no
  request ended as a rate-limit failure (retried 429s are not logged). 1.9 GB on disk.
- **Selection** (`research/concepts/select_images.py`): per head up to 650 positives (at most 4 per Flickr author) and 300
  verified negatives, plus 8,000 random Open Images validation/test photographs as generic negatives shared by every
  head: 45,659 images, 16,574 authors.
- **Splits**: by Flickr author (sha256), 60/20/20 train/val/test; no photographer is in two splits. Near-duplicates
  (both embedding halves cosine > 0.95, or 64-bit pHash Hamming <= 4, against any earlier split) removed: 53. Final
  27,068 / 9,555 / 8,983.
- **Negatives and missing labels**: Open Images verifies only candidate classes, so "no label" is unknown. An image that
  is a positive of head A counts as a negative for head B only if raters verified B on it, or if among A-positives
  verified for B fewer than 1 % are B (with at least 10 such images); otherwise it is excluded from B. Positives of a
  parent or near-synonym class (an `IGNORE` list per head in `common.py`, e.g. Plant for flowers, Darkness for night)
  are excluded from that head's negatives. Generic photos remain negatives at their base rate, so an unlabelled
  positive among them counts as a false positive: **test precision is a lower bound in that respect**.

## Method

Per head: logistic regression (class_weight balanced), C from {0.1, 0.3, 1, 3, 10} by validation average precision;
Platt scaling on validation folded into the weights. Threshold: the lowest validation score at which validation
precision is >= 0.95 at that and every higher threshold with at least 20 fires. Test read once
(`research/concepts/train.py`, `reports/concepts/heads_results.json`). Pre-registered ship rule (backlog E60):
test precision >= 0.90, Wilson 95 % lower bound >= 0.80, recall >= 0.30, at least 20 test positives. The weights that
ship are the train-split fit whose threshold was chosen on val and measured on test (no refit).

Development runs (`--no-test`) looked only at train/val; two changes were made from them before test was read: the
co-occurrence rule for unknown negatives (the first selection treated, e.g., unverified sunset photos as moon negatives)
and the minimum of 20 fires for the threshold envelope (8 made one early false positive veto a head).

## Results (test split, read once)

"FPR generic" = share of generic test photos labelled negative (about 1,100–1,350 per head) on which the head fires;
with that many, 0 observed means an upper 95 % bound of roughly 0.003. "P at generic prevalence" re-weights test
precision to the concept's prevalence among the generic photos (– where that prevalence is 0); it rests on very few
generic false positives and is noisy, but it is the better guide for rare concepts in real uploads. SigLIP columns:
the zero-shot comparator, same rows, same threshold rule.

| concept | `_WORDS` key | test pos | test P | Wilson 95 % LB | test R | FPR generic | P at generic prevalence | threshold | decision | SigLIP test P / R | SigLIP decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beetle | beetle | 135 | 0.926 | 0.848 | 0.556 | 0.0000 | – | 0.916 | **shipped** | 1.000 / 0.170 | unsupported |
| bird | birds | 383 | 0.969 | 0.944 | 0.822 | 0.0009 | 0.968 | 0.589 | **shipped** | – / – | unsupported |
| boat | boat | 315 | 0.944 | 0.908 | 0.749 | 0.0018 | 0.904 | 0.752 | **shipped** | 1.000 / 0.149 | unsupported |
| cabin | cabin | 148 | 0.934 | 0.864 | 0.574 | 0.0000 | 1.000 | 0.784 | **shipped** | 1.000 / 0.155 | unsupported |
| construction | construction | 135 | 0.971 | 0.900 | 0.496 | 0.0000 | 1.000 | 0.879 | **shipped** | 0.914 / 0.237 | unsupported |
| desert | desert | 119 | 0.956 | 0.891 | 0.723 | 0.0000 | 1.000 | 0.797 | **shipped** | 1.000 / 0.160 | unsupported |
| dock | dock | 134 | 0.929 | 0.853 | 0.582 | 0.0000 | 1.000 | 0.787 | **shipped** | 1.000 / 0.134 | unsupported |
| flowers | flowers | 853 | 0.954 | 0.935 | 0.773 | 0.0000 | 1.000 | 0.682 | **shipped** | 0.936 / 0.652 | would_ship |
| fox | fox | 133 | 0.936 | 0.874 | 0.774 | 0.0000 | 1.000 | 0.800 | **shipped** | 0.952 / 0.752 | would_ship |
| golden_hour | golden_hour | 176 | 0.977 | 0.921 | 0.489 | 0.0000 | 1.000 | 0.928 | **shipped** | – / – | unsupported |
| ice | ice | 141 | 0.928 | 0.851 | 0.546 | 0.0017 | 0.353 | 0.902 | **shipped** | 0.900 / 0.255 | unsupported |
| lotus_pond | lotus_pond | 142 | 0.945 | 0.891 | 0.852 | 0.0000 | 1.000 | 0.676 | **shipped** | – / – | unsupported |
| motorcycle | motorcycle | 178 | 0.942 | 0.860 | 0.365 | 0.0000 | 1.000 | 0.931 | **shipped** | 0.876 / 0.556 | unsupported |
| power_plant | power_plant | 75 | 0.934 | 0.843 | 0.760 | 0.0000 | 1.000 | 0.733 | **shipped** | 0.895 / 0.453 | unsupported |
| pumpkin | pumpkin | 134 | 0.983 | 0.910 | 0.433 | 0.0000 | – | 0.966 | **shipped** | 0.970 / 0.478 | would_ship |
| rickshaw | rickshaw | 66 | 1.000 | 0.857 | 0.348 | 0.0000 | – | 0.984 | **shipped** | – / – | unsupported |
| small_animal | birds | 415 | 0.987 | 0.955 | 0.376 | 0.0000 | 1.000 | 0.943 | **shipped** | – / – | unsupported |
| spider_web | spider_web | 103 | 0.959 | 0.899 | 0.903 | 0.0000 | 1.000 | 0.617 | **shipped** | 0.957 / 0.437 | would_ship |
| starry | starry | 108 | 0.908 | 0.813 | 0.546 | 0.0000 | 1.000 | 0.786 | **shipped** | 0.851 / 0.528 | unsupported |
| sunflower | sunflower | 126 | 0.988 | 0.936 | 0.659 | 0.0000 | 1.000 | 0.863 | **shipped** | 0.960 / 0.190 | unsupported |
| tunnel | tunnel | 150 | 0.960 | 0.903 | 0.647 | 0.0000 | 1.000 | 0.906 | **shipped** | 0.968 / 0.400 | would_ship |
| wheat | wheat | 80 | 0.943 | 0.846 | 0.625 | 0.0000 | 1.000 | 0.903 | **shipped** | – / – | unsupported |
| blue_hour | blue_hour | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| car_window | car_window | 151 | 0.938 | 0.799 | 0.199 | 0.0000 | 1.000 | 0.819 | **unsupported** (Wilson lower bound 0.799 < 0.8; recall 0.199 < 0.3) | 0.973 / 0.238 | unsupported |
| corridor | corridor | 119 | 1.000 | 0.816 | 0.143 | 0.0000 | 1.000 | 0.941 | **unsupported** (recall 0.143 < 0.3) | – / – | unsupported |
| diner | diner | 150 | 0.939 | 0.804 | 0.207 | 0.0000 | 1.000 | 0.952 | **unsupported** (recall 0.207 < 0.3) | – / – | unsupported |
| dragonfly | dragonfly | 154 | 0.966 | 0.828 | 0.182 | 0.0000 | 1.000 | 0.992 | **unsupported** (recall 0.182 < 0.3) | 1.000 / 0.266 | unsupported |
| duck | duck | 123 | 0.971 | 0.855 | 0.276 | 0.0000 | 1.000 | 0.967 | **unsupported** (recall 0.276 < 0.3) | – / – | unsupported |
| fishing | fishing | 122 | 0.880 | 0.700 | 0.180 | 0.0000 | 1.000 | 0.747 | **unsupported** (test precision 0.880 < 0.9; Wilson lower bound 0.700 < 0.8; recall 0.180 < 0.3) | 1.000 / 0.238 | unsupported |
| fog | fog | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | 1.000 / 0.111 | unsupported |
| gas_station | gas_station | 89 | 0.854 | 0.761 | 0.787 | 0.0007 | 0.000 | 0.526 | **unsupported** (test precision 0.854 < 0.9; Wilson lower bound 0.761 < 0.8) | 0.840 / 0.888 | unsupported |
| industrial | industrial | 126 | 0.938 | 0.799 | 0.238 | 0.0000 | 1.000 | 0.912 | **unsupported** (Wilson lower bound 0.799 < 0.8; recall 0.238 < 0.3) | – / – | unsupported |
| maple | maple | 117 | 0.879 | 0.771 | 0.436 | 0.0000 | 1.000 | 0.904 | **unsupported** (test precision 0.879 < 0.9; Wilson lower bound 0.771 < 0.8) | 0.931 / 0.231 | unsupported |
| moon | moon | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| night | night | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| oranges | oranges | 164 | 0.872 | 0.733 | 0.207 | 0.0008 | 0.624 | 0.804 | **unsupported** (test precision 0.872 < 0.9; Wilson lower bound 0.733 < 0.8; recall 0.207 < 0.3) | 0.907 / 0.299 | unsupported |
| pink | pink | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| reflection | lesson_reflection | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| school_uniform | sailor | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | 0.903 / 0.406 | unsupported |
| scooter | scooter | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | 0.887 / 0.433 | unsupported |
| street | street | 199 | 0.938 | 0.717 | 0.075 | 0.0000 | 1.000 | 0.935 | **unsupported** (Wilson lower bound 0.717 < 0.8; recall 0.075 < 0.3) | – / – | unsupported |
| suburb | suburb | 137 | 0.848 | 0.743 | 0.409 | 0.0008 | 0.620 | 0.923 | **unsupported** (test precision 0.848 < 0.9; Wilson lower bound 0.743 < 0.8) | 0.900 / 0.131 | unsupported |
| umbrella | umbrella | 143 | 1.000 | 0.824 | 0.126 | 0.0000 | 1.000 | 0.978 | **unsupported** (recall 0.126 < 0.3) | – / – | unsupported |
| water_lily | water_lily | – | – | – | – | – | – | – | **unsupported** (no validation threshold reaches precision 0.95 with >= 20 fires) | – / – | unsupported |
| wave | wave | 116 | 0.846 | 0.703 | 0.284 | 0.0000 | 1.000 | 0.903 | **unsupported** (test precision 0.846 < 0.9; Wilson lower bound 0.703 < 0.8; recall 0.284 < 0.3) | – / – | unsupported |
| willow | willow | 47 | 0.896 | 0.778 | 0.915 | 0.0000 | – | 0.661 | **unsupported** (test precision 0.896 < 0.9; Wilson lower bound 0.778 < 0.8) | 0.920 / 0.489 | unsupported |
| windmill | windmill | 128 | 0.875 | 0.640 | 0.109 | 0.0000 | 1.000 | 0.975 | **unsupported** (test precision 0.875 < 0.9; Wilson lower bound 0.640 < 0.8; recall 0.109 < 0.3) | 0.864 / 0.148 | unsupported |
| window_light | window_light | – | – | – | – | – | – | – | **unsupported** (no source label for light falling through a window (Window + Sunlight is not the same thing)) | – | – |
| grid | grid | – | – | – | – | – | – | – | **unsupported** (pattern/composition concept; no source label that means a grid composition) | – | – |
| round_glasses | round_glasses | – | – | – | – | – | – | – | **unsupported** (source labels have Glasses but not the frame shape) | – | – |
| qipao | qipao | – | – | – | – | – | – | – | **unsupported** (no Open Images class for qipao/cheongsam; no licensed labelled source found in budget) | – | – |
| tatami | tatami | – | – | – | – | – | – | – | **unsupported** (no Open Images class; no licensed labelled source found in budget) | – | – |
| rain_alley | rain_alley | – | – | – | – | – | – | – | **unsupported** (compound scene (oil-paper umbrella / rainy alley); no source label) | – | – |
| peach_blossom | peach_blossom | – | – | – | – | – | – | – | **unsupported** (source labels cannot separate peach from cherry/plum blossom) | – | – |
| two_trees | two_trees | – | – | – | – | – | – | – | **unsupported** (a count of trees, not a class; no source label) | – | – |
| frame | lesson_frame | – | – | – | – | – | – | – | **unsupported** (frame-within-frame is a composition judgement; no source label) | – | – |

Badge-level check for `birds` (bird OR small_animal head): test precision 0.975, recall 0.619, 754 positives.
Across all 22 heads, 0.31 % of the generic test photos received a signal on a head whose label is negative, 2.7 % a
signal on a head where the label is unknown (an `IGNORE` parent such as Plant), 13.6 % a verified-correct signal.

Pixel rules (one HSV statistic, one threshold, same rule; `research/concepts/pixel_rules.py`): night (fraction of
dark pixels), golden hour (warm saturated fraction), blue hour (deep blue dim fraction), pink (pink hue fraction).
**None** reached validation precision 0.95 (test AP 0.44, 0.08, 0.10, 0.43). Time of day and colour words stay
unsupported; the golden-hour head (an image classifier, not a rule) ships.

### Reading the decisions

- Strong (Wilson LB >= 0.90): bird, small_animal, flowers, sunflower, golden_hour, boat, pumpkin, construction, tunnel.
- Shipped but near the bar (LB 0.81–0.89): starry 0.813, power_plant 0.843, wheat 0.846, beetle 0.848, ice 0.851,
  dock 0.853, rickshaw 0.857 (66 positives), motorcycle 0.860, cabin 0.864, fox 0.874, desert 0.891, lotus_pond 0.891,
  spider_web 0.899. **ice** shows one generic false positive, which at ice's low generic prevalence implies precision of
  about 0.35 in random photos; **boat** about 0.90. Running these in shadow mode first is advisable.
- Fail on precision: gas_station (0.854: canopies, car parks), suburb, maple, oranges, fishing, wave, windmill, willow
  (0.896 with 47 positives).
- Fail on recall at the precision bar: umbrella and corridor (precision 1.000 but recall 0.13 / 0.14), duck 0.276,
  dragonfly 0.182, diner, car_window, industrial, street (0.075).
- No validation threshold at precision 0.95: night, blue_hour, fog, moon, pink, reflection, scooter, water_lily,
  school_uniform. Moon is a small object in a sky: the verified moon negatives are dusk and night skies without one,
  and a whole-image embedding cannot separate them. Night, blue hour, fog and pink are atmospheres or colours whose
  Open Images labels are loose. Reflection was expected to fail.
- No adequate legal source label (never trained): window_light, grid, round_glasses, qipao, tatami, rain_alley,
  peach_blossom, two_trees, lesson_frame (reasons in `concept_map.py`).

## SigLIP-base zero-shot comparison

`google/siglip-base-patch16-224`, one prompt per head, text embeddings precomputed, sigmoid(scale · cos + bias).
Same deduplicated val/test rows and threshold rule (`research/concepts/siglip_compare.py`).

| measure | linear heads on DINOv2 (shipped path) | SigLIP-base zero-shot |
|---|---|---|
| heads passing the ship rule | 22 | 5 (pumpkin, flowers, tunnel, fox, spider_web) |
| mean test AP over the 38 heads with both | 0.849 | 0.758 (heads better on 31 of 38) |
| extra model RSS (ONNX Runtime, 1 thread) | 0 MB (reuses the photo-gate vector) | fp32 609 MB; int8 (dynamic) 197 MB |
| extra latency per photo, workstation, 1 thread | 0.006 ms (heads only) | int8 87–380 ms, fp32 240–750 ms (machine under shared load; ranges are two runs) |
| model file | 68 KB of weights | int8 89 MB, fp32 354 MB |

Int8 SigLIP also changes the image vector (cosine to fp32 mean 0.969, min 0.882 on 100 images), so its thresholds would
have to be re-chosen on the int8 model. With the app at about 170 MB plus the 90 MB encoder, int8 SigLIP would fit the
640 MB container, but it is worse on this data and adds a second model. Not recommended.

## Latency and memory of the shipped path

- Heads: 22 x 768 float32 (68 KB in memory), artefact JSON parse 2.7 ms, prediction p50 0.006 ms / p99 0.006 ms
  (NumPy, 1 thread). The encoder pass is the photo gate's, already paid (workstation p50 97–112 ms, 1 thread; the
  Oracle VM measured p50 385 ms for it on 2026-11-09).
- Thumbnail vs production input: for 77 test images re-fetched at about 1024 px and passed through the website's
  preparation (long edge <= 2048, JPEG q90), the embedding cosine to the 320 px thumbnail is mean 0.993 (min 0.952), and
  the 22 heads' fire/no-fire decisions agree on 1,693 of 1,694 (99.94 %).

## Limits

- Open Images is Flickr 2005–2016; FilmAnalysis uploads (film scans, phone photos, Chinese street scenes) differ.
  Nothing here was measured on the site's own photos.
- Labels are human-verified but not exhaustive; generic photos may hide unlabelled positives (precision is
  conservative there), and verified labels themselves contain some rater error.
- Class mappings are approximations of the words: `small_animal` is Cat/Dog/Squirrel/Wildlife; `cabin` is Log
  cabin/Hut/Shack; `dock` is Pier/Dock; `starry` is Star/Milky way; `rickshaw` includes auto rickshaws (the word matches
  them too); `lotus_pond` is Sacred lotus / Lotus family (the plant, not necessarily a pond); `wheat` is
  Wheat/Barley/Rye (plants or fields). `golden_hour` is Sunset/Sunrise/Afterglow.
- Several shipped heads sit just above the Wilson bound; test sets of 66–176 positives cannot pin precision closer.
- No concept for which this study found no legal labelled source was attempted with other data (Wikimedia Commons'
  API is disallowed for generic agents by robots.txt, see DATASET_SOURCES.md; GBIF was not used to avoid interfering
  with the running species download).

## Reproduce

```bash
python -m research.concepts.oi_meta labels            # stream-filter the 2.7 GB train label CSV
python -m research.concepts.select_images ids
python -m research.concepts.oi_meta images            # stream-filter the train image list
python -m research.concepts.select_images build
python -m research.concepts.download --workers 32     # CVDF copy
python -m research.concepts.download --flickr-fallback --workers 8
python -m research.concepts.features [--parity]
python -m research.concepts.train                     # reads test; --no-test for development
python -m research.concepts.pixel_rules
python -m research.concepts.siglip_compare embed|eval|footprint
python -m research.concepts.export
python -m research.concepts.resolution_check
python -m research.concepts.tables
pytest -q tests/test_concept_heads.py
```
