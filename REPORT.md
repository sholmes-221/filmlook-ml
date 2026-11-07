# FilmLook ML: v1 build report (2026-11-06)

Two deliverables, both trained and evaluated in this repository; the FilmLook application was not touched.

1. **Film-authenticity model v1** (server-side, Python + ONNX): `dist/filmlook-infer-1.0.0/`.
2. **Photographer-style similarity v1** (on-device, pure JavaScript for the Xiaohongshu Mini Tool): `js/filmlook-style/dist/filmlook-style.js`.

**Read first:**
- **No human labels exist**, so agreement with human judgement is **unverified** for both models.
- **No Claude labels exist.** `ANTHROPIC_API_KEY` was not available, and the project owner chose to proceed without Claude. $0 was spent.
- Every number below is either **teacher agreement** (heuristic or synthetic labels), **metadata agreement** (EXIF capture medium), or a **label-free** measurement (stability, latency, size). None of them is human accuracy.

## 0. Environment

RTX 4090 24 GB, shared: another job used about 5.7 GB and 65–82% utilisation; all my GPU work stayed under 2 GB. Python 3.12 `.venv`, PyTorch 2.11 + CUDA 12.8, onnxruntime 1.31, scikit-learn, OpenCV 5.0 (headless), and Node 24 (from the `nodejs-wheel-binaries` pip package) for JS tests. Tests: **69 pytest tests passed**, which includes the **9-test Node suite**.

## 1. Dataset composition and licensing

| source | license (checked per record) | originals | notes |
|---|---|---|---|
| CommonCatalog CC-BY (YFCC100M Flickr) | CC-BY 2.0 | 819 | 320 film scans (Frontier/Noritsu/Coolscan EXIF), 260 digital cameras, 120 phones, 119 phones/cameras with filter tags (VSCO, Instagram…); street 141, portrait 164, night 82, architecture 140, landscape 145 |
| PD12M | CC0 / PDM | 149 | archival, mostly monochrome |
| synthetic | n/a | 40 | procedural |

- **Images:** 1,008 originals × 9 variants = **9,072 images** (8 controlled edits per photo).
- **Splits:** grouped **by uploader**, so the same photographer, lab or scanner never spans train/val/test: 730 / 145 / 133 families. The test set has only **30 film scans vs 75 digital originals**, so capture-medium confidence intervals are wide.
- **How the data was fetched:** metadata first (80 shards scanned for license, EXIF and caption), then only the selected images, via Parquet page-level range reads through `huggingface_hub`. About 800 MB was fetched. No Flickr scraping, no whole-shard downloads.
- **Rejected:**
  - Wikimedia Commons API and Openverse API: robots.txt disallows them (Openverse also blocks AI crawlers).
  - Unsplash Lite: internal-use-only terms, and v1 is a public product.
  - CC-BY-SA, AVA, Unsplash Full, and Hugging Face re-uploads.
  - Details in `DATASET_SOURCES.md` and `configs/datasets.yaml`.

**Claude authorization:**
- **What's recorded:** what you told me, in `configs/labeling.yaml` → `training_authorization`, with `reference: null`. No reference number was invented.
- **How it's enforced:** the gate checks this record (or the `FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF` env var), and every Claude label carries the provenance string.
- **Not covered:** photo rights are still enforced separately, and only consented datasets are ever uploaded.

## 2. Claude labeling: cost and quality

**Not run** (no API key). The pipeline is implemented and tested with a fake client:
- **Inputs:** rubric v2 prompt with a 1024 px view plus a native 384 px crop.
- **Output schema:** strict structured output (6 attributes, style vector, mood tags, uncertainty, strengths/weaknesses), plus pairwise judgements.
- **Calls:** cache keyed by content hash and prompt version (resume without repeat charges), typed retries with backoff, and a per-call usage log (model, prompt version, tokens, $).
- **Budget:** the **US$5 cap is cumulative across runs**.
- **Quality checks:** automatic flags (score vs attribute mismatch, monochrome mismatch, grain and temperature contradictions) send labels to a review queue, and the excluding flags keep them out of training.

| dry-run estimate (Opus 5.5, $4/$20 per MTok) | cost |
|---|---|
| per image / per pair judgement | $0.024 / $0.022 |
| pilot: 40 images + 10 pairs | $1.16 |
| $5 buys | ~180 images + ~35 pairs |
| all 1,008 originals | ~$24 (needs your approval) |
| 60 style judgements (`scripts/style_labels.py`) | $1.42 |

## 3. Film model: training configuration and results

- **Labels:** heuristic teacher v2 (transparent formulas on the new 6-attribute rubric) for all 9,072 images, synthetic edit orderings (8,064 partial labels, 19,152 within-photo pairs), and the capture medium from EXIF (819 originals).
- **Losses:**
  - SmoothL1 on the score and attributes; the attribute loss is averaged per attribute so the overall score can't dominate;
  - margin ranking on the pairs;
  - BCE on the capture-medium head;
  - **film-scan > digital cross-image ranking pairs** on the score (weight 2, 8 per scan, train split only).
- **Ensemble:** 3 seeds, exported as one graph. Seed spread drives `confidence`.

**Test split (133 unseen uploader groups, 1,197 images):**

| metric | value |
|---|---|
| within-photo edit orderings (2,527 pairs) | **0.994** (film_score 0.998, highlight 0.991, colour 1.0, microcontrast 1.0, grain 1.0, tonal depth 0.982) |
| film-score agreement with the heuristic teacher (originals) | MAE 0.038 (0–1 scale), Spearman 0.82 |
| film score: real film scans vs digital originals (AUROC, uploader-bootstrap 95% CI) | **0.65 [0.54, 0.77]** (val 0.60 [0.43, 0.74]) |
| capture-medium head: film vs digital | 0.86 [0.79, 0.92]; vs *filtered* digital 0.88; ECE 0.075 (val 0.81 [0.67, 0.92]) |
| mean score by edit | B&W emulation 81, film emulation 78, originals 71, posterised 70, fake grain 69, oversharpened 68, clipped digital 62, HDR 60, harsh B&W 60 |
| mean score by subject (originals) | landscape 73, portrait 71, street 70, architecture 69, **night 61** (low-light photos score lower) |
| seed stability | mean spread 1.5 points (p95 2.9) |
| confidence vs absolute error (Spearman) | **0.08: weakly informative.** Treat `confidence` as a soft hint only |

**Two findings that matter more than the headline numbers:**
1. **The heuristic teacher does not track real film capture.** Its own score separates film scans from digital originals at AUROC 0.52–0.55, essentially chance. A student trained only on it inherits that: 0.52–0.58 across all architectures. The ranking pairs from real metadata move the shipped score to 0.60 (val) / 0.65 (test).
2. **"Looks like a real film scan" and the rubric partly conflict.** The capture head recognises real scans well but ranks degraded edits of a scan above cleaner versions (pair accuracy 0.23), and blending it into the score destroys the edit orderings. A likely cause: minilab scanners (Fuji Frontier, Noritsu) apply heavy digital sharpening, which the rubric penalises. Our synthetic film emulation also differs from real lab scans. So the capture probability is **kept internal**: it's in the ONNX outputs but not in the JSON. **Human labels are needed to decide which notion of "film-like" users mean.**

## 4. Comparison: handcrafted vs DINOv2 vs distillation

`scripts/compare_models.py`: 5 seeds × each configuration, selection on val, reporting on test. Mean ± std over seeds:

| model | edit orderings (test) | film vs digital, capture head: **val** / test | vs filtered digital: val / test | score MAE vs heuristic |
|---|---|---|---|---|
| heuristic teacher (no training) | 0.887 | 0.553 / 0.519 | 0.599 / 0.506 | – |
| **A texture @1280 px** | 0.994 ± .001 | **0.810** / 0.844 | **0.774** / 0.870 | 0.028 |
| A texture @1024 | 0.991 | 0.767 / 0.796 | 0.730 / 0.840 | 0.032 |
| A texture @768 | 0.990 | 0.733 / 0.763 | 0.672 / 0.798 | 0.033 |
| B DINOv2 @224 | 0.946 ± .004 | 0.717 / 0.816 | 0.676 / 0.870 | 0.091 |
| B DINOv2 @168 / @112 | 0.942 / 0.922 | 0.726 / 0.670 (val) | – | 0.09 |
| C fused @224 | 0.995 ± .001 | 0.794 / 0.879 | 0.754 / 0.924 | 0.047 |
| C fused @168 | 0.997 | 0.790 / 0.887 | 0.745 / 0.902 | 0.049 |
| C fused @112 | 0.995 | 0.760 / 0.838 | 0.679 / 0.867 | 0.047 |

**Decision: Model A (texture-only, 1280 px).**
- **DINOv2 alone (B)** is clearly worse at the controlled orderings.
- **Fused (C)** wins on test (+0.035) but loses on val (−0.016), the split used for selection. With 30 test film scans, the CIs (about ±0.07) overlap completely.
- **Cost:** C costs 43× the artifact size (26.9 vs 0.62 MB INT8), +45 MB RSS and +70 ms.
- **Resolution:** lower texture resolution hurts A on every metric, so 1280 stays.
- **Model D (distilled student): not built.** It is only warranted if B/C beat A, which they did not. A is already the smallest practical model (a 0.6 MB MLP); its cost is NumPy feature extraction, not the network.
- **Stage B fine-tuning: not run** for the same reason (it remains available: `scripts/train.py --kind fused --stage-b`).

## 5. Human evaluation

**None available.** The review app (`scripts/review_labels.py`) now covers the v1 rubric (single image + pairwise) and a new **Style** mode. Labels on test-split images form the isolated human benchmark, since training reads only the train split and selection only the val split. `scripts/style_evaluate.py` and `scripts/evaluate.py` report human agreement automatically once labels exist; until then they print "unverified" / "BLOCKED".

## 6. ONNX size, memory, latency (film model)

| | Model A (shipped) | Model C INT8 |
|---|---|---|
| artifact | **0.62 MB** | 26.9 MB |
| peak process RSS, end to end on 4–6 MP camera JPEGs | **160 MB** | 205 MB |
| cold start / warm p50 (1 thread, workstation) | **0.48 s / 0.30 s** | 0.60 s / 0.37 s |
| Docker 256 MB, 0.25 CPU (simulation) | cold 5.0 s, warm 1.3 s (p95 2.5 s), cgroup peak 109 MB | – |
| Docker 256 MB, 0.125 CPU (simulation) | cold 5.4 s, warm 2.5 s | – |

Details and reproduction: `deployment/README.md`. Not measured on real deployment hardware.

## 7. Selected production model

**`filmlook-ml-v1.0.0` = Model A**, a 3-seed texture MLP ensemble. It is the smallest model and equal or better on val, the selection split. Package: `dist/filmlook-infer-1.0.0/` (1.1 MB), verified in a fresh venv with only numpy, pillow, scipy, and onnxruntime installed.

## 8. Inference command and sample output

```bash
pip install dist/filmlook-infer-1.0.0        # or copy filmlook_infer/ + model/ next to your service
python -m filmlook_infer.cli photo.jpg --lang zh --model dist/filmlook-infer-1.0.0/model
```
```json
{"schema_version": "1.0", "model_version": "filmlook-ml-v1.0.0", "language": "en", "film_score": 86,
 "attributes": {"highlight_rolloff": 0.94, "color_response": 0.94, "microcontrast": 0.84, "grain_naturalness": 0.83,
                "tonal_depth": 0.89, "film_coherence": 0.88},
 "attribute_names": {"...": "..."},
 "style_tags": ["warm", "low_contrast", "fine_grain"], "style_tag_names": ["warm", "low contrast", "fine grain"],
 "monochrome": false, "confidence": 0.91, "confidence_level": "high",
 "strengths": ["smooth highlight roll-off and tonal transitions", "coherent, natural color palette",
               "rich shadow gradation and tonal depth"],
 "weaknesses": [], "summary": "Strongly film-like rendering.",
 "disclaimer": "Automated visual assessment of rendering characteristics. It does not identify the camera, film stock or photographer, and it is not a judgement of artistic merit."}
```
This is the actual output for the packaged sample (`examples/sample.jpg`, a synthetic scene run through our film emulation; abridged only at `attribute_names`). Full EN/ZH samples: `dist/filmlook-infer-1.0.0/examples/`. Schema: `filmlook_infer/schema/filmlook_output_v1.json`.

## 9. Photographer-style similarity (JS, Mini Tool)

- **Platform:** the Mini Tool is a sandboxed HTML/JS page.
  - Canvas 2D is available.
  - eval, the Function constructor, and WebAssembly are blocked.
  - Code must be ES2017 (Chrome 61 baseline).
  - At most one classic Worker.
  - No network.
  - No numeric package size limit is stated.
- **Design:**
  - **Features:** 48 JS-reproducible features (tone, colour, hue distribution, texture/sharpness, local contrast, 8×8 spatial composition). They're computed identically in Python/OpenCV and in ES2017 JS: **max difference < 1e-9**, engine JSON identical, `box2x` bit-identical.
  - **Dimensions:** 13 interpretable dimensions, normalised to the median/IQR of real photos.
  - **Profiles:** 9 photographer profiles as measurable targets (Leiter, Eggleston, Moriyama, Webb, Fan Ho, Shore, Herzog, Kawauchi, Haas), with a monochrome gate.
  - **Scoring:** distinctiveness (a typical photo scores ~0, overshoot is penalised), then **rarity calibration** against 601 ordinary train-split photos, then a "reliable" rule. "No reliable match" is a first-class result.
- **Bundle:** one classic script, **37 KB (15 KB gzip)**. It parses as ES2017 with acorn, and the tests ban eval, WASM, fetch, and post-2017 syntax.
- **Speed:** 8.7 ms p50 / 28 ms p95 per photo on one desktop core; **phones not measured**.
- **Stability:** a WebView-dependent downscale initially changed the top match 21% of the time, so I redesigned preprocessing as a browser draw at 2× followed by our own exact 2×2 average. Top-1 now stays the same under WebView-like bilinear halving 97.7%, JPEG q75 92%, 4% crop 96%, +8% exposure 84%, flip 100% (218 held-out photos).
- **Match rate:** the engine commits to a match on about 26% of ordinary photos.
- **Known failure:** processing artefacts can read as style. For example, an over-processed HDR edit matched "Stephen Shore" (flat contrast plus crisp detail), and about 14% of HDR edits match some profile.
- **Evaluation against human judgement: BLOCKED.**
  - **No labels from any source:** no human judgements, no Claude judgements, and Flickr tags of photographer names turned out to be essentially absent (1 hit in about 800k photos; earlier "Leiter" hits were the German word).
  - **What's ready:** `scripts/style_evaluate.py` runs the full comparison as soon as judgements exist: profile engine vs linear / random forest / MLP on the JS features vs a DINOv2 logistic baseline; top-1/top-3/MRR, uploader-grouped CV, train on Claude and evaluate on human. It is tested on synthetic judgements only.
  - **Ready to ship once one wins:** JS inference for all three learned types is implemented and parity-tested.
  - **Until then**, v1 ships the transparent profile engine and labels it unvalidated. No accuracy numbers are claimed.
- **Licensing:**
  - No photographer images were used or shipped.
  - Profiles are textual.
  - The calibration pool is CC-BY ordinary photos; only quantiles ship.
  - Fixtures are CC0/synthetic and git-ignored, regenerated by script.

## 10. Remaining limitations

1. **No human labels.** Neither "film-like" nor "style resemblance" has been checked against people. The rubric/capture conflict (§3) can only be settled by human judgement.
2. **Teacher quality.** The heuristic teacher is hand-tuned and does not track real film capture. Claude labels would be a stronger teacher but were not available.
3. **Small capture-medium test set** (30 film scans), so confidence intervals are wide.
4. **Domain.** Flickr 2004–2014 photos. Modern phone computational photography (HDR+, Night mode) is under-represented: 120 phones, mostly older models. Night photos score about 10 points lower.
5. **Confidence** is only weakly related to error (Spearman 0.08).
6. **Style profiles** are expert-description hypotheses, are generic for some photographers (Haas and Herzog match most often), and can be triggered by processing artefacts.
7. **Phone runtime** for the JS module and **real server hardware** for the film model were not measured.

## 11. Next steps for integrating with the separate FilmLook application

1. **Server film model:** copy `dist/filmlook-infer-1.0.0/` into the backend, load `FilmLookAnalyzer` once per worker, and validate `schema_version`. See `deployment/INTEGRATION.md`. Run `deployment/container_bench.py` on the real instance.
2. **Mini Tool style module:** copy `js/filmlook-style/dist/filmlook-style.js`, call `FilmLookStyle.analyzeImage(img, lang)`, and show photographer names only when `status == "match"`. Measure latency on a low-end Android device; move to the Worker if it exceeds about 300 ms. See `js/filmlook-style/README.md`.
3. **Highest-value next experiment:** about 300 human judgements in the review app's Pairwise + Style modes on the test split, plus a $5 Claude pilot (40 images + 10 pairs for film, then style judgements) once the key is in `.env`. Then re-run `scripts/compare_models.py`, `scripts/evaluate.py`, and `scripts/style_evaluate.py`. That decides:
   - whether the film score should follow the rubric or real-scan appearance;
   - whether a learned style model beats the profiles;
   - whether DINOv2 adds anything that justifies its cost.

## Exploratory categorization run (2026-11-07)

Workflow: `scripts/annotate_categories.py`; policy: `docs/AI_DATA_POLICY.md` §4a.

| item | value |
|---|---|
| images in manifest | 9,072 (1,008 originals) |
| eligible for categorization | 968 originals (CommonCatalog CC-BY 819, PD12M CC0/PDM 149) |
| skipped | 8,064 derived edits; 40 synthetic test scenes (not photographs) |
| categorized | **0** |
| failed | 0 |
| photographer-style / film-aesthetic distribution | none (no annotations exist) |
| annotation provenance | per-file record: image dataset, license, attribution, model, prompt and taxonomy version, permission status, tokens/US$ |
| training permission | **uncertain**: owner statement recorded, no written reference. Export refused; preflight check added |

**Blockers (Claude analysis not run):**
1. No `ANTHROPIC_API_KEY` in the environment or `.env`.
2. `claude.exploratory_use` is not approved in `configs/labeling.yaml`. A paid run needs an explicit owner decision.
3. Training use of the outputs needs a written authorization reference from Anthropic. The Usage Policy requires
   "prior authorization", and the Help Center lists "using Outputs as training targets" as prohibited even for
   specialized classifiers.
4. Using this Claude Code session to annotate the images was ruled out. The Consumer Terms (s.3(7)) bar automated
   access except via an API key, and the Usage Policy bars routing requests through consumer subscriptions.

Cost at current settings: about US$0.024 per image, or about US$23 for all 968. The cumulative US$5 cap covers
about 200 images.

## Categorization pilot: 20 images (2026-11-07)

Schema `categorization-2.0`, taxonomy `taxonomy-v2`. Annotated by Claude (claude-opus-5-5) in an interactive
Claude Code session at the owner's request, with `annotation_interface: claude_code_interactive`. No API key was
available, so there is no API run. Full results: `reports/categorization_pilot.json`; annotations:
`data/annotations/claude_exploratory/annotations/` (git-ignored).

| item | result |
|---|---|
| eligible originals (revalidated) | 968 = 819 CC-BY-2.0 + 54 CC0-1.0 + 95 PDM-1.0; skipped 8,064 edits and 40 synthetic scenes |
| pilot sample | 20 (6 film scans, 4 digital camera, 2 phone, 4 filtered, 4 PD12M archival); all images loaded |
| categorized / schema-valid | 20 / 20; 0 failed |
| photographer primary | none_of_the_above 17, Alex Webb 1, Stephen Shore 1, Rinko Kawauchi 1 (all three at confidence 0.35-0.45) |
| film primary | none_of_the_above 14, Superia X-TRA 400 2, Portra 400 1, Velvia 50 1, C200 1, unknown 1 |
| needs review | 9 (low confidence 7, monochrome/contrast disagreement with measurement 4, film unknown 1) |
| training_eligible | 11: all 11 are none_of_the_above for the photographer head; film head 10 none_of_the_above + 1 Velvia 50 |
| time | median about 10 s per image between records; 205 s for 19 intervals |
| cost | not observable inside an interactive session. API estimate about US$0.024 per image, about US$23 for 968 |
| DINOv2 heads | **not trained**: the readiness gate refused (no class besides none_of_the_above has 8 or more train examples) |

Caveats:
- The selection printout showed the annotator the capture stratum and the uploader film-stock tags before
  annotation. The 3/3 agreement with uploader stock tags is therefore **not** independent evidence. API runs never
  send metadata.
- Measurement disagreements on sepia or yellow-toned archival prints and on night scenes are QA false positives
  (the measured mono test is strict; y_std is low on mostly dark frames).

Data issue found: `f993c25b62208870_original` (CommonCatalog CC-BY) is a phone photo of a printed 1967 yearbook
page. The uploader's CC-BY may not cover the underlying photograph, so quarantine it pending review. The
four PD12M items are archival prints, which is consistent with their PD/CC0 records.

## Compact models, E2 Micro constraints, synthetic film study (2026-11-07)

- **Production host corrected:** `VM.Standard.E2.1.Micro`, 1/8 OCPU, 1 GB RAM. See `deployment/E2_MICRO_PLAN.md`
  (simulated limits; real-host measurement pending approval).
- **MobileNetV3-Small student** (`scripts/train_student.py`):
  - 1.5M parameters, 4.3 MB fp32 ONNX; about 100 MB process RSS in the 1/8-CPU simulation;
  - beats frozen DINOv2 + linear heads on all six attribute heads (e.g. temperature macro-F1 0.89 vs 0.55);
  - capture type is level (0.35 vs 0.40);
  - distillation from DINOv2 gave no gain;
  - dynamic INT8 broke it, so it ships as fp32.
- **Synthetic film-aesthetic study** (100 sources × 8; `docs/AESTHETICS_LEARNING_RESEARCH.md`):
  - frozen DINOv2 global features memorise sources;
  - intermediate-layer statistics plus explicit features reach 0.73 test macro-F1 on 9 families;
  - pairwise ranking trained on synthetic pairs gets 0.77–0.84 on real independent-edit pairs;
  - real film scans are mostly predicted `identity` (weak film transfer);
  - composition is not decodable from pooled features.

## Cross-generator study and real-film evaluation (2026-11-07, later)

- Second generator (SDXL img2img, permissive licences) and a real-film set with owner-verified stock labels (BlueNeg,
  491 negatives) added; FilmPortrait and other FLUX-based models rejected on licence.
- Leave-one-generator-out failed: probes trained on the programmatic generator score SDXL outputs below chance
  (AUROC 0.35) because the two pipelines move measured noise in opposite directions. The earlier synthetic signal is
  largely a generator signature.
- Real-film transfer still works along the grain axis: BlueNeg film-vs-digital AUROC 0.80-0.81 with no real labels;
  the EXIF-supervised student remains better (0.86). Same-stock retrieval on real scans reaches 0.24 vs 0.12 chance.
- Training on both generators improved synthetic scores and hurt real transfer (0.67): rejected.
- Details and decisions: `docs/EXPERIMENT_LOG.md`, `docs/AESTHETICS_LEARNING_RESEARCH.md` section 10.
- **Leakage fixed (later):** grain randomised independently of the label lifts cross-generator AUROC from 0.35 to
  0.58 and lowers generator identifiability; real film-vs-digital drops to 0.70 because that signal was label-tied
  texture. The EXIF authenticity student audits at 0.78 [0.69, 0.88] within source (ECE 0.20), not 0.86. A 2×2
  spatial grid makes framing partly decodable (R² ≈ 0.23). Shadow-mode interface and badge mapping documented.
