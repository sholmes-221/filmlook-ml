# filmlook-ml

Dataset preparation, labeling, training, evaluation and export for FilmLook's specialized photography-aesthetics
models: film-look scoring, visual attributes, and (experimental) film-aesthetic and photographer-style matching. A
standalone CPU runtime (`filmlook_infer`) can be used without the training environment. No LLM runs at inference.

**Development status (2026-11-07).**
- **Shipped v1 film-score model:** measured statistics + small MLP, ONNX, 0.6 MB.
- **Main DINOv2-Small model (experimental, not promoted):**
  - frozen `facebook/dinov2-small` baseline with retrieval and lightweight heads (`scripts/dinov2_baseline.py`);
  - a MobileNetV3-Small student for CPU-constrained hosts (`scripts/train_student.py`);
  - a film-inspired synthetic augmentation study (`scripts/synth_*.py`);
  - a database-driven taxonomy (`taxonomy/`, 139 photographers, 333 film stocks, 67 aesthetic directions).
- **Not yet available:** photographer and film-stock matching are not validated; there are no human evaluation labels;
  most attribute labels are measured-statistic proxies.

Details: `REPORT.md`, `docs/AESTHETICS_LEARNING_RESEARCH.md`, `deployment/E2_MICRO_PLAN.md`.

## Install

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU training; CPU wheels also work
uv pip install -e ".[train,label,dev]"
pytest -q
```

Images, labels, embeddings, checkpoints and reports are not in the repository: `scripts/download_*.py` and
`scripts/prepare_dataset.py` rebuild the licensed dataset (see `DATASET_SOURCES.md`).

## Film-aesthetics model v1 (shipped runtime)

For one photograph, v1 returns a **film score** (0-100), six **rendering attributes**, measured
**style tags**, a heuristic **confidence**, and bilingual (English / Simplified Chinese) **strengths,
weaknesses and summary** from deterministic templates. No LLM, GPU or network is used at inference.

> v1 removed photographer-name matching and film-stock similarity: no licensed reference data or
> reliable evaluation exists for them. See `REPORT.md` for measured results and limitations, and
> `DATASET_SOURCES.md` for licensing. **No human labels exist yet; human alignment is unverified.**

## Photographer-style similarity (Mini Tool, pure JavaScript)

`js/filmlook-style/dist/filmlook-style.js`: a 37 KB ES2017 bundle with no dependencies. It gives the top 3 style
resemblances with evidence, or "no reliable match"; it is not authorship. Python/OpenCV reference in
`src/filmlook_ml/style/`, with exact parity tests. See `js/filmlook-style/README.md`. **Unvalidated against human
judgement.**

```bash
python scripts/style_extract.py && python scripts/style_export_js.py && python scripts/style_make_fixtures.py
node --test js/filmlook-style/test/style.test.js
python scripts/style_stability.py          # label-free robustness
python scripts/style_labels.py --images 60 # Claude style judgements (dry run unless --no-dry-run --confirm-paid)
python scripts/style_evaluate.py           # profile vs linear/forest/MLP vs DINOv2 against judgements
```

## Exploratory categorization (Claude API, offline, gated)

`scripts/annotate_categories.py` adds one photographer-inspired style, one film aesthetic and enumerated visual
attributes per original photo, stored in `data/annotations/claude_exploratory/` (never training targets).
Use `--status` for discovery, progress and distributions; it is a dry run unless approved (`docs/AI_DATA_POLICY.md` §4a).
`scripts/export_annotations.py` is the only path to `training_approved/` and needs a written authorization reference.

## Main DINOv2-small model (work in progress)

- Taxonomy from the FilmLook app's own database, read-only: `python scripts/extract_taxonomy.py --source ~/misc/film-evaluation`
  writes `taxonomy/*.v1.json` (139 photographers, 333 film stocks, 67 aesthetic directions); load it with
  `filmlook_ml.taxonomy_db.load_taxonomy()`.
- Baseline: `python scripts/dinov2_baseline.py [--latency]` (frozen DINOv2-small, Stage A retrieval plus Stage B heads,
  no Claude labels) writes `reports/dinov2_baseline.json` and `checkpoints/dinov2_baseline_v0/`.
- Reports: `reports/evaluation_database_inventory.md`, `reports/taxonomy_comparison.md`,
  `reports/dinov2_training_strategy.md`.
- Knowledge bases with sources: `knowledge/film_kb.v1.yaml` (manufacturer-measured grain metrics kept apart from
  claims and editorial text), `knowledge/photographer_kb.v1.yaml`.
- Generators and datasets: `docs/HF_FILM_MODEL_RESEARCH.md`, `docs/FILM_DATASET_RESEARCH.md`,
  `data_registry/film_datasets.json`; `scripts/gen_sdxl_film.py` (SDXL img2img), `scripts/ingest_blueneg.py`.
- Experiments: `scripts/synth_experiment_v2.py` (cross-generator + real film), `scripts/train_style_student.py`;
  leaderboard and decisions in `docs/EXPERIMENT_LOG.md`; content-invariance benchmark `docs/BENCHMARK.md`
  (`scripts/style_benchmark.py`, `scripts/abstention_analysis.py`); validated findings `docs/FINDINGS.md`; research
  model card `docs/MODEL_CARD.md`; leakage benchmark `scripts/synth_experiment_v3.py`
  (film-synth-v2 presets with label-independent grain), `scripts/spatial_probe.py`, `scripts/eval_film_authenticity.py`.
- Research interfaces: `filmlook_infer.shadow.ShadowAnalyzer` (`docs/SHADOW_MODE_INTEGRATION.md`),
  `docs/BADGE_ATTRIBUTE_MAPPING.md`.

## Two Python packages

| package | purpose | dependencies |
|---|---|---|
| `src/filmlook_infer/` | runtime: image loading, measured statistics, ONNX inference, templates, JSON schema | numpy, pillow, scipy, onnxruntime |
| `src/filmlook_ml/` | training: acquisition, licensing gate, labeling (heuristic / Claude teacher), training, evaluation, export | + torch, transformers, anthropic, ... |

Training imports the runtime's feature/loader code, so train-time and serve-time preprocessing are the same code.

## Model variants (see `REPORT.md` for the comparison)

```
A  texture : image (<=1280 px) -> 28 measured statistics -> MLP            -> score, 6 attributes, film-capture
B  dino    : image -> DINOv2-small (frozen, 22M) -> CLS++mean-patch -> MLP -> same outputs
C  fused   : A ++ B
```
Each shipped model is a 3-seed ensemble exported as one ONNX graph; ensemble spread drives `confidence`.

## Pipeline

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[train,label,dev]"

python scripts/discover_commoncatalog.py --shards 80      # metadata only: license, EXIF medium, subject
python scripts/download_commoncatalog.py --plan           # show the stratified selection; then run without --plan
python scripts/download_datasets.py --source pd12m --limit 150
python scripts/download_datasets.py --source synthetic --limit 40
python scripts/prepare_dataset.py                          # license gate, validate, dedupe, edits, uploader-grouped splits
python scripts/preflight_check.py                          # compliance preflight: provenance + permissions (must PASS)
python scripts/generate_pseudo_labels.py --teacher heuristic
python scripts/generate_pseudo_labels.py --teacher claude --images 40 --pairs 10   # dry run: cost estimate
python scripts/compare_models.py --seeds 0 1 2 3 4         # A/B/C x resolutions x seeds -> reports/model_comparison.json
python scripts/train.py --kind texture --seeds 0 1 2 --promote
python scripts/evaluate.py                                 # reports/eval_report.html
python scripts/export_model.py                             # ONNX + parity
python scripts/benchmark_cpu.py --models checkpoints/best  # end-to-end CPU latency / RSS
python scripts/build_package.py --verify                   # dist/filmlook-infer-<ver>/ (+ fresh-venv check)
pytest -q
```

### Compliance (mandatory)

`CLAUDE.md` and `docs/AI_DATA_POLICY.md` govern every labeling, training and evaluation run. `python
scripts/preflight_check.py` must PASS first; the same checks run inside `load_train_data` and the Claude labeling
scripts. Training on Claude outputs is currently **blocked** (no written authorization reference); human,
heuristic, synthetic and capture-medium supervision is permitted.

### Claude teacher (offline only, budget-capped)

```bash
echo 'ANTHROPIC_API_KEY=...' >> .env          # git-ignored; never printed
python scripts/preflight_check.py --mode labeling --intended-use evaluation --images 40 --pairs 10
python scripts/generate_pseudo_labels.py --teacher claude --intended-use evaluation --images 40 --pairs 10   # estimate
python scripts/generate_pseudo_labels.py --teacher claude --intended-use evaluation --images 40 --pairs 10 --no-dry-run --confirm-paid
```
`--intended-use` is required; `training` needs a written authorization reference, `evaluation` needs the owner's
recorded approval in `configs/labeling.yaml` (`claude.evaluation_use`).
The US$5 cap in `configs/labeling.yaml` is cumulative across runs (`data/labels/claude_usage.jsonl`); re-runs
resume from the cache. Training authorization provenance is recorded in `configs/labeling.yaml` and
`DATASET_SOURCES.md`. Flagged labels go to `data/labels/claude_review_queue.jsonl`.

### Human review

`python scripts/review_labels.py --reviewer <name>` (Streamlit, 127.0.0.1). Labels on **test-split** images form the
isolated human benchmark; training uses only the train split and model selection only the val split.

## Inference

```python
from filmlook_infer import FilmLookAnalyzer
an = FilmLookAnalyzer("checkpoints/best")          # or the packaged dist/.../model
print(an.analyze("photo.jpg", language="zh"))
```
CLI: `python -m filmlook_infer.cli photo.jpg --lang en --model checkpoints/best` (JSON; exit 2 on invalid input).
Schema: `src/filmlook_infer/schema/filmlook_output_v1.json`. Integration notes: `deployment/INTEGRATION.md`.

Images, labels, embeddings, checkpoints, reports and `dist/` are git-ignored.
