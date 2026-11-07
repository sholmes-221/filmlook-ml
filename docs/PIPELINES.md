# FilmLook-ML pipelines: data, research, evaluation

Three independently executable pipelines share one lightweight, local-first orchestrator built on the existing
research framework (`research/experiment_registry.py`, `experiment_runner.py`, `evaluator.py`, `planner.py`,
`result_analyzer.py`). A slow download, a network failure, an unavailable dataset or a failed experiment never blocks
unrelated work: every pipeline is a CLI that can be run on its own, and the orchestrator only decides *when* to run
each one.

```
DATA PIPELINE  (research/data_pipeline.py, research/downloader.py, scripts/ingest_commons_film.py)
  commons_film acquisition  (resumable; rate-limit aware)  ──┐
  existing datasets -> validated -> ready                     │  dataset_state.json
                                                             ▼
RESEARCH PIPELINE  (research/research_loop.py -> experiment_runner -> evaluator)   dataset:<id> ready
  backlog entry with a runnable spec -> metered run -> decision -> EXPERIMENT_LOG      │
                                                                                       ▼
EVALUATION PIPELINE  (research/eval_pipeline.py -> scripts/eval_commons_film.py)
  eligible student checkpoints + fixed candidates on the analysis half -> pre-registered rules -> history.jsonl
  (test half: once per dataset, --final, never for selection)
```

## Commands

```bash
# data
python -m research.data_pipeline status
python -m research.data_pipeline validate-all              # existing datasets from their manifests (CPU, seconds)
python -m research.data_pipeline acquire commons_film      # one resumable download run (exit 0 ready, 3 resumable, 2 permanent)
python scripts/ingest_commons_film.py --download --max-minutes 230 --stall-minutes 20 --spacing 2.0   # what acquire runs
# research
python -m research.research_loop --once --dry-run          # what would run next, with its cost estimate
python -m research.research_loop --once [--id E26_layer_statistic_ablation]
# evaluation
python -m research.eval_pipeline status
python -m research.eval_pipeline run --dataset commons_film --split analysis
python -m research.eval_pipeline run --dataset commons_film --split test --final      # once per dataset snapshot
# orchestration
python -m research.orchestrator seed [--auto-research]     # standard tasks (idempotent)
python -m research.orchestrator status
python -m research.orchestrator tick                       # one scheduling pass
python -m research.orchestrator loop --interval 900        # periodic ticks; sleeps between them
python -m research.orchestrator add <task_id> <data|research|eval> --command "..." [--depends dataset:commons_film] [--gpu-mb 6000]
```

## Data pipeline

`research/downloader.py` is the resumable, rate-limit-aware downloader. Per-file state lives in an append-only
JSONL (`download_state.jsonl`: pending -> downloading -> validated | retryable_failure | permanent_failure |
quarantined). Files are written to `<name>.part` and renamed only after the caller's validator decoded them, so a
partial file is never treated as valid; every validated file is recorded with its sha256 and re-verified on resume
(a tampered or truncated file is fetched again). Stable ids (sha1 of the Commons title) prevent duplicates across runs.
Requests use connect/read timeouts, bounded exponential backoff, and honour HTTP 429/503 `Retry-After` up to a cap
(120 s); after 20 consecutive rate-limit waits with no success the run exits `rate_limited` (resumable later) instead
of sleeping for hours. A *stall* means no HTTP response at all for `--stall-minutes` (dead network), and is distinct
from a slow or rate-limiting server. `--max-minutes` caps one run (`paused`, resumable). A heartbeat line and
`download_status.json` are written every minute even when nothing completes; the manifest, attribution file and
summary are rewritten at every 100 files and at exit, so a killed run loses nothing.

What the Commons incident taught (2026-11-07): the first download spent 2.7 of 3.1 hours sleeping on
`Retry-After: 600` responses. Originals on `upload.wikimedia.org`, and the "thumbnail_unscaled" alias the API returns
for files whose requested width rounds up to the original, are rate-limited per client; scaled renditions are served
only at the standard widths (20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840 px; anything else is HTTP 400).
`scripts/ingest_commons_film.py` therefore requests the largest standard width below the original (1280 for large
files, downsized locally to 1024) and never fetches originals. After the incident the client was throttled on every
rendition (`Retry-After: 10`), so the acquisition task is retried by the orchestrator with a 30-minute backoff that
doubles per attempt (12 attempts max); each attempt resumes where the last stopped.

`research/data_pipeline.py` keeps dataset readiness in `data/manifests/dataset_state.json`:
pending -> downloading -> validated -> ready, failures retryable_failure / permanent_failure / quarantined. `ready`
requires the manifest (minimum rows), every file present, a deterministic sample that decodes and matches its recorded
sha256, and for acquired datasets a complete acquisition (`summary.json: complete`) or an explicit
`validate --allow-partial`, which records the manifest sha256 so evaluations name the snapshot they used. The
evaluation task depends on `dataset:commons_film`; it is launched by the orchestrator on the first tick after the
dataset becomes ready, and features are recomputed only when the snapshot changes.

Provenance: every Commons manifest row carries licence, attribution, source page URL, download URL, uploader,
category-derived stock claim (`stock_label_source: commons_category`), film type, roll/scanner/camera categories when
present (hidden-category backfill: `--hidden-cats`), and `training_allowed: false` in `configs/datasets.yaml`.

## Research pipeline

`research/research_loop.py` runs one cycle: read the registry, pick the highest value-per-cost *pending* backlog entry
that has a runnable spec (`command`, `result_json`, `metrics_of_interest`) and whose `datasets` are ready, estimate the
cost (`gpu_minutes`, `gpu_mb`; defaults are conservative), check free GPU memory and the session budget, register and
run it through the metered runner, compare with its `baseline_id` under `research/evaluator.py` (a real-film regression
rejects; synthetic-up-real-down rejects), apply the entry's pre-registered `falsify` rule (`{metric, below|above}`),
write the decision (supported / rejected / inconclusive) to the backlog and registry, and append the result paragraph to
`docs/EXPERIMENT_LOG.md`. Entries without a spec are listed as "needs spec or data", never run. Rejected hypotheses are
not re-selected (the registry remembers), so a materially new hypothesis needs a new id and a new falsification rule.

Selection rules are fixed before a run: the entry's `falsified_if` / `falsify` and the evaluator's CI-separated
`min_delta` 0.02. Test-split results are never read by the loop.

## Evaluation pipeline

`research/eval_pipeline.py` evaluates the fixed benchmark candidates plus every eligible student checkpoint
(`checkpoints/student_style_*/model.onnx` exporting `style_embedding`) on a ready real-film dataset. The Commons
protocol (`scripts/eval_commons_film.py`) hashes uploaders into analysis/test halves, excludes the query's uploader
from retrieval, bootstraps CIs over uploaders, and reports same-stock and same-type precision@5, uploader share
(photographer leakage), the scanner probe where scanner categories exist (coverage is reported; most files have none),
and film-vs-emulation authenticity. Pre-registered decision rules are in the module docstring: primary metric
same-stock p@5, `improved` needs delta >= 0.02 and non-overlapping CIs, and leakage guards (uploader share excess
<= 0.02, scanner probe <= +0.05) reject regardless of the primary metric. The test half runs once per dataset
(`--final`; a second run needs `--force-final` and is recorded as forced).

## Orchestrator

`research/task_store.py` keeps tasks in `research/tasks.json` under a file lock (safe for concurrent processes);
`research/task_wrapper.py` runs one task detached, holds the GPU lock for GPU tasks (`research/.gpu.lock`; a busy lock
is exit 3 = retryable), enforces the task's wall-clock cap (process group killed, outcome retryable) and records the
exit code. A tick: crash recovery (running tasks whose process is gone are re-queued or failed when attempts are
exhausted; datasets left `downloading` by a dead process become retryable), dependency check (`task:<id>` done,
`dataset:<id>` ready; a permanently failed dependency marks dependants `blocked`, visibly), resource check (free GPU
memory from nvidia-smi plus a 500 MB margin, one GPU task at a time, `--max-parallel`), launch. Retryable tasks wait
`retry_backoff_s * 2^(attempts-1)`. Events go to `reports/orchestrator/events.jsonl`, task output to
`reports/orchestrator/<task>.log`.

Recovery procedures:
- Orchestrator or machine restarted: `python -m research.orchestrator tick` (recovery runs first; nothing is lost).
- Download stalled or rate-limited: nothing to do; the task retries with backoff. To resume by hand:
  `python -m research.data_pipeline acquire commons_film` (or the ingest script directly). To inspect:
  `data/raw/commons_film/download_status.json`, `download.log`.
- A dataset marked `quarantined` (a file failed to decode or its sha256 changed): inspect `last_error`, fix or delete
  the file, `python -m research.data_pipeline reset <id>` then `validate`.
- A task `failed`: read `reports/orchestrator/<task>.log`, fix, then re-add with `replace` (Python API) or a new id.
- Budget exhausted: the runner marks experiments `blocked` with the reason; `research/ledger.json` is the session tally.

## Current limitations

- Commons acquisition is bound by Wikimedia's per-client throttle; at 2 s spacing a full run of ~5,500 files takes
  about 3-4 hours *if* the throttle lifts; the dataset stays `validated`/partial until then and is evaluated only on a
  recorded snapshot.
- The evaluation pipeline covers `commons_film`; BlueNeg and CommonCatalog film scans are evaluated by the benchmark
  scripts (`scripts/style_benchmark.py`, `scripts/scanner_effects.py`) and are not yet routed through it.
- Falsification rules must be scalar thresholds; narrative `falsified_if` text is recorded but not executed.
- The research loop runs one experiment per cycle and only entries that ship a script; it does not write scripts.
