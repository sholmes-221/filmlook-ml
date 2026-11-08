# Resource usage

Source of truth: `research/ledger.json` (written by `research/experiment_runner.py` for every metered run: wall
seconds, GPU use, peak GPU memory on the card, disk delta on watched paths). Budget per autonomous session: 4 GPU-hours
and 20 GB of additional disk. Nothing paid was used at any point (no cloud GPU, no paid API calls).

## Hardware
Development workstation: one NVIDIA RTX 4090 (24 GB) shared with an unrelated job (~5.7 GB resident), 32 CPU
cores, 61 GB RAM. The owner's training GPU is an RTX 4070 (12 GB); the production host is an Oracle E2.1.Micro
(1/8 OCPU, 1 GB). Timings quoted in reports name the machine they were measured on.

## Session 1 (2026-11-07, autonomous loop E8-E18)
- GPU: 64.0 minutes of 240 (13 metered runs; the 35-minute Z-Image generation that was already running at session
  start was charged manually).
- Disk charged: 0.22 GB (synthetic images and reports). The 33 GB Z-Image-Turbo download and the 7.8 GB SDXL
  download preceded the budget and are reusable caches under `data/hf_cache/` (39 GB in total).
- A later brief quoted "8.2 GB" for this session; the ledger records 0.22 GB.

## Session 2 (2026-11-07, benchmark v1, E19, B1, E10)
- GPU: 17.1 metered minutes of 240 (B0 feature cache 5.3 min, E19 0.1, B1 0.3, E10 SDXL strength-0.5 generation
  11.5 min at 9.9 GB peak VRAM) plus about 5 unmetered minutes (two `eval_sdxl_strength.py` passes, the E19
  distinctness diagnostic, `abstention_analysis.py`): about 22 minutes in total.
- Disk charged: 0.11 GB (benchmark cache 37 MB, SDXL strength-0.5 set 70 MB, reports).
- Ledger totals across both sessions: 81 GPU-minutes, 0.33 GB.

## Session 3 (2026-11-07, Commons acquisition, E22-E26, pipeline separation)
- GPU: 3.0 metered minutes of 240 (E22 0.05, E23a/b/c 2.1 for three student trainings, E24 0.06, E25 0.1,
  E26 0.5 incl. the 12-layer feature pass at 7.3 GB peak on the card) plus about 8 unmetered minutes (two
  `student_bench_embed.py` passes, two `style_benchmark.py` runs with extra candidates, the E26 attempt that failed on
  a script bug): about 11 minutes in total.
- CPU/network: the Commons enumeration (~50 min of serial API calls, repeated once after a crash) and the download
  (3.1 h first run, mostly asleep on `Retry-After: 600`; see D1 in `docs/EXPERIMENT_LOG.md`), all off-GPU.
- Disk charged: 0.02 GB (three student checkpoints 13 MB, 266 Commons renditions 63 MB, per-layer feature cache
  `data/embeddings/bench_layers_v1.npz` 115 MB, reports).
- Ledger totals across the three sessions: 84 GPU-minutes, 0.34 GB.

## Session 4 (2026-11-07/08, 20-hour orchestrator, dataset discovery, E27b-E31)
- GPU: 0.2 metered minutes (E28 5.8 s, E30 4.6 s, E31 1.8 s CPU; all on cached features) plus about 6 unmetered
  minutes: `student_bench_embed.py` for the L3 student (CPU), two real-film feature passes on `commoncatalog_film`
  (487 and 905 images, ~90 s each), one on `aic_photographs` (64 s), one evaluation smoke pass on a partial snapshot,
  and the two evaluation-pipeline runs (SupCon projections, seconds). The acquisition tasks use no GPU.
- Disk (not charged by the ledger, which meters experiment runs): acquired evaluation datasets 1.6 GB in total:
  `commoncatalog_film` 1,005 MB (905 original JPEGs), `artefact_film` 418 MB (88 files), `aic_photographs` 134 MB
  (593 renditions); discovery registry and candidate files < 5 MB. Within the 20 GB budget by a wide margin.
- Network etiquette: Wikimedia retries honour `Retry-After` (21 requests per attempt, doubling backoff, four attempts
  all throttled); Hugging Face parquet range reads at 2 workers; figshare and the AIC API in series with >= 1 s
  spacing (the AIC API caps a query at 1,000 results, so the listing is partitioned by decade).

## Standing storage (not charged to any session)
`data/hf_cache` 39 GB (SDXL, Z-Image-Turbo, LoRAs, BlueNeg previews), `data/synthetic` 0.2 GB, `data/raw/commons_film` 0.07 GB (growing; evaluation-only),
`data/embeddings` 0.1 GB + benchmark caches (0.15 GB), `checkpoints` 0.3 GB. Nothing is deleted without approval.

## Session 5 (2026-11-08, E29 investigation E32-E39)
- GPU: E32-E37 ran on cached features (CPU, 6-81 s each; the tiny SupCon heads train on the card in seconds). E38 trained
  three MobileNetV3-Small students (186 s wall, peak 7.6 GB on the card including the 5.7 GB other job) and E39 three
  more (188 s); E40-E41 ran on cached features (9-13 s); E42 one retrain (69 s) and E43 two (147 s); the final
  held-out pass E44 is CPU ONNX (27 s); E45 two retrains (146 s); two frozen-teacher feature passes over the 601 CC-BY training originals (about 60 s each). Session
  total about 16 GPU-minutes, all metered through the research loop; ledger total across sessions 5,765 GPU-seconds
  (96 minutes) of the 4-hour-per-session budget.
- Disk: thirteen student checkpoints (57 MB), feature caches `ccby_train_originals_features_v1` and `student_matrix_*`
  (< 30 MB), reports < 5 MB. `checkpoints` 299 MB, `data/embeddings` 316 MB in total. Nothing deleted.
- Network: none (no acquisition this session; the Commons retry stays with the orchestrator's backoff).
