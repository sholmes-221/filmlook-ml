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

## Standing storage (not charged to any session)
`data/hf_cache` 39 GB (SDXL, Z-Image-Turbo, LoRAs, BlueNeg previews), `data/synthetic` 0.2 GB,
`data/embeddings` 0.1 GB + benchmark cache, `checkpoints` 0.3 GB. Nothing is deleted without approval.
