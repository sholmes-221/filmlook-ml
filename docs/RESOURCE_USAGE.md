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

## Session 2 (2026-11-07, benchmark v1 onward)
Running totals are in the ledger; this file is updated at session end.

## Standing storage (not charged to any session)
`data/hf_cache` 39 GB (SDXL, Z-Image-Turbo, LoRAs, BlueNeg previews), `data/synthetic` 0.2 GB,
`data/embeddings` 0.1 GB + benchmark cache, `checkpoints` 0.3 GB. Nothing is deleted without approval.
