# Photographer retrieval on the public-domain reference bank

Question: given a photograph, which reference photographs is it closest to, and does the closest work's author
tell us anything? Everything here uses only works that their source marks public domain or CC0
(`scripts/refdb_discover.py`, `scripts/refdb_download.py`, `src/filmlook_ml/refdb/`). Photographs by in-copyright
photographers are never downloaded.

```bash
python scripts/refdb_discover.py --only <ids>            # metadata + licence manifest (no image bytes)
python scripts/refdb_download.py --only <ids> --cap 150  # scaled renditions, validated, de-duplicated
bash research/photographers/refresh.sh [seed]            # embeddings, measured descriptors, benchmark, serving index
python -m research.photographers.evaluate --seed 1 --out reports/photographers/benchmark_seed1.json
(cd ../film-evaluation && python tools/vendor_refmatch.py)   # copies vectors and provenance into the website
```

- `embed.py`: the production encoder (the ONNX DINOv2-small the website's photo gate runs) and three comparison
  encoders (DINOv2-base, SigLIP-base, CLIP ViT-B/32).
- `evaluate.py`: the protocol (leakage groups by sitting, perceptual hash and embedding; 60 / 20 / 20 by group;
  every choice on validation), retrieval and photographer-level metrics, baselines, open-set readings, late fusion.
- `export.py`: the serving index (vectors, per-work provenance, thresholds and fusion weights from validation).

Results and their limits: film-evaluation `docs/PHOTOGRAPHER_RETRIEVAL_BENCHMARK.md`.
Claude annotations as an extra feature space: `research/claude_annotations/` (film-evaluation
`docs/CLAUDE_API_RESEARCH.md`).
