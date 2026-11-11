#!/usr/bin/env bash
# Rebuild everything that depends on the bank after new works were downloaded:
# embeddings (production encoder + comparison encoders), measured descriptors, benchmark, serving index.
#   bash research/photographers/refresh.sh [seed]
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-0}"
for m in prod dinov2_base siglip_base clip_b32; do .venv/bin/python -m research.photographers.embed --model "$m" | tail -1 || echo "encoder $m failed (comparison only)"; done
(cd ../film-evaluation && .venv/bin/python tools/refdb_descriptors.py --workers 12)
.venv/bin/python scripts/preflight_check.py | tail -1
.venv/bin/python -m research.photographers.evaluate --seed "$SEED" | grep -E 'R@1|fusion|majority'
.venv/bin/python -m research.photographers.export --seed "$SEED" | head -1
