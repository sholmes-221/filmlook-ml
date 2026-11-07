"""E27 candidates: hybrid teachers on early-layer mean/std features (from the E26 per-layer cache) for style_benchmark.

  python scripts/style_benchmark.py --split val --extra research.candidates_e27:build --candidates hybrid_supcon_family,e27_L3_ms,e27_L2-4_ms

Same recipe as hybrid_supcon_family (explicit28 ++ DINOv2 patch mean++std -> family SupCon on filmsynth TRAIN), with
the DINOv2 statistics taken from layer 3, or layers 2-4 concatenated, instead of layers 4+8.
"""
from __future__ import annotations

import numpy as np

import synth_experiment as X

from filmlook_ml import paths as P

CACHE = P.EMBEDDINGS / "bench_layers_v1.npz"
VARIANTS = {"e27_L3_ms": (3,), "e27_L2-4_ms": (2, 3, 4)}


def build(F, rows, dev):
    z = np.load(CACHE); assert (z["ids"] == np.array([r["id"] for r in rows])).all(), "per-layer cache does not match the benchmark image list"
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    tr = (st == "filmsynth") & (sp == "train"); out = {}
    for name, layers in VARIANTS.items():
        H = np.hstack([F["explicit"]] + [z[f"ms_L{l}"].astype(np.float32) for l in layers])
        proj = X.supcon(H[tr], fam[tr], dev); out[name] = (lambda f, Z=proj(H): Z)
    return out
