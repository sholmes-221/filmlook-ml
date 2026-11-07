"""Benchmark candidates from student checkpoints embedded with scripts/student_bench_embed.py.

  STUDENT_TAGS=style_fs_v2_sel,style_fs_v2_real python scripts/style_benchmark.py --extra research.candidates_student:build
"""
from __future__ import annotations

import os

import numpy as np

from filmlook_ml import paths as P


def build(F, rows, dev):
    out = {}
    ids = np.array([r["id"] for r in rows])
    for tag in [t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]:
        z = np.load(P.EMBEDDINGS / f"student_{tag}_bench.npz")
        assert (z["ids"] == ids).all(), f"image list mismatch for {tag}"
        out[f"student_{tag}_64"] = (lambda f, Z=z["student64"]: Z)
    return out
