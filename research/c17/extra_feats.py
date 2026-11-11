"""Extra real-scan features for C17: the website's 36 film statistics (fine-grain noise, highlight and shadow
behaviour) via research.c08.prod_inputs, and the 312 colour values after grey-world normalisation (each channel scaled so
the image mean is neutral: removes a scanner's or inversion's global cast, keeps relative rendering).
    python -m research.c17.extra_feats   # -> data/c08/pairs/extra_real/{f36.npy, colour_greyworld.npy}
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from research.c08.prod_inputs import compute
from research.lut.features import colour_features_from_array
from research.lut.synthetic_library import load_photo

P = Path("data/c08/pairs/extra_real")


def greyworld(path):
    x = load_photo(path, 512).astype(np.float32)
    g = x.reshape(-1, 3).mean(0); x = np.clip(x * (g.mean() / np.maximum(g, 1e-3)), 0, 255).astype(np.uint8)
    return colour_features_from_array(x).astype(np.float32)


def main():
    paths = [json.loads(l)["path"] for l in open(P / "meta.jsonl")]
    with Pool(24) as p:
        np.save(P / "colour_greyworld.npy", np.stack(p.map(greyworld, paths, chunksize=8)))
    F, _ = compute(paths)
    np.save(P / "f36.npy", np.nan_to_num(F))
    print("done", len(paths))


if __name__ == "__main__":
    main()
