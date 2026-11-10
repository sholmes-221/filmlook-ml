"""Frozen DINOv2-small embeddings (CLS token and mean patch token) for every dataset row.

  python -m research.species.features [--batch 128]   -> data/species/feats/dinov2s.npz (idx, cls, patch_mean)

Model: facebook/dinov2-small (Apache-2.0) via transformers, 224 px centre crop, ImageNet normalisation (the same
preprocessing as the production photo gate). Frozen: no gradient, no fine-tuning.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from research.species import common as C
from research.species import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = D.load_rows()
    dev = D.device()
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    dl = DataLoader(D.ImageRows(rows, {}, D.eval_transform(224)), batch_size=a.batch, num_workers=a.workers)
    cls, pm, idx = [], [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        for x, _, i in dl:
            h = model(pixel_values=x.to(dev)).last_hidden_state
            cls.append(h[:, 0].float().cpu().numpy()); pm.append(h[:, 1:].mean(1).float().cpu().numpy()); idx.append(i.numpy())
    out = C.DATA / "feats"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "dinov2s.npz", idx=np.concatenate(idx), cls=np.concatenate(cls), patch_mean=np.concatenate(pm))
    el = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
    print(f"{len(rows)} rows embedded in {el:.0f} s on {dev} (peak {peak:.0f} MB)")
    C.append_jsonl(C.DATA / "gpu_ledger.jsonl", {"step": "features_dinov2s", "gpu_seconds": el if dev.type == "cuda" else 0, "rows": len(rows), "peak_mb": peak})


if __name__ == "__main__":
    main()
