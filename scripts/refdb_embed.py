#!/usr/bin/env python
"""DINOv2-small embeddings for the reference database (GPU, deterministic, resumable, versioned).

  python scripts/refdb_embed.py [--batch 64] [--device cuda]

For every eligible, downloaded, non-duplicate work it stores two L2-normalised 384-d float32 vectors in the manifest's
``embeddings`` table: ``dinov2s_cls_224`` (the CLS token; the primary vector) and ``dinov2s_patchmean_224`` (mean of
the patch tokens). Preprocessing is the existing pipeline's ``filmlook_infer.pixels.dinov2_pixels`` (shortest side to
256 bicubic, centre crop 224, ImageNet normalisation), so these vectors are comparable with the ones the website's ML
backend would compute with the same function. Rows already present for (work, model, version) are skipped, so an
interrupted run resumes. Version string: ``v1-<model revision sha>-fp32``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

from filmlook_ml.refdb import manifest  # noqa: E402

MODEL = "facebook/dinov2-small"


def load_model(device: str):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(MODEL).to(device).eval()
    rev = getattr(m.config, "_commit_hash", None) or "unknown"
    return m, f"v1-{str(rev)[:12]}-fp32"


@torch.no_grad()
def embed(model, pixels: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    out = model(pixel_values=torch.from_numpy(pixels).to(device)).last_hidden_state.float()
    cls, patch = out[:, 0], out[:, 1:].mean(dim=1)
    n = lambda t: torch.nn.functional.normalize(t, dim=1).cpu().numpy().astype(np.float32)  # noqa: E731
    return n(cls), n(patch)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, version = load_model(a.device)
    with manifest.connect() as c:
        todo = [r["work_id"] for r in c.execute(
            "SELECT work_id FROM works w WHERE rights_status = 'eligible' AND sha256 IS NOT NULL AND duplicate_of IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.work_id = w.work_id AND e.model = 'dinov2s_cls_224' AND e.version = ?)", (version,))]
    print(f"{len(todo)} works to embed with {MODEL} {version} on {a.device}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        ids = todo[i:i + a.batch]
        px = np.stack([dinov2_pixels(Image.open(manifest.image_path(w)).convert("RGB"), 224) for w in ids])
        cls, patch = embed(model, px, a.device)
        stamp = manifest.now()
        with manifest.connect() as c:
            for w, v1, v2 in zip(ids, cls, patch):
                c.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", (w, "dinov2s_cls_224", version, 384, v1.tobytes(), stamp))
                c.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", (w, "dinov2s_patchmean_224", version, 384, v2.tobytes(), stamp))
        print(f"{i + len(ids)}/{len(todo)} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
