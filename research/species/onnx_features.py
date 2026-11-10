"""Production-identical embeddings: the website's photo-gate encoder (dinov2_small_cls_patch.onnx, [CLS, mean patch], each
half L2-normalised, 768 values) with the website's exact preprocessing (filmlook_infer.pixels.dinov2_pixels, pinned by a
website test to filmlook.photo_gate.dinov2_pixels), applied to the same prepared JPEG the site analyses (EXIF-transposed,
long edge <= 2048, JPEG q90). Heads trained on these vectors read exactly what the server computes.

  python -m research.species.onnx_features [--threads 16] [--batch 32]   -> data/species/feats/onnx768.npz (idx, emb)
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.species import common as C
from research.species import data as D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

ONNX = Path("/home/shanmo/misc/film-evaluation/models/photo_gate/dinov2_small_cls_patch.onnx")


def site_prepared(path: Path) -> Image.Image:
    """What the site's prepare_image produces, closely enough for the encoder: upright RGB, long edge <= 2048, JPEG q90."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90, subsampling=0)
    return Image.open(io.BytesIO(buf.getvalue()))


def pixels(path: Path) -> np.ndarray:
    return dinov2_pixels(site_prepared(path), 224)


def embed_paths(paths: list[Path], threads: int = 16, batch: int = 32) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    out = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(paths), batch):
            x = np.stack(list(pool.map(pixels, paths[i:i + batch])))
            out.append(sess.run(None, {"pixel_values": x})[0])
            if (i // batch) % 50 == 0:
                print(f"  {i + len(x)}/{len(paths)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    rows = D.load_rows()
    t0 = time.time()
    emb = embed_paths([D.abs_path(r) for r in rows], a.threads, a.batch)
    out = C.DATA / "feats"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "onnx768.npz", idx=np.arange(len(rows)), emb=emb)
    print(f"{len(rows)} rows embedded with the production encoder in {time.time() - t0:.0f} s (CPU)")


if __name__ == "__main__":
    main()
