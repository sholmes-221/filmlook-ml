"""Head inputs computed exactly as production computes them, by calling the website's own code (read-only import of
~/misc/film-evaluation): prepare_image -> prepared JPEG -> engine load_image + describe -> film_features (36 values);
photo_gate.dinov2_pixels on the decoded JPEG -> DINOv2-small blocks 4, 8, 12 (fp32 PyTorch, identical to the ONNX).

    python -m research.c08.prod_inputs   # -> data/c08/encoders/prod_inputs.npz (F36, layers, paths)
"""
from __future__ import annotations

import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

WEB = Path.home() / "misc" / "film-evaluation"
OUT = Path("data/c08/encoders/prod_inputs.npz")


def _web():
    if str(WEB) not in sys.path:
        sys.path.insert(0, str(WEB))


def one(path: str):
    _web()
    from PIL import Image

    from filmlook.filmstyle.descriptors import describe
    from filmlook.filmstyle.engine import load_image
    from filmlook.filmstyle.likeness_model import film_features
    from filmlook.image_utils import prepare_image
    from filmlook.photo_gate import dinov2_pixels
    jpeg = prepare_image(Path(path).read_bytes()).data
    img = load_image(jpeg)
    f = film_features(img, describe(img))
    im = Image.open(io.BytesIO(jpeg)); im.load()
    return f.astype(np.float32), dinov2_pixels(im).astype(np.float32)


def compute(paths: list[str], workers: int = 16):
    import torch
    from transformers import AutoModel
    with Pool(workers) as p:
        res = p.map(one, paths, chunksize=4)
    F = np.stack([r[0] for r in res]); px = np.stack([r[1] for r in res])
    m = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(px), 128):
            o = m(pixel_values=torch.from_numpy(px[i:i + 128]).cuda(), output_hidden_states=True)
            feats = []
            for layer in (4, 8, 12):
                h = o.last_hidden_state if layer == 12 else o.hidden_states[layer]
                feats += [h[:, 0], h[:, 1:].mean(1)]
            outs.append(torch.cat(feats, 1).cpu().numpy())
    return F, np.concatenate(outs).astype(np.float32)


def main() -> None:
    rows = [json.loads(l) for l in open("data/lut/film_likeness/manifest.jsonl")]
    paths = [r["path"] for r in rows]
    F, L = compute(paths)
    np.savez(OUT, F36=F, layers=L, paths=np.array(paths))
    print("saved", F.shape, L.shape, "finite stats rows", int(np.isfinite(F).all(1).sum()))


if __name__ == "__main__":
    main()
