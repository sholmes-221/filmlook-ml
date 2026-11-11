"""A second DINOv2-small embedding of a native-resolution centre crop (224 px, no resize) for the film set and the bank.

  python -m research.claude_annotations.crop_embed   -> data/embeddings/photographers/{filmset,bank}_crop_prod.npz

Why: the photo gate's embedding is computed after resizing the whole picture to 224 px, which erases grain and fine
highlight structure, the very things the annotator reads to judge film-likeness (Sonnet 0.958 zero-shot vs the local
head's 0.871). A crop at native resolution keeps them. Same ONNX model and preprocessing as production, applied to
the crop instead of the resized frame, so production could compute it with one more encoder pass per upload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.pixels import MEAN, STD  # noqa: E402
from filmlook_ml.refdb import manifest  # noqa: E402
from research.claude_annotations import pipeline as P  # noqa: E402

EMB = ROOT / "data" / "embeddings" / "photographers"


def native_crop_pixels(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    if min(img.size) < size:
        img = img.resize((max(size, img.width), max(size, img.height)), Image.BICUBIC)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    a = (np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0 - MEAN) / STD
    return a.transpose(2, 0, 1).copy()


def main() -> None:
    import onnxruntime as ort
    s = ort.InferenceSession(str(ROOT / "checkpoints/refdb/dinov2_small_cls_patch.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    enc = lambda img: s.run(None, {"pixel_values": native_crop_pixels(img)[None]})[0][0]  # noqa: E731
    rows = [json.loads(l) for l in (ROOT / "data/lut/film_likeness/manifest.jsonl").read_text().splitlines()]
    X = np.stack([enc(Image.open(r["path"])) for r in rows]).astype(np.float32)
    np.savez(EMB / "filmset_crop_prod.npz", X=X, paths=np.array([r["path"] for r in rows])); print("filmset", X.shape)
    bank = sorted(P.annotated("bank", P.cfg()["models"]["cheap"]))
    XB = np.stack([enc(Image.open(manifest.image_path(w))) for w in bank]).astype(np.float32)
    np.savez(EMB / "bank_crop_prod.npz", X=XB, work_ids=np.array(bank)); print("bank", XB.shape)


if __name__ == "__main__":
    main()
