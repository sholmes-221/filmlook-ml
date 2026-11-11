"""Embeddings of the public-domain photographer bank, one file per encoder.

  python -m research.photographers.embed --model prod|dinov2_base|siglip_base|clip_b32 [--device cuda]

``prod`` is the ONNX file the website's photo gate runs for every upload (DINOv2-small, [CLS, mean patch], each half
L2-normalised, 768 values) with the website's preprocessing, so bank and query vectors come from the same function.
The other encoders are comparisons run with their own Hugging Face preprocessing. Output:
``data/embeddings/photographers/<model>.npz`` (work_ids, X float32). Works already present are skipped (resumable).
Only eligible (public domain / CC0), downloaded, non-duplicate works are read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

from filmlook_ml.refdb import manifest  # noqa: E402

OUT = ROOT / "data" / "embeddings" / "photographers"
HF = {"dinov2_base": "facebook/dinov2-base", "siglip_base": "google/siglip-base-patch16-224", "clip_b32": "openai/clip-vit-base-patch32"}


def works() -> list[str]:
    with manifest.connect() as c:
        return [r["work_id"] for r in c.execute(
            "SELECT work_id FROM works WHERE rights_status = 'eligible' AND sha256 IS NOT NULL AND duplicate_of IS NULL ORDER BY work_id")]


def prod_encoder(device: str):
    import onnxruntime as ort
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    s = ort.InferenceSession(str(ROOT / "checkpoints" / "refdb" / "dinov2_small_cls_patch.onnx"), providers=prov)
    return lambda imgs: np.concatenate([s.run(None, {"pixel_values": dinov2_pixels(i)[None]})[0] for i in imgs])


def hf_encoder(name: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc, model = AutoImageProcessor.from_pretrained(HF[name]), AutoModel.from_pretrained(HF[name]).to(device).eval()

    @torch.no_grad()
    def enc(imgs):
        px = proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt")["pixel_values"].to(device)
        if name == "dinov2_base":
            h = model(pixel_values=px).last_hidden_state
            v = torch.cat([torch.nn.functional.normalize(h[:, 0], dim=1), torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)], dim=1)
        else:
            v = model.get_image_features(pixel_values=px)
            v = getattr(v, "pooler_output", v)
            v = torch.nn.functional.normalize(v, dim=1)
        return v.float().cpu().numpy()
    return enc


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / f"{a.model}.npz"
    have: dict[str, np.ndarray] = {}
    if path.exists():
        d = np.load(path, allow_pickle=False); have = dict(zip(d["work_ids"].tolist(), d["X"]))
    todo = [w for w in works() if w not in have]
    enc = prod_encoder("cpu") if a.model == "prod" else hf_encoder(a.model, a.device)
    for i in range(0, len(todo), a.batch):
        ids = todo[i:i + a.batch]
        imgs = [Image.open(manifest.image_path(w)) for w in ids]
        for w, v in zip(ids, enc(imgs)):
            have[w] = v.astype(np.float32)
        if i % (a.batch * 20) == 0:
            print(a.model, i, len(todo), flush=True)
    ids = sorted(w for w in have if w in set(works()))
    np.savez(path, work_ids=np.array(ids), X=np.stack([have[w] for w in ids]))
    print(a.model, "saved", len(ids), np.stack([have[w] for w in ids]).shape)


if __name__ == "__main__":
    main()
