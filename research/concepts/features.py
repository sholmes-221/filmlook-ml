"""Production photo-gate embedding for every downloaded image (and a parity check against the deployed ONNX encoder).

  python -m research.concepts.features            -> data/concepts/feats_dinov2s.npz (ids, X: N x 768 float32)
  python -m research.concepts.features --parity   -> ONNX vs torch cosine on 200 images

Embedding spec (identical to the photo gate, `dinov2s-cls-patch-768`): facebook/dinov2-small (rev ed25f3a31f01, frozen),
`filmlook_infer.pixels.dinov2_pixels` (shorter side -> 256 bicubic, centre crop 224, ImageNet mean/std),
last_hidden_state -> [L2norm(CLS), L2norm(mean of the 256 patch tokens)], 768 float32, not renormalised as a whole.
Computed in fp32 with TF32 off, exactly as scripts/photo_gate_train.py did for the deployed gate.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from research.concepts.common import DATA, ROOT, read_jsonl

sys.path.insert(0, str(ROOT / "src"))
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

ONNX = ROOT / "checkpoints" / "refdb" / "dinov2_small_cls_patch.onnx"
WEB_ONNX = "/home/shanmo/misc/film-evaluation/models/photo_gate/dinov2_small_cls_patch.onnx"   # read-only


def pixels(path) -> np.ndarray:
    im = Image.open(path); im.load()
    return dinov2_pixels(im.convert("RGB"), 224)


@torch.no_grad()
def embed_torch(paths, batch=128) -> np.ndarray:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = []
    with ThreadPoolExecutor(8) as ex:
        for b in range(0, len(paths), batch):
            px = np.stack(list(ex.map(pixels, paths[b:b + batch])))
            h = m(pixel_values=torch.from_numpy(px).cuda()).last_hidden_state.float()
            v = torch.cat([torch.nn.functional.normalize(h[:, 0], dim=1), torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)], dim=1)
            out.append(v.cpu().numpy())
            if (b // batch) % 50 == 0:
                print(f"{b + len(px)}/{len(paths)}", flush=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    return np.concatenate(out).astype(np.float32)


def embed_onnx(paths, onnx_path=WEB_ONNX) -> np.ndarray:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    return np.stack([s.run(None, {"pixel_values": pixels(p)[None]})[0][0] for p in paths]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    man = read_jsonl(DATA / "manifest.jsonl")
    ids = [m["image_id"] for m in man]
    paths = [DATA / m["path"] for m in man]
    if a.parity:
        sel = paths[::max(1, len(paths) // 200)][:200]
        t, o = embed_torch(sel), embed_onnx(sel)
        cos = (t * o).sum(1) / np.linalg.norm(t, axis=1) / np.linalg.norm(o, axis=1)
        print(f"torch vs deployed ONNX cosine over {len(sel)}: min {cos.min():.5f} mean {cos.mean():.5f}; max |diff| {np.abs(t - o).max():.2e}")
        return
    t0 = time.time()
    X = embed_torch(paths)
    np.savez(DATA / "feats_dinov2s.npz", ids=np.array(ids), X=X)
    print(f"{len(ids)} embedded in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
