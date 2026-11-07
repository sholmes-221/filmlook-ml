"""MobileNetV3-Small student for constrained CPU hosts (main FilmLook model, compact variant).

Heads: the same targets as the DINOv2 baseline (capture type from EXIF; six attribute classes from deterministic
measurements). Optional knowledge distillation: a projection of the student's pooled feature is pulled toward the
frozen DINOv2-small embedding of the same image (cosine loss). No Claude labels are used.

Images go through ``filmlook_infer.compact.preprocess`` geometry: short side 256, centre crop 224, ImageNet
normalisation. The train-time cache stores 256x256 centre squares; training takes random 224 crops and flips but
no colour augmentation, because colour is a target. Validation and test use the runtime ``preprocess`` itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from filmlook_infer.compact import MEAN, STD

STUDENT_VERSION = "filmlook-student-mnv3s-v0"


class Student(nn.Module):
    def __init__(self, heads: dict[str, int], kd_dim: int = 768, pretrained: bool = True):
        super().__init__()
        import torchvision
        m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        self.features, self.pool = m.features, nn.AdaptiveAvgPool2d(1)
        dim = 576
        self.neck = nn.Sequential(nn.Linear(dim, 256), nn.Hardswish(), nn.Dropout(0.2))
        self.head_names = list(heads)
        self.heads = nn.ModuleList(nn.Linear(256, n) for n in heads.values())
        self.kd = nn.Linear(dim, kd_dim)

    def forward(self, x, with_kd: bool = False):
        f = self.pool(self.features(x)).flatten(1)
        h = self.neck(f)
        outs = [hd(h) for hd in self.heads]
        return (outs, self.kd(f)) if with_kd else outs


class ExportWrapper(nn.Module):
    """ONNX graph: pixel_values -> one logits tensor per head (no KD branch)."""

    def __init__(self, s: Student):
        super().__init__()
        self.s = s

    def forward(self, pixel_values):
        return tuple(self.s(pixel_values))


def load_square(path: str, side: int = 256) -> np.ndarray:
    """Same geometry as compact.preprocess, but keep a side x side centre square (uint8)."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    if im.format == "JPEG":
        im.draft("RGB", (side * 2, side * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = side / min(im.size)
    im = im.resize((max(side, round(im.width * s)), max(side, round(im.height * s))), Image.BILINEAR)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    return np.asarray(im.crop((left, top, left + side, top + side)), dtype=np.uint8)


def to_tensor(batch_u8: np.ndarray, dev) -> torch.Tensor:
    x = torch.from_numpy(batch_u8).to(dev).float().div_(255)
    x = (x - torch.tensor(MEAN, device=dev)) / torch.tensor(STD, device=dev)
    return x.permute(0, 3, 1, 2).contiguous()


def export_onnx(s: Student, out_dir: Path, meta: dict, size: int = 224) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    w = ExportWrapper(s.eval().cpu())
    x = torch.zeros(1, 3, size, size)
    torch.onnx.export(w, (x,), str(out_dir / "model.onnx"), input_names=["pixel_values"],
                      output_names=list(s.head_names), dynamic_axes={"pixel_values": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    # No INT8: dynamic quantization of this depthwise CNN dropped top-1 agreement with fp32 to 11-47% (measured
    # 2026-11-07), and the fp32 graph is only ~4 MB. Static (calibrated QDQ) quantization is untested.
    stale = out_dir / "model.int8.onnx"
    if stale.exists():
        stale.unlink()
    meta = {**meta, "outputs": list(s.head_names), "input_size": size}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"model.onnx": round((out_dir / "model.onnx").stat().st_size / 1e6, 2)}
