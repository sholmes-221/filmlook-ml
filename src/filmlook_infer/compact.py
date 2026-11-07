"""Compact ONNX inference for tightly constrained hosts (e.g. Oracle VM.Standard.E2.1.Micro: 1/8 OCPU, 1 GB RAM).

Dependencies: numpy, Pillow, onnxruntime only. The same ``preprocess`` is used at training time, so there is no
train/serve skew. Memory is bounded on purpose:
- JPEG DCT draft decoding: a 50 MP JPEG is decoded at 1/2-1/8 scale instead of being fully materialised;
- a pixel-count guard;
- one thread and a single session, with the memory arena disabled.

Output: structured classifications only (label + uncalibrated scores per head). No text generation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_PIXELS = 60_000_000


def preprocess(src, size: int = 224, resize: int | None = None) -> np.ndarray:
    """Path, file object or PIL image -> (1, 3, size, size) float32. Short side to ``resize``, centre crop."""
    resize = resize or round(size * 256 / 224)
    im = src if isinstance(src, Image.Image) else Image.open(src)
    if im.width * im.height > MAX_PIXELS:
        raise ValueError("image too large")
    if im.format == "JPEG":
        im.draft("RGB", (resize * 2, resize * 2))     # decode at reduced scale: bounded memory and time
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = resize / min(im.size)
    im = im.resize((max(resize, round(im.width * s)), max(resize, round(im.height * s))), Image.BILINEAR)
    left, top = (im.width - size) // 2, (im.height - size) // 2
    im = im.crop((left, top, left + size, top + size))
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


class CompactAnalyzer:
    """Loads ``model.onnx`` (or ``model.int8.onnx``) plus ``meta.json`` (head names and classes)."""

    def __init__(self, model_dir: str | Path, int8: bool = True, threads: int = 1):
        import onnxruntime as ort
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        path = d / ("model.int8.onnx" if int8 and (d / "model.int8.onnx").exists() else "model.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.model_file = path.name
        self.size = int(self.meta.get("input_size", 224))

    def analyze(self, src) -> dict:
        t0 = time.perf_counter()
        x = preprocess(src, self.size)
        t1 = time.perf_counter()
        outs = self.session.run(None, {"pixel_values": x})
        t2 = time.perf_counter()
        res = {}
        for name, logits in zip(self.meta["outputs"], outs):
            classes = self.meta["heads"][name]
            z = logits[0] - logits[0].max()
            p = np.exp(z) / np.exp(z).sum()
            order = np.argsort(-p)[:3]
            res[name] = {"label": classes[order[0]], "scores": {classes[i]: round(float(p[i]), 4) for i in order}}
        return {"model_version": self.meta["model_version"], "visual_attributes": res,
                "timing_ms": {"preprocess": round(1000 * (t1 - t0), 1), "inference": round(1000 * (t2 - t1), 1)},
                "note": "uncalibrated scores; resemblance and attributes only"}
