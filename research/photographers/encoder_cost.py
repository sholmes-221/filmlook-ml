"""C19 production cost of a candidate image encoder: ONNX export (fp32 + int8 dynamic), parity, bank vectors, CPU cost.

    python -m research.photographers.encoder_cost export  --model clip_b32      # checkpoints/c19/<model>{,_int8}.onnx + parity
    python -m research.photographers.encoder_cost embed   --model clip_b32 --variant int8   # data/embeddings/photographers/<model>_onnx_int8.npz
    python -m research.photographers.encoder_cost measure --model clip_b32 --variant int8   # one fresh process: size, RSS, latency
    python -m research.photographers.encoder_cost ort     --model clip_b32 --variant int8   # ORT flatbuffer (in-place weights)
    python -m research.photographers.encoder_cost measure --model clip_b32 --variant int8 --lean --format ort
    python -m research.photographers.encoder_cost measure --model prod                       # the shipped DINOv2-small, for scale

The ONNX graph is the image tower only, output = L2-normalised image embedding (CLIP: image_embeds after the visual
projection; SigLIP: the attention-pooled pooler_output), i.e. exactly the vectors `embed.py` stored. Preprocessing is
re-implemented in numpy/PIL (what the website would run, no torch); parity is measured end to end against the stored
benchmark vectors (Hugging Face processor + PyTorch model). Cost: CPU, onnxruntime intra_op_num_threads=1,
inter_op_num_threads=1, 224 x 224 input; peak RSS increase over a process that already imported numpy, PIL and
onnxruntime; latency = decode + preprocessing + inference per image, median over 30 bank images after 3 warm-ups.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CK = ROOT / "checkpoints" / "c19"
EMB = ROOT / "data" / "embeddings" / "photographers"
REP = ROOT / "reports" / "photographers"
HF = {"siglip_base": "google/siglip-base-patch16-224", "clip_b32": "openai/clip-vit-base-patch32"}
PRE = {"clip_b32": dict(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), crop=True),
       "siglip_base": dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop=False)}


def pixels(img: Image.Image, model: str, size: int = 224) -> np.ndarray:
    """CLIP: shortest side to 224 (bicubic), centre crop 224. SigLIP: resize to 224 x 224 (bicubic), no crop."""
    if model == "prod":
        from filmlook_infer.pixels import dinov2_pixels
        return dinov2_pixels(img)
    p = PRE[model]; img = img.convert("RGB"); w, h = img.size
    if p["crop"]:
        s = size / min(w, h)
        img = img.resize((max(size, int(w * s)) if w >= h else size, size if w >= h else max(size, int(h * s))), Image.BICUBIC)
        w, h = img.size; left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    else:
        img = img.resize((size, size), Image.BICUBIC)
    a = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(p["mean"], np.float32)) / np.array(p["std"], np.float32)
    return a.transpose(2, 0, 1).copy()


def to_ort(src: Path, dst: Path | None = None) -> Path:
    """ONNX -> ORT flatbuffer (basic graph optimisations baked in). Loaded from bytes with
    session.use_ort_model_bytes_directly / use_ort_model_bytes_for_initializers, the weights are used in place
    instead of being copied out of a parsed protobuf, which removes the load-time double copy."""
    import onnxruntime as ort
    dst = dst or src.with_suffix(".ort")
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst); so.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    return dst


def to_external(src: Path, align: int = 65536) -> Path:
    """Same graph, every initializer >= 64 KB moved to `<name>.weights` at 64 KB-aligned offsets, so onnxruntime can
    memory-map the weights (file-backed, read-only pages) instead of copying them out of a parsed protobuf."""
    import onnx
    from onnx.external_data_helper import set_external_data
    m = onnx.load(str(src)); dst = src.with_name(src.stem + "_ext.onnx"); wpath = src.with_name(src.stem + "_ext.weights")
    off = 0
    with open(wpath, "wb") as f:
        for t in m.graph.initializer:
            if len(t.raw_data) < 65536:
                continue
            pad = (-off) % align; f.write(b"\0" * pad); off += pad
            data = t.raw_data; f.write(data)
            set_external_data(t, location=wpath.name, offset=off, length=len(data)); t.ClearField("raw_data"); t.data_location = onnx.TensorProto.EXTERNAL
            off += len(data)
    onnx.save_model(m, str(dst))
    return dst


def onnx_path(model: str, variant: str) -> Path:
    if model == "prod":
        return ROOT / "checkpoints" / "refdb" / "dinov2_small_cls_patch.onnx"
    return CK / f"{model}{'' if variant == 'fp32' else '_int8'}.onnx"


def session(path: Path, threads: int, lean: bool = False):
    """lean: no CPU memory arena, no weight pre-packing (ORT otherwise keeps a second, re-laid-out copy of the
    MatMul weights), no memory-pattern planning. Same outputs, less resident memory."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    if lean:
        so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
        so.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bank_ids() -> tuple[list[str], np.ndarray, dict]:
    from research.photographers.evaluate import load_rows
    ids = [r["work_id"] for r in load_rows()]
    return ids


def export(model: str) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import CLIPVisionModelWithProjection, SiglipVisionModel
    torch.set_num_threads(4)
    CK.mkdir(parents=True, exist_ok=True)
    tower = (CLIPVisionModelWithProjection if model == "clip_b32" else SiglipVisionModel).from_pretrained(HF[model]).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            v = o.image_embeds if model == "clip_b32" else o.pooler_output
            return torch.nn.functional.normalize(v, dim=1)

    fp32 = onnx_path(model, "fp32")
    torch.onnx.export(Wrap(tower), (torch.zeros(1, 3, 224, 224),), str(fp32), input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(onnx_path(model, "int8")), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])  # MatMul only: patch conv stays fp32 (parity 0.989 vs 0.971 with Conv)
    # parity on 300 bank works against the stored benchmark vectors (HF processor + PyTorch)
    from filmlook_ml.refdb import manifest
    ids = bank_ids()[::15][:300]
    d = np.load(EMB / f"{model}.npz"); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
    ref = d["X"][[pos[w] for w in ids]]
    out = {"model": model, "hf_id": HF[model], "n": len(ids), "files_bytes": {}}
    for variant in ("fp32", "int8"):
        s = session(onnx_path(model, variant), 4)
        V = np.concatenate([s.run(None, {"pixel_values": pixels(Image.open(manifest.image_path(w)), model)[None]})[0] for w in ids])
        cos = (V * ref).sum(1) / np.linalg.norm(V, axis=1) / np.linalg.norm(ref, axis=1)
        out[variant] = {"cosine_vs_pytorch_mean": float(cos.mean()), "p01": float(np.percentile(cos, 1)), "min": float(cos.min())}
        out["files_bytes"][variant] = onnx_path(model, variant).stat().st_size
        print(model, variant, out[variant], out["files_bytes"][variant], flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REP / f"encoder_parity_{model}.json", "w"), indent=1)


def embed(model: str, variant: str) -> None:
    from filmlook_ml.refdb import manifest
    ids = bank_ids(); s = session(onnx_path(model, variant), 4)
    X = np.concatenate([s.run(None, {"pixel_values": pixels(Image.open(manifest.image_path(w)), model)[None]})[0] for w in ids])
    np.savez(EMB / f"{model}_onnx_{variant}.npz", work_ids=np.array(ids), X=X.astype(np.float32))
    print("saved", X.shape)


def measure(model: str, variant: str, lean: bool = False, fmt: str = "onnx") -> None:
    import onnxruntime  # noqa: F401  (baseline: the website process has it loaded already)
    import psutil
    from filmlook_ml.refdb import manifest
    proc = psutil.Process()
    ids = bank_ids()[::150][:33]
    jpegs = [open(manifest.image_path(w), "rb").read() for w in ids]
    path = onnx_path(model, variant)
    if fmt == "ext":
        path = path.with_name(path.stem + "_ext.onnx") if model != "prod" else CK / "dinov2_small_cls_patch_ext.onnx"
    if fmt == "ort":
        path = path.with_suffix(".ort") if model != "prod" else CK / "dinov2_small_cls_patch.ort"
        if not path.exists():
            raise SystemExit(f"{path} missing: run the `ort` command first")
    rss0, peak0 = proc.memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t0 = time.perf_counter(); s = session(path, 1, lean); t_load = time.perf_counter() - t0
    rss_loaded = proc.memory_info().rss
    lat, inf = [], []
    for k, j in enumerate(jpegs):
        t0 = time.perf_counter()
        x = pixels(Image.open(io.BytesIO(j)), model)[None]
        t1 = time.perf_counter(); s.run(None, {"pixel_values": x}); t2 = time.perf_counter()
        if k >= 3:
            lat.append(t2 - t0); inf.append(t2 - t1)
    rss1, peak1 = proc.memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    out = {"model": model, "variant": variant, "format": fmt, "file_mb": round(sum(p.stat().st_size for p in path.parent.glob(path.stem + ".*")) / 2**20, 1) if fmt == "ext" else round(path.stat().st_size / 2**20, 1),
           "session_load_s": round(t_load, 3), "rss_after_load_extra_mb": round((rss_loaded - rss0) / 2**20, 1),
           "rss_after_inference_extra_mb": round((rss1 - rss0) / 2**20, 1), "peak_rss_extra_mb": round((peak1 - max(peak0, rss0)) / 2**20, 1),
           "latency_median_s": round(float(np.median(lat)), 4), "latency_p90_s": round(float(np.percentile(lat, 90)), 4),
           "inference_median_s": round(float(np.median(inf)), 4), "n_images": len(lat), "threads": 1, "lean_session": lean, "input": "1x3x224x224",
           "cpu": os.popen("lscpu | grep 'Model name'").read().split(":")[-1].strip()}
    print(json.dumps(out))
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REP / f"encoder_cost_{model}_{variant}{'_lean' if lean else ''}{'' if fmt == 'onnx' else '_' + fmt}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["export", "embed", "measure", "ort", "ext"])
    ap.add_argument("--model", required=True); ap.add_argument("--variant", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--format", default="onnx", choices=["onnx", "ort", "ext"])
    ap.add_argument("--lean", action="store_true", help="measure with the lean session options")
    a = ap.parse_args()
    {"export": lambda: export(a.model), "embed": lambda: embed(a.model, a.variant), "measure": lambda: measure(a.model, a.variant, a.lean, a.format),
     "ext": lambda: print(to_external(onnx_path(a.model, a.variant)) if a.model != "prod" else None),
     "ort": lambda: print(to_ort(onnx_path(a.model, a.variant)) if a.model != "prod" else
                          to_ort(onnx_path("prod", "fp32"), CK / "dinov2_small_cls_patch.ort"))}[a.cmd]()
