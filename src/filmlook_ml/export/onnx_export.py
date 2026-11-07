"""ONNX export (fp32) + dynamic INT8 quantisation + parity check for ensemble checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from filmlook_ml.labeling.features import N_FEATURES
from filmlook_ml.models.filmlook_model import OUTPUT_NAMES, DinoEmbedder, ExportModel, load_ensemble


def _export_model(ckpt_dir: Path) -> tuple[ExportModel, dict]:
    ens, meta = load_ensemble(ckpt_dir)
    emb = DinoEmbedder(meta["backbone"]).eval() if meta["kind"] in ("dino", "fused") else None
    return ExportModel(ens, emb).eval(), meta


def export_onnx(ckpt_dir: Path, opset: int = 18) -> Path:
    ckpt_dir = Path(ckpt_dir)
    model, meta = _export_model(ckpt_dir)
    ft = torch.as_tensor(np.array(meta["feat_mean"], np.float32)[None])
    out = ckpt_dir / "model.onnx"
    if model.embedder is not None:
        s = meta["input_size"]
        args, names = (ft, torch.randn(1, 3, s, s)), ["feats", "pixel_values"]
    else:
        args, names = (ft,), ["feats"]
    torch.onnx.export(model, args, str(out), input_names=names, output_names=OUTPUT_NAMES,
                      opset_version=opset, dynamo=True, external_data=False)  # static batch=1 (deployment)
    return out


def quantize_int8(fp32_path: Path) -> Path:
    """Dynamic INT8 (int8 weights, activations quantised on the fly) for MatMul/Gemm, per-channel.

    The dynamo exporter writes intermediate value_info that ORT's quantiser re-infers inconsistently,
    so it is dropped (shapes are re-inferred) before quantising.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = Path(fp32_path)
    clean = fp32_path.with_name("model.clean.onnx")
    out = fp32_path.with_name("model.int8.onnx")
    m = onnx.load(str(fp32_path))
    del m.graph.value_info[:]
    onnx.save(m, str(clean))
    try:
        quantize_dynamic(str(clean), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                         per_channel=True)
    finally:
        clean.unlink(missing_ok=True)
    return out


def parity(ckpt_dir: Path, onnx_path: Path, n: int = 8, seed: int = 0, images: list | None = None) -> dict:
    """Max/mean abs diff between the torch checkpoint and an ONNX file on random or real inputs."""
    import onnxruntime as ort

    from filmlook_infer.features import compute_features, feature_vector
    from filmlook_infer.imageio import load_image
    from filmlook_infer.pixels import dinov2_pixels

    model, meta = _export_model(ckpt_dir)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    s = meta.get("input_size") or 224
    mu, sd = np.array(meta["feat_mean"], np.float32), np.array(meta["feat_std"], np.float32)
    rng = np.random.default_rng(seed)
    inputs = []
    if images:
        for p in images[:n]:
            im = load_image(p, max_side=meta["max_side"])
            inputs.append((feature_vector(compute_features(im, meta["texture_crop"]))[None], dinov2_pixels(im, s)[None]))
    else:
        for _ in range(n):
            inputs.append(((mu + sd * rng.standard_normal(N_FEATURES)).astype(np.float32)[None],
                           rng.standard_normal((1, 3, s, s)).astype(np.float32)))
    diffs = {k: [] for k in OUTPUT_NAMES}
    for ft, px in inputs:
        with torch.no_grad():
            ref = [o.numpy() for o in (model(torch.from_numpy(ft), torch.from_numpy(px)) if model.embedder is not None
                                       else model(torch.from_numpy(ft)))]
        feed = {"feats": ft, **({"pixel_values": px} if "pixel_values" in in_names else {})}
        for k, a, b in zip(OUTPUT_NAMES, ref, sess.run(None, feed)):
            diffs[k].append(float(np.abs(a - b).max()))
    return {k: {"max_abs": max(v), "mean_abs": float(np.mean(v))} for k, v in diffs.items()}


def write_export_report(out_dir: Path, info: dict) -> None:
    (Path(out_dir) / "export_report.json").write_text(json.dumps(info, indent=2))
