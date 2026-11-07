#!/usr/bin/env python
"""Export DINOv2-small + baseline heads as ONE ONNX graph (fp32 and dynamic INT8) for constrained-CPU measurement.

  python scripts/export_dinov2_baseline.py   # -> checkpoints/dinov2_baseline_v0/onnx/{model.onnx,model.int8.onnx,meta.json}

The graph takes the same 224 px input as filmlook_infer.compact.preprocess. Embeddings in the baseline were computed
with filmlook_infer.pixels (short side 256, centre crop 224, ImageNet normalisation); preprocess has the same geometry
and normalisation, but uses a different resampler, so parity is measured rather than assumed.
"""
import json

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from filmlook_infer.compact import preprocess
from filmlook_ml import paths as P
from filmlook_ml.models.filmlook_model import DinoEmbedder

CK = P.CHECKPOINTS / "dinov2_baseline_v0"


class Graph(nn.Module):
    def __init__(self, z, meta):
        super().__init__()
        self.emb = DinoEmbedder()
        self.register_buffer("mu", torch.tensor(z["embedding_mean"], dtype=torch.float32))
        self.names = list(meta["heads"])
        for n in self.names:
            W, b = z[f"{n}.coef"], z[f"{n}.intercept"]
            if W.shape[0] == 1:                         # binary: two logits [0, w.x+b]
                W, b = np.vstack([np.zeros_like(W), W]), np.array([0.0, b[0]])
            self.register_buffer(f"{n}_W", torch.tensor(W, dtype=torch.float32))
            self.register_buffer(f"{n}_b", torch.tensor(b, dtype=torch.float32))
            self.register_buffer(f"{n}_m", torch.tensor(z[f"{n}.scale_mean"], dtype=torch.float32))
            self.register_buffer(f"{n}_s", torch.tensor(z[f"{n}.scale_std"], dtype=torch.float32))

    def forward(self, pixel_values):
        e = self.emb(pixel_values) - self.mu
        e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        return tuple(((e - getattr(self, f"{n}_m")) / getattr(self, f"{n}_s")) @ getattr(self, f"{n}_W").T
                     + getattr(self, f"{n}_b") for n in self.names)


def main():
    meta = json.loads((CK / "meta.json").read_text())
    z = np.load(CK / "heads.npz")
    g = Graph(z, meta).eval()
    out = CK / "onnx"
    out.mkdir(exist_ok=True)
    torch.onnx.export(g, (torch.zeros(1, 3, 224, 224),), str(out / "model.onnx"), input_names=["pixel_values"],
                      output_names=g.names, dynamic_axes={"pixel_values": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
    (out / "meta.json").write_text(json.dumps({"model_version": meta["baseline_version"] + "-onnx", "outputs": g.names,
                                               "heads": {n: meta["heads"][n]["classes"] for n in g.names},
                                               "input_size": 224}, indent=1))
    man = [json.loads(x) for x in open(P.MANIFESTS / "manifest.jsonl")]
    xs = np.concatenate([preprocess(r["path"]) for r in man if r["edit_type"] == "original" and r["split"] == "test"][:48])
    with torch.no_grad():
        ref = [t.numpy().argmax(1) for t in g(torch.from_numpy(xs))]
    rep = {}
    for f in ("model.onnx", "model.int8.onnx"):
        s = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        o = s.run(None, {"pixel_values": xs})
        rep[f] = {"mb": round((out / f).stat().st_size / 1e6, 1),
                  "top1_agreement_with_pytorch": {n: round(float((o[i].argmax(1) == ref[i]).mean()), 3)
                                                   for i, n in enumerate(g.names)}}
    print(json.dumps(rep, indent=1))
    (P.REPORTS / "dinov2_onnx_export.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
