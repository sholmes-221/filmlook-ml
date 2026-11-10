"""Export a trained classifier to ONNX (fp32) and int8 (static QDQ with calibration images, plus dynamic for comparison).

  python -m research.species.export --run mobilenet_v3_large_last [--calib 256]
  python -m research.species.export --run dinov2s_linear                   # frozen DINOv2-small + linear probe head

Outputs in dist/species/: <run>.onnx, <run>-int8.onnx (static), <run>-int8dyn.onnx (dynamic), labels.json
(class order, GBIF keys, genus / family, preprocessing spec). Nothing here is committed.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quant_pre_process, quantize_dynamic, quantize_static

from research.species import common as C
from research.species import data as D
from research.species import train as TR


class DinoLinear(nn.Module):
    def __init__(self, w: np.ndarray, b: np.ndarray, mu: np.ndarray, feat: str):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        self.feat = feat
        self.register_buffer("mu", torch.tensor(mu))
        self.head = nn.Linear(w.shape[1], w.shape[0])
        self.head.weight.data = torch.tensor(w); self.head.bias.data = torch.tensor(b)

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        cls, pm = h[:, 0], h[:, 1:].mean(1)
        f = {"cls": cls, "patch_mean": pm, "concat": torch.cat([cls, pm], 1)}[self.feat]
        f = f - self.mu
        f = f / torch.clamp(f.norm(dim=1, keepdim=True), min=1e-8)
        return self.head(f)


def load_model(run: str):
    if run == "dinov2s_linear":
        z = np.load(C.DIST / "dinov2s_linear.npz")
        rows = D.load_rows()
        classes = D.label_space(rows)["classes"]
        return DinoLinear(z["weight"], z["bias"], z["mean"], str(z["feat"])), classes, "facebook/dinov2-small (frozen) + linear probe"
    ck = torch.load(C.DIST / f"{run}.pt", map_location="cpu", weights_only=False)
    m = TR.build(ck["arch"], len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["classes"], ck["arch"]


class Reader(CalibrationDataReader):
    def __init__(self, rows, n):
        tf = D.eval_transform(224)
        self.items = iter([{"input": tf(__import__("PIL.Image", fromlist=["Image"]).open(D.abs_path(r)).convert("RGB")).unsqueeze(0).numpy()} for r in rows[:n]])

    def get_next(self):
        return next(self.items, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    m, classes, arch = load_model(a.run)
    C.DIST.mkdir(parents=True, exist_ok=True)
    fp32 = C.DIST / f"{a.run}.onnx"
    x = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(m, x, str(fp32), input_names=["input"], output_names=["logits"], opset_version=a.opset, dynamo=False,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(str(fp32))
    pre = C.DIST / f"{a.run}-pre.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    rows = D.load_rows()
    rng = np.random.default_rng(0)
    train_rows = [r for r in rows if r["split"] == "train"]
    calib = [train_rows[i] for i in rng.choice(len(train_rows), min(a.calib, len(train_rows)), replace=False)]
    t0 = time.time()
    int8 = C.DIST / f"{a.run}-int8.onnx"
    quantize_static(str(pre), str(int8), Reader(calib, a.calib), quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
    dyn = C.DIST / f"{a.run}-int8dyn.onnx"
    quantize_dynamic(str(pre), str(dyn), weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    labels = {"run": a.run, "arch": arch, "classes": classes,
              "preprocessing": {"resize_short_side": 256, "center_crop": 224, "color": "RGB", "scale": "x/255", "mean": D.MEAN, "std": D.STD,
                                "layout": "NCHW float32", "input_name": "input", "output_name": "logits", "output": "unnormalised logits; apply softmax"},
              "hierarchy": {"genus": sorted({c["genus"] for c in classes}), "family": sorted({c["family"] for c in classes})},
              "exported": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(labels, open(C.DIST / f"{a.run}.labels.json", "w"), indent=1, ensure_ascii=False)
    sizes = {p.name: p.stat().st_size for p in (fp32, int8, dyn)}
    print(json.dumps({"sizes_bytes": sizes, "quant_seconds": time.time() - t0, "calib_images": len(calib)}, indent=1))


if __name__ == "__main__":
    main()
