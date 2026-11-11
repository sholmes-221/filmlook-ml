"""Re-export the production DINOv2-small ONNX with a second output, for the film head v2.

    python -m research.c08.export_layers_onnx   # -> checkpoints/c08/dinov2_small_cls_patch_layers.onnx + parity report

Output `features` [n, 768] is unchanged: block-12 [CLS, mean patch], each L2-normalised (what the photo gate, the
local vision heads, the reference index and film head v1 read). New output `layers` [n, 1536]: blocks 4 and 8,
[CLS, mean patch] each, each part L2-normalised. Same weights (facebook/dinov2-small), same input, one encoder pass.
The export is accepted only if `features` matches the current production file to 1e-4 on prepared photographs.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

WEB = Path.home() / "misc" / "film-evaluation"
CURRENT = WEB / "models/photo_gate/dinov2_small_cls_patch.onnx"
OUT = Path("checkpoints/c08/dinov2_small_cls_patch_layers.onnx")


class Graph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values, output_hidden_states=True)
        n = torch.nn.functional.normalize
        h12, h4, h8 = o.last_hidden_state, o.hidden_states[4], o.hidden_states[8]
        features = torch.cat([n(h12[:, 0], dim=1), n(h12[:, 1:].mean(1), dim=1)], 1)
        layers = torch.cat([n(h4[:, 0], dim=1), n(h4[:, 1:].mean(1), dim=1), n(h8[:, 0], dim=1), n(h8[:, 1:].mean(1), dim=1)], 1)
        return features, layers


def main() -> None:
    sys.path.insert(0, str(WEB))
    import onnxruntime as ort
    from PIL import Image

    from filmlook.image_utils import prepare_image
    from filmlook.photo_gate import dinov2_pixels
    from research.c08.film_head_v2 import parts_normalised
    g = Graph()
    torch.onnx.export(g, (torch.zeros(1, 3, 224, 224),), str(OUT), input_names=["pixel_values"], output_names=["features", "layers"],
                      dynamic_axes={"pixel_values": {0: "n"}, "features": {0: "n"}, "layers": {0: "n"}}, opset_version=17, dynamo=False)
    d = np.load("data/c08/encoders/prod_inputs.npz")
    paths = d["paths"][:: max(1, len(d["paths"]) // 60)][:60]
    old = ort.InferenceSession(str(CURRENT), providers=["CPUExecutionProvider"]); new = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    fd, ld = [], []
    ref_layers = parts_normalised(d["layers"][:: max(1, len(d["paths"]) // 60)][:60].astype(np.float32))
    for i, p in enumerate(paths):
        im = Image.open(io.BytesIO(prepare_image(Path(p).read_bytes()).data)); im.load(); x = dinov2_pixels(im)[None]
        a = old.run(None, {"pixel_values": x})[0]; f, l = new.run(None, {"pixel_values": x})
        fd.append(float(np.abs(a - f).max())); ld.append(float(np.abs(l[0] - ref_layers[i, :1536]).max()))
    rep = {"onnx_bytes": OUT.stat().st_size, "features_max_abs_diff_vs_production": max(fd), "layers_max_abs_diff_vs_research_inputs": max(ld), "n": len(paths)}
    Path("reports/c08").mkdir(parents=True, exist_ok=True); Path("reports/c08/onnx_export.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    if max(fd) > 1e-4:
        raise SystemExit("features output differs from production: not accepted")


if __name__ == "__main__":
    main()
