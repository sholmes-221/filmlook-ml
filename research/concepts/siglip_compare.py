"""Comparator: SigLIP-base zero-shot (google/siglip-base-patch16-224, Apache-2.0) on the same deduplicated val/test rows.

  python -m research.concepts.siglip_compare embed      -> data/concepts/feats_siglip.npz (val+test image embeddings)
  python -m research.concepts.siglip_compare eval       -> reports/concepts/siglip_results.json
  python -m research.concepts.siglip_compare footprint  -> ONNX fp32 + int8 image tower, CPU latency (1 thread), RSS

Text embeddings are computed once offline (one prompt per head, "a photo of ..."); score = sigmoid(scale * cos + bias)
as SigLIP is trained. Threshold chosen on val with the same rule as the linear heads; test read once.
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from research.concepts.common import DATA, REPORTS
from research.concepts.concept_map import CONCEPTS

MODEL = "google/siglip-base-patch16-224"
PROMPTS = {
    "night": "a photo taken at night", "blue_hour": "a photo taken at dusk, in the blue hour", "golden_hour": "a photo of a sunset or sunrise",
    "street": "a photo of a street", "bird": "a photo of a bird", "small_animal": "a photo of a cat, a dog or a squirrel",
    "school_uniform": "a photo of a school uniform", "gas_station": "a photo of a gas station", "car_window": "a photo of a car's rear-view mirror or windshield",
    "suburb": "a photo of a suburb", "industrial": "a photo of a warehouse, a parking lot or a factory", "pumpkin": "a photo of a pumpkin",
    "desert": "a photo of a desert", "power_plant": "a photo of a power station", "construction": "a photo of a construction site",
    "motorcycle": "a photo of a motorcycle", "starry": "a photo of a starry night sky", "water_lily": "a photo of water lilies",
    "sunflower": "a photo of a sunflower", "diner": "a photo of a diner or a convenience store", "wave": "a photo of ocean waves",
    "fog": "a photo of fog", "pink": "a pink photo", "corridor": "a photo of a corridor", "umbrella": "a photo of an umbrella",
    "scooter": "a photo of a scooter", "moon": "a photo of the moon", "fishing": "a photo of fishing", "maple": "a photo of maple leaves",
    "willow": "a photo of a willow tree", "dragonfly": "a photo of a dragonfly", "duck": "a photo of a duck", "boat": "a photo of a boat",
    "flowers": "a photo of flowers", "oranges": "a photo of oranges", "tunnel": "a photo of a tunnel", "dock": "a photo of a pier",
    "cabin": "a photo of a hut or a log cabin", "windmill": "a photo of a windmill", "wheat": "a photo of a wheat field", "ice": "a photo of ice",
    "fox": "a photo of a fox", "rickshaw": "a photo of a rickshaw", "lotus_pond": "a photo of a lotus pond", "spider_web": "a photo of a spider web",
    "beetle": "a photo of a beetle", "reflection": "a photo of a reflection",
}


def rows_valtest():
    from research.concepts.train import dedup, load
    rows, X = load()
    keep = dedup(rows, X)
    rows = rows[keep].reset_index(drop=True)
    return rows[rows.split.isin(["val", "test"])].reset_index(drop=True)


@torch.no_grad()
def embed() -> None:
    from transformers import AutoModel, AutoProcessor
    rows = rows_valtest()
    proc = AutoProcessor.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL).to("cuda").eval()
    out = []
    for b in range(0, len(rows), 128):
        ims = [Image.open(DATA / "images" / i[:2] / f"{i}.jpg").convert("RGB") for i in rows.image_id[b:b + 128]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda()
        v = m.get_image_features(pixel_values=px)
        v = v if torch.is_tensor(v) else v.pooler_output
        out.append(torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy())
    heads = list(CONCEPTS)
    tok = proc(text=[PROMPTS[h] for h in heads], padding="max_length", return_tensors="pt").to("cuda")
    t = m.get_text_features(**tok)
    t = t if torch.is_tensor(t) else t.pooler_output
    T = torch.nn.functional.normalize(t.float(), dim=1).cpu().numpy()
    np.savez(DATA / "feats_siglip.npz", ids=rows.image_id.to_numpy().astype(str), X=np.concatenate(out), T=T, heads=np.array(heads),
             scale=float(m.logit_scale.exp()), bias=float(m.logit_bias))
    print(f"embedded {len(rows)}; peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


def evaluate() -> None:
    from research.concepts.train import CONCEPTS as C2, metrics, threshold, SHIP_P, SHIP_LB, SHIP_R, SHIP_N
    f = np.load(DATA / "feats_siglip.npz")
    rows = rows_valtest().set_index("image_id").loc[list(f["ids"])].reset_index()
    S = 1 / (1 + np.exp(-(f["scale"] * f["X"] @ f["T"].T + f["bias"])))
    split, generic = rows.split.to_numpy(), rows.generic.to_numpy()
    res = {}
    for j, head in enumerate(f["heads"]):
        key = C2[head][0]
        y = rows[head].to_numpy().astype(int)
        for other, (k2, _) in C2.items():
            if other != head and k2 == key:
                y = np.where((rows[other].to_numpy() == 1) & (y == 0), -1, y)
        va, te = (split == "val") & (y >= 0), (split == "test") & (y >= 0)
        t = threshold(S[va, j], y[va])
        r = {"prompt": PROMPTS[head]}
        if t is None:
            r["decision"] = "unsupported"; r["reason"] = "no val threshold reaches precision 0.95"
        else:
            r["threshold"] = t; r["test"] = tm = metrics(S[te, j], y[te], generic[te], t)
            ok = tm["n_pos"] >= SHIP_N and tm["fires"] > 0 and tm["precision"] >= SHIP_P and tm["precision_wilson95"][0] >= SHIP_LB and tm["recall"] >= SHIP_R
            r["decision"] = "would_ship" if ok else "unsupported"
        r["test_ap"] = float(__import__("sklearn.metrics").metrics.average_precision_score(y[te] == 1, S[te, j])) if (y[te] == 1).any() else None
        res[str(head)] = r
        tm = r.get("test", {})
        print(f"{head:15s} AP={r['test_ap'] or 0:.3f} P={tm.get('precision', float('nan')):.3f} R={tm.get('recall', float('nan')):.3f} -> {r['decision']}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "siglip_results.json").write_text(json.dumps(res, indent=1))
    print("would ship", sum(r["decision"] == "would_ship" for r in res.values()))


def footprint() -> None:
    """Export the SigLIP image tower to ONNX (fp32, then dynamic int8), then measure each in a fresh process."""
    from transformers import SiglipVisionModel
    out = DATA / "siglip_onnx"; out.mkdir(exist_ok=True)
    fp32, int8 = out / "siglip_vision_fp32.onnx", out / "siglip_vision_int8.onnx"
    if not fp32.exists():
        m = SiglipVisionModel.from_pretrained(MODEL).eval()

        class Wrap(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x): return s.m(pixel_values=x).pooler_output
        torch.onnx.export(Wrap(m), torch.randn(1, 3, 224, 224), str(fp32), input_names=["pixel_values"], output_names=["emb"], opset_version=17, dynamo=False)
    if not int8.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    res = {}
    for name, path in [("siglip_fp32", fp32), ("siglip_int8", int8), ("dinov2s_photo_gate_fp32", "/home/shanmo/misc/film-evaluation/models/photo_gate/dinov2_small_cls_patch.onnx")]:
        code = f"""
import time, numpy as np, onnxruntime as ort
def st(k): return [int(l.split()[1]) for l in open('/proc/self/status') if l.startswith(k)][0]
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
r0=st('VmRSS:')
t=time.time(); s=ort.InferenceSession({str(path)!r}, so, providers=['CPUExecutionProvider']); load=time.time()-t
inp=s.get_inputs()[0].name; x=np.random.randn(1,3,224,224).astype(np.float32)
s.run(None,{{inp:x}}); ts=[]
for _ in range(20):
    t=time.time(); s.run(None,{{inp:x}}); ts.append(time.time()-t)
print(r0/1024, st('VmHWM:')/1024, load, np.median(ts)*1000, np.percentile(ts,95)*1000)
"""
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        import os
        res[name] = {"file_mb": os.path.getsize(path) / 2**20, "rss_after_imports_mb": float(o[0]), "peak_rss_mb": float(o[1]), "model_increment_mb": float(o[1]) - float(o[0]), "load_s": float(o[2]),
                     "p50_ms_1thread": float(o[3]), "p95_ms_1thread": float(o[4])}
        print(name, res[name])
    # cosine of int8 vs fp32 on real images
    prev = json.loads((REPORTS / "siglip_footprint.json").read_text()) if (REPORTS / "siglip_footprint.json").exists() else {}
    if not (DATA / "feats_dinov2s.npz").exists():
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "siglip_footprint.json").write_text(json.dumps({**prev, **res}, indent=1)); return
    import onnxruntime as ort
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    rows = rows_valtest().head(100)
    a = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]); b = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    cs = []
    for i in rows.image_id:
        px = proc(images=Image.open(DATA / "images" / i[:2] / f"{i}.jpg").convert("RGB"), return_tensors="np")["pixel_values"].astype(np.float32)
        u, v = a.run(None, {"pixel_values": px})[0][0], b.run(None, {"pixel_values": px})[0][0]
        cs.append(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    res["int8_vs_fp32_cosine"] = {"min": min(cs), "mean": float(np.mean(cs))}
    print(res["int8_vs_fp32_cosine"])
    (REPORTS / "siglip_footprint.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"embed": embed, "eval": evaluate, "footprint": footprint}[sys.argv[1]]()
