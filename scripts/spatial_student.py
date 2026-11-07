#!/usr/bin/env python
"""E12: a MobileNetV3-Small spatial student trained on exact crop geometry (zoom, signed x/y offset).

  python scripts/spatial_student.py     # -> reports/spatial_student.json, checkpoints/spatial_student_v0/

Supervision is synthetic but exact: for each real source (never a filmsynth source), crops with known zoom in [1, 2]
and offsets in [-1, 1] are generated; the student sees the crop and predicts the three numbers. Train/test split by
source (the existing uploader-grouped split). Heads: a 2x2 grid of the 7x7 feature map (2304-d) and, as an ablation,
global pooling only (576-d). Metrics on held-out sources: R^2 per target, mirror sensitivity ratio of the grid
embedding, ONNX size and CPU latency. Baseline: the frozen DINOv2 2x2 grid probe (reports/spatial_probe.json).
This measures framing recoverability, a proxy; it is not human composition judgement.
"""
import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402

from filmlook_infer.compact import preprocess  # noqa: E402
from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

N_SOURCES, CROPS_TRAIN, CROPS_TEST = 400, 8, 4


class SpatialStudent(nn.Module):
    def __init__(self, grid: bool = True):
        super().__init__()
        import torchvision
        self.f = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").features
        self.grid = grid
        d = 576 * (4 if grid else 1)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, 3))

    def embed(self, x):
        fm = self.f(x)
        # 7x7 map -> 2x2 grid with a fixed 4x4 window, stride 3 (ONNX-exportable; adaptive pooling 7->2 is not)
        return nn.functional.avg_pool2d(fm, kernel_size=4, stride=3).flatten(1) if self.grid else fm.mean((2, 3))

    def forward(self, x):
        return self.head(self.embed(x))


def make_crops(im, n, rng):
    out = []
    for _ in range(n):
        z, ox, oy = float(rng.uniform(1.0, 2.0)), float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        w, h = im.width / z, im.height / z
        cx, cy = im.width / 2 + ox * (im.width - w) / 2, im.height / 2 + oy * (im.height - h) / 2
        box = tuple(int(round(v)) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        out.append((im.crop(box), (z, ox, oy)))
    return out


def r2(p, t):
    return round(float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4)


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(0); rng = np.random.default_rng(0)
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    used = {a["source_image_id"] for a in read_jsonl(P.DATA / "synthetic" / "film_synth_v1" / "annotations.jsonl")}
    srcs = [r for r in eligible if r["image_id"] not in used and r["split"] in ("train", "test")][:N_SOURCES]
    Xtr, Ytr, Xte, Yte, Bte = [], [], [], [], []
    for r in srcs:
        im = X.to_short(load_image(r["path"], max_side=1280), 512)
        if r["split"] == "train":
            for c, y in make_crops(im, CROPS_TRAIN, rng):
                Xtr.append(preprocess(c)); Ytr.append(y)
        else:
            for c, y in make_crops(im, CROPS_TEST, rng):
                Xte.append(preprocess(c)); Yte.append(y)
            Bte.append(X.to_short(im))
    Xtr, Xte = np.concatenate(Xtr), np.concatenate(Xte); Ytr, Yte = np.array(Ytr, np.float32), np.array(Yte, np.float32)
    mu, sd = Ytr.mean(0), Ytr.std(0)
    rep = {"n_sources": len(srcs), "n_train_crops": int(len(Xtr)), "n_test_sources": sum(r["split"] == "test" for r in srcs), "n_test_crops": int(len(Xte)), "variants": {}}
    torch.cuda.reset_peak_memory_stats()
    for name, grid in (("grid2x2", True), ("global", False)):
        net = SpatialStudent(grid).to(dev)
        opt = torch.optim.AdamW([{"params": net.f.parameters(), "lr": 3e-4}, {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        Yn = torch.tensor((Ytr - mu) / sd, device=dev)
        t1 = time.time()
        for ep in range(20):
            net.train(); perm = rng.permutation(len(Xtr))
            for b in range(0, len(perm), 64):
                bi = perm[b:b + 64]
                x = torch.from_numpy(Xtr[bi]).to(dev)
                loss = nn.functional.mse_loss(net(x), Yn[torch.as_tensor(bi, device=dev)])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = np.concatenate([net(torch.from_numpy(Xte[i:i + 128]).to(dev)).cpu().numpy() for i in range(0, len(Xte), 128)]) * sd + mu
            A = np.concatenate([net.embed(torch.from_numpy(np.concatenate([preprocess(b) for b in Bte[i:i + 64]])).to(dev)).cpu().numpy() for i in range(0, len(Bte), 64)])
            M = np.concatenate([net.embed(torch.from_numpy(np.concatenate([preprocess(b.transpose(Image.FLIP_LEFT_RIGHT)) for b in Bte[i:i + 64]])).to(dev)).cpu().numpy() for i in range(0, len(Bte), 64)])
        An = A / np.linalg.norm(A, axis=1, keepdims=True); Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        d_flip = 1 - (An * Mn).sum(1); S = An @ An.T; np.fill_diagonal(S, np.nan); d_other = 1 - np.nanmedian(S, 1)
        rep["variants"][name] = {"train_seconds": round(time.time() - t1, 1),
                                 "test_r2": {"zoom": r2(pred[:, 0], Yte[:, 0]), "x_offset": r2(pred[:, 1], Yte[:, 1]), "y_offset": r2(pred[:, 2], Yte[:, 2])},
                                 "test_mae": {"zoom": round(float(np.abs(pred[:, 0] - Yte[:, 0]).mean()), 4), "x_offset": round(float(np.abs(pred[:, 1] - Yte[:, 1]).mean()), 4), "y_offset": round(float(np.abs(pred[:, 2] - Yte[:, 2]).mean()), 4)},
                                 "mirror_sensitivity_ratio": round(float(np.median(d_flip / (d_other + 1e-8))), 4)}
        if grid:
            out = P.CHECKPOINTS / "spatial_student_v0"; out.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(net.cpu().eval(), (torch.zeros(1, 3, 224, 224),), str(out / "model.onnx"), input_names=["pixel_values"], output_names=["zoom_x_y"],
                              dynamic_axes={"pixel_values": {0: "batch"}}, opset_version=17, dynamo=False)
            (out / "meta.json").write_text(json.dumps({"model_version": "filmlook-spatial-student-v0", "outputs": ["zoom_x_y"], "target_mean": mu.tolist(), "target_std": sd.tolist(), "input_size": 224,
                                                       "labels": "exact synthetic crop geometry on real photos; framing proxy, not composition judgement"}, indent=1))
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(out / "model.onnx"), so, providers=["CPUExecutionProvider"]); x1 = Xte[:1]; sess.run(None, {"pixel_values": x1})
            ts = []
            for _ in range(30):
                tt = time.perf_counter(); sess.run(None, {"pixel_values": x1}); ts.append(time.perf_counter() - tt)
            rep["variants"][name]["cost"] = {"onnx_mb": round((out / "model.onnx").stat().st_size / 1e6, 2), "cpu_1thread_ms_p50": round(1000 * float(np.median(ts)), 1)}
    rep["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rep["baseline_frozen_dino_grid2_probe"] = {"x_offset": 0.2371, "y_offset": 0.2275, "zoom": -0.3466, "mirror": 0.1752, "source": "reports/spatial_probe.json"}
    rep["seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / "spatial_student.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
