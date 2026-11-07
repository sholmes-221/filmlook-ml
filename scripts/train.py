#!/usr/bin/env python
"""Train a seed ensemble of heads (Stage A) for one model variant and save it as a checkpoint.

  python scripts/train.py --kind texture --seeds 0 1 2 --promote      # Model A (selected for v1)
  python scripts/train.py --kind fused --input-size 224 --seeds 0 1 2  # Model C
  python scripts/train.py --kind fused --stage-b                       # optional partial fine-tune check
Resume: re-run the same command with --resume (per-seed state in checkpoints/<name>/last_*.pt).
"""
import argparse
import json
import shutil
import time

import numpy as np

from filmlook_ml import paths as P
from filmlook_ml.constants import ATTRIBUTES, MODEL_VERSION, SCHEMA_VERSION
from filmlook_ml.models.filmlook_model import save_ensemble
from filmlook_ml.training.data import load_train_data
from filmlook_ml.training.train import gpu_free_gb, train_ensemble, train_stage_b, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["texture", "dino", "fused"], default="texture")
    ap.add_argument("--input-size", type=int, default=224, help="DINOv2 input (multiple of 14)")
    ap.add_argument("--feat-max-side", type=int, default=None, help="texture-statistics resolution (default config)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage-b", action="store_true")
    ap.add_argument("--promote", action="store_true", help="copy to checkpoints/best")
    a = ap.parse_args()
    cfg, lcfg = P.load_yaml("training.yaml"), P.load_yaml("labeling.yaml")
    free = gpu_free_gb()
    print(f"GPU free: {free:.1f} GB" if free is not None else "no GPU: training on CPU")
    if free is not None and free < cfg.get("min_free_gpu_gb", 2):
        raise SystemExit("not enough free GPU memory; other jobs are using the GPU")
    data, manifest = load_train_data(cfg, lcfg, with_emb=a.kind != "texture", emb_size=a.input_size,
                                     feat_max_side=a.feat_max_side)
    ms = a.feat_max_side or cfg["max_side"]
    name = a.name or (f"{a.kind}_{a.input_size}" if a.kind != "texture" else f"texture_ms{ms}")
    out = P.CHECKPOINTS / name
    print(f"images={len(data.ids)} train={int((data.split == 'train').sum())} val={int((data.split == 'val').sum())} "
          f"test={int((data.split == 'test').sum())} pairs={len(data.pairs[0])} "
          f"medium-labelled originals={int((data.t.medium_w > 0).sum())}")
    members = train_ensemble(data, a.kind, cfg, a.seeds, out, a.resume)
    src = {s: int((np.array(data.t.source) == s).sum()) for s in set(data.t.source)}
    meta = {"model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION, "kind": a.kind,
            "backbone": cfg["backbone"] if a.kind != "texture" else None,
            "input_size": a.input_size if a.kind != "texture" else None,
            "max_side": ms, "texture_crop": cfg["texture_crop"], "attributes": ATTRIBUTES, "seeds": a.seeds,
            "feat_mean": members[0].feat_mean.tolist(), "feat_std": members[0].feat_std.tolist(),
            "confidence": cfg.get("confidence", {"std_scale": 6.0, "ood_scale": 1.5}),
            "label_sources": src, "human_labels": src.get("human", 0),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_ensemble(out, members, meta)
    print(f"saved {len(members)}-member ensemble -> {out}")
    if a.stage_b:
        if a.kind == "texture":
            raise SystemExit("Stage B needs a DINOv2 variant")
        paths = {r["image_id"]: r["path"] for r in manifest}
        res = train_stage_b(data, paths, cfg, members[0], a.seeds[0], a.input_size)
        write_json(P.REPORTS / f"stage_b_{name}.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
    if a.promote:
        dst = P.CHECKPOINTS / "best"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst, ignore=shutil.ignore_patterns("last_*.pt", "*.onnx"))
        print(f"promoted {out.name} -> {dst}")


if __name__ == "__main__":
    main()
