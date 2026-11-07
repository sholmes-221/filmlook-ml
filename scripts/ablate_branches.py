#!/usr/bin/env python
"""Ablation: full vs texture-only vs DINOv2-only heads (3 seeds). Writes reports/ablation_stage_a.json."""
import copy, json, logging, tempfile
import numpy as np, torch
from filmlook_ml import paths as P
from filmlook_ml.training.data import load_train_data
from filmlook_ml.training.train import train_stage_a
from filmlook_ml.models.filmlook_model import load_heads
from filmlook_ml.evaluation.metrics import evaluate
def main():
    cfg, lcfg = P.load_yaml("training.yaml"), P.load_yaml("labeling.yaml")
    data, man = load_train_data(cfg, lcfg)
    by_id = {r["image_id"]: r for r in man}
    ia, ib, *_ = data.pairs
    pm = (data.split[ia] == "test") & (data.split[ib] == "test")
    out = {}
    for name in ["full", "texture_only", "dinov2_only"]:
        for seed in (1337, 1, 2):
            d = copy.copy(data)
            if name == "texture_only": d.emb = np.zeros_like(data.emb)
            if name == "dinov2_only": d.feats = np.zeros_like(data.feats)
            with tempfile.TemporaryDirectory() as td:
                train_stage_a(d, cfg, td, seed, log=lambda *_: None)
                h, _ = load_heads(td); h.eval()
                with torch.no_grad(): o = h(torch.from_numpy(d.emb), torch.from_numpy(d.feats))
            pred = {"score": o["score"].numpy(), "attrs": o["attrs"].numpy(),
                    "ph": torch.sigmoid(o["ph_logits"]/h.t_ph).numpy(), "fs": torch.sigmoid(o["fs_logits"]/h.t_fs).numpy()}
            r = evaluate(pred, d.t, d.split == "test", tuple(x[pm] for x in d.pairs), by_id)
            out.setdefault(name, []).append([r["pairwise"]["acc"], r["film_score"]["mae"], r["film_score"]["pearson"],
                                             np.mean([v["mae"] for v in r["attributes"].values()]), r["film_stock"]["top1"], r["photographer"]["top1"]])
    print("variant        pair_acc  score_mae  score_r  attr_mae  fs_top1  ph_top1   (mean ± sd over 3 seeds)")
    for k, v in out.items():
        v = np.array(v); print(f"{k:14s}" + "".join(f"  {m:.3f}±{s:.3f}" for m, s in zip(v.mean(0), v.std(0))))
    json.dump({k: np.array(v).tolist() for k, v in out.items()}, open("reports/ablation_stage_a.json", "w"))


if __name__ == "__main__":
    main()
