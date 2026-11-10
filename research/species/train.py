"""Fine-tune an ImageNet-pretrained torchvision CNN on the species training split (head first, then last blocks).

  python -m research.species.train --arch mobilenet_v3_large [--epochs-head 3 --epochs-ft 12 --unfreeze last|all]

Writes dist/species/<run>.pt (state dict + label space), predictions for val / test / unknown splits to
data/species/preds/<run>__<split>.npz and a run record to data/species/runs.jsonl (GPU seconds, peak memory, val
accuracy per epoch). Selection: best val top-1. Mixed precision on CUDA.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as M

from research.species import common as C
from research.species import data as D

ARCHS = {
    "mobilenet_v3_small": (M.mobilenet_v3_small, M.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (M.mobilenet_v3_large, M.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (M.efficientnet_b0, M.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (M.efficientnet_b1, M.EfficientNet_B1_Weights.IMAGENET1K_V2),
}


def build(arch: str, n_classes: int) -> nn.Module:
    ctor, weights = ARCHS[arch]
    m = ctor(weights=weights)
    if arch.startswith("mobilenet"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def head_params(m: nn.Module):
    return list(m.classifier.parameters())


def set_trainable(m: nn.Module, mode: str, n_last: int) -> list[nn.Parameter]:
    for p in m.parameters():
        p.requires_grad = False
    params = head_params(m)
    if mode == "all":
        params = list(m.parameters())
    elif mode == "last":
        for blk in list(m.features)[-n_last:]:
            params += list(blk.parameters())
    for p in params:
        p.requires_grad = True
    return params


@torch.no_grad()
def predict(m: nn.Module, rows: list[dict], index: dict, dev, batch: int, workers: int):
    dl = DataLoader(D.ImageRows(rows, index, D.eval_transform(224)), batch_size=batch, num_workers=workers)
    m.eval()
    logits, idx, ys = [], [], []
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        for x, y, i in dl:
            logits.append(m(x.to(dev)).float().cpu()); idx.append(i); ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(idx).numpy(), torch.cat(ys).numpy()


def run_epochs(m, params, dl, dev, epochs, lr, val_rows, index, a, log, tag, best):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    for ep in range(epochs):
        m.train()
        # BatchNorm statistics stay frozen when the backbone is frozen
        if tag == "head":
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y, _ in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(m(x), y, label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        lg, _, ys = predict(m, val_rows, index, dev, a.batch, a.workers)
        top1 = float((lg.argmax(1) == ys).mean())
        top5 = float(np.mean([y in np.argsort(-l)[:5] for l, y in zip(lg, ys)]))
        rec = {"stage": tag, "epoch": ep + 1, "train_loss": tot / n, "val_top1": top1, "val_top5": top5, "seconds": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if top1 > best["top1"]:
            best.update(top1=top1, top5=top5, state={k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, stage=tag, epoch=ep + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large", choices=sorted(ARCHS))
    ap.add_argument("--run", default=None)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=12)
    ap.add_argument("--lr-head", type=float, default=2e-3)
    ap.add_argument("--lr-ft", type=float, default=4e-4)
    ap.add_argument("--unfreeze", default="last", choices=["last", "all", "none"])
    ap.add_argument("--n-last", type=int, default=4, help="number of trailing feature blocks unfrozen with --unfreeze last")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = a.run or f"{a.arch}_{a.unfreeze}"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rows = D.load_rows()
    space = D.label_space(rows)
    index = space["index"]
    by = {s: [r for r in rows if r["split"] == s] for s in D.SPLITS}
    train_rows = [r for r in by["train"] if r["species_id"] in index]
    val_rows = [r for r in by["val"] if r["species_id"] in index]
    dev = D.device()
    print(f"{run}: {len(index)} classes, train {len(train_rows)}, val {len(val_rows)}, test {len(by['test'])}, device {dev}")
    m = build(a.arch, len(index)).to(dev).to(memory_format=torch.channels_last)
    dl = DataLoader(D.ImageRows(train_rows, index, D.train_transform(224)), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    log, best = [], {"top1": -1.0}
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    params = set_trainable(m, "none", 0)
    run_epochs(m, params, dl, dev, a.epochs_head, a.lr_head, val_rows, index, a, log, "head", best)
    if a.unfreeze != "none" and a.epochs_ft > 0:
        params = set_trainable(m, a.unfreeze, a.n_last)
        run_epochs(m, params, dl, dev, a.epochs_ft, a.lr_ft, val_rows, index, a, log, f"ft_{a.unfreeze}", best)
    gpu_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0
    m.load_state_dict(best["state"])
    C.DIST.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": a.arch, "state_dict": best["state"], "classes": space["classes"], "input_size": 224, "mean": D.MEAN, "std": D.STD,
                "args": vars(a), "best": {k: v for k, v in best.items() if k != "state"}}, C.DIST / f"{run}.pt")
    for split in ("val", "test", "unknown_taxa", "unknown_nonwild"):
        if by[split]:
            lg, idx, _ = predict(m, by[split], index, dev, a.batch, a.workers)
            D.save_preds(run, split, idx, lg)
    n_params = sum(p.numel() for p in m.parameters())
    rec = {"run": run, "arch": a.arch, "args": vars(a), "n_classes": len(index), "n_train": len(train_rows), "n_val": len(val_rows),
           "best_val_top1": best["top1"], "best_val_top5": best["top5"], "best_stage": best["stage"], "best_epoch": best["epoch"],
           "gpu_seconds": gpu_s if dev.type == "cuda" else 0, "wall_seconds": gpu_s, "peak_gpu_mb": peak, "params": n_params, "log": log,
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    C.append_jsonl(C.DATA / "runs.jsonl", rec)
    C.append_jsonl(C.DATA / "gpu_ledger.jsonl", {"step": f"train_{run}", "gpu_seconds": rec["gpu_seconds"], "peak_mb": peak})
    print(f"done {run}: best val top1 {best['top1']:.4f} top5 {best['top5']:.4f} ({best['stage']} ep {best['epoch']}); {gpu_s:.0f} s, peak {peak:.0f} MB, {n_params / 1e6:.2f} M params")


if __name__ == "__main__":
    main()
