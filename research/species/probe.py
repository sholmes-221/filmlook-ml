"""Frozen DINOv2-small: cosine kNN retrieval and a linear probe on the cached embeddings.

  python -m research.species.probe [--feat cls|patch_mean|concat] [--k 10]

Writes predictions (data/species/preds/dinov2s_knn__<split>.npz with a kNN "logit" = class vote mass, plus
`knn_dist` = cosine distance to the nearest training neighbour for rejection; dinov2s_linear__<split>.npz) and
dist/species/dinov2s_linear.npz (the probe weights for an ONNX head). CPU only; a few seconds.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from research.species import common as C
from research.species import data as D


def l2n(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def knn_scores(q, bank, bank_y, n_classes, k):
    sims = q @ bank.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    out = np.zeros((len(q), n_classes), np.float32)
    for i in range(len(q)):
        s, ys = sims[i, top[i]], bank_y[top[i]]
        for sv, y in zip(s, ys):
            out[i, y] += max(sv, 0.0)
    out /= np.clip(out.sum(1, keepdims=True), 1e-8, None)
    return out, 1.0 - sims.max(1)


def train_linear(x, y, n_classes, xv, yv, epochs=300, wd=1e-3, lr=1e-2):
    xt, yt = torch.tensor(x), torch.tensor(y)
    xvt, yvt = torch.tensor(xv), torch.tensor(yv)
    w = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = (-1, None)
    for ep in range(epochs):
        w.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(w(xt[b]), yt[b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            w.eval()
            with torch.no_grad():
                acc = (w(xvt).argmax(1) == yvt).float().mean().item()
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in w.state_dict().items()})
    w.load_state_dict(best[1])
    return w, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="concat", choices=["cls", "patch_mean", "concat"])
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    rows = D.load_rows()
    space = D.label_space(rows)
    index = space["index"]
    z = np.load(C.DATA / "feats" / "dinov2s.npz")
    feats = {"cls": z["cls"], "patch_mean": z["patch_mean"], "concat": np.concatenate([z["cls"], z["patch_mean"]], 1)}[a.feat]
    assert (z["idx"] == np.arange(len(rows))).all()
    mu = feats[[r["idx"] for r in rows if r["split"] == "train"]].mean(0)
    x = l2n(feats - mu)
    split_idx = {s: np.array([r["idx"] for r in rows if r["split"] == s and (s.startswith("unknown") or r["species_id"] in index)], int) for s in D.SPLITS}
    y = np.array([index.get(r.get("species_id"), -1) for r in rows])
    tr = split_idx["train"]
    # kNN
    for s in ("val", "test", "unknown_taxa", "unknown_nonwild"):
        if len(split_idx[s]) == 0:
            continue
        sc, dist = knn_scores(x[split_idx[s]], x[tr], y[tr], len(index), a.k)
        D.save_preds("dinov2s_knn", s, split_idx[s], np.log(np.clip(sc, 1e-6, None)), knn_dist=dist)
    va = split_idx["val"]
    sc, _ = knn_scores(x[va], x[tr], y[tr], len(index), a.k)
    knn_val = float((sc.argmax(1) == y[va]).mean())
    # linear probe
    torch.manual_seed(0)
    w, lin_val = train_linear(x[tr].astype(np.float32), y[tr], len(index), x[va].astype(np.float32), y[va])
    with torch.no_grad():
        for s in ("val", "test", "unknown_taxa", "unknown_nonwild"):
            if len(split_idx[s]) == 0:
                continue
            lg = w(torch.tensor(x[split_idx[s]].astype(np.float32))).numpy()
            _, dist = knn_scores(x[split_idx[s]], x[tr], y[tr], len(index), 1)
            D.save_preds("dinov2s_linear", s, split_idx[s], lg, knn_dist=dist)
    C.DIST.mkdir(parents=True, exist_ok=True)
    np.savez(C.DIST / "dinov2s_linear.npz", weight=w.weight.detach().numpy(), bias=w.bias.detach().numpy(), mean=mu.astype(np.float32), feat=a.feat)
    rec = {"run": "dinov2s_probe", "feat": a.feat, "k": a.k, "n_classes": len(index), "n_train": int(len(tr)), "knn_val_top1": knn_val, "linear_val_top1": lin_val,
           "seconds": time.time() - t0, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    C.append_jsonl(C.DATA / "runs.jsonl", rec)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
