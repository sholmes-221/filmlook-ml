"""E19 candidates for scripts/style_benchmark.py --extra research.candidates_e19:build

Projection on frozen hybrid features (explicit 28 ++ DINOv2 L4+L8), trained on TRAIN-split sources only, with:
  labels      filmsynth family ids + SDXL treatment ids + Z-Image treatment ids; all controls one class ("control")
  hard_neg    same-source pairs with different labels count `hard_w` times in the denominator (content must not help)
  decorr      penalty lam * mean |corr(z_k, c_j)| between projection dims and the (standardised) DINOv2 global content
              vector, VICReg-style cross-covariance, pushing the projection to be unpredictable from content
Variants: all_gen (labels from all generators, no extras), all_gen_hard (hard_w=4), all_gen_hard_decorr (hard_w=4,
lam=1.0), fs_only_hard_decorr (filmsynth labels only, for the ablation).
"""
from __future__ import annotations

import numpy as np
import torch


def train_projection(Xtr, labels, src, content, dev, dim=64, epochs=400, temp=0.1, hard_w=1.0, lam=0.0, seed=0):
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    X = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    lab = torch.tensor(np.unique(labels, return_inverse=True)[1], device=dev)
    same_src = torch.tensor(src[:, None] == src[None], device=dev)
    pos = (lab[:, None] == lab[None]).float()
    eye = torch.eye(len(X), device=dev, dtype=torch.bool)
    pos = pos.masked_fill(eye | same_src, 0.0)                       # positives: same label, different source
    neg_w = torch.ones_like(pos).masked_fill(same_src & (lab[:, None] != lab[None]), float(hard_w)).masked_fill(eye, 0.0)
    C = torch.tensor(content, dtype=torch.float32, device=dev); C = (C - C.mean(0)) / (C.std(0) + 1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dim)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        zraw = net(X + 0.05 * torch.randn_like(X))
        z = torch.nn.functional.normalize(zraw, dim=1)
        logits = z @ z.T / temp
        logits = logits.masked_fill(eye, -1e9)
        # weighted log-softmax denominator (hard negatives count more)
        m = logits.max(1, keepdim=True).values
        den = torch.log((neg_w * torch.exp(logits - m)).sum(1, keepdim=True) + 1e-9) + m
        logp = logits - den
        loss = (-(pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()
        if lam > 0:
            zc = (zraw - zraw.mean(0)) / (zraw.std(0) + 1e-6)
            cross = (zc.T @ C) / len(X)                                   # dim x content, correlation-like
            loss = loss + lam * cross.abs().mean() * 10.0
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor(sc.transform(Xn), dtype=torch.float32, device=dev)).detach().cpu().numpy()


def build(F, rows, dev):
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); src = np.array([r["source"] for r in rows])
    fam = np.array([r["family"] for r in rows]); treated = np.array([bool(r.get("treated", False)) for r in rows])
    gen_mask = np.isin(st, ["filmsynth", "sdxl", "zimage"]) & (sp == "train")
    lab_all = np.where(treated, np.char.add(np.char.add(st.astype(str), ":"), fam.astype(str)), "control")
    fs_mask = (st == "filmsynth") & (sp == "train")
    out = {}
    H, G = F["hybrid"], F["dino_global"]
    out["e19_all_gen"] = train_projection(H[gen_mask], lab_all[gen_mask], src[gen_mask], G[gen_mask], dev)
    out["e19_all_gen_hard"] = train_projection(H[gen_mask], lab_all[gen_mask], src[gen_mask], G[gen_mask], dev, hard_w=4.0)
    out["e19_all_gen_hard_decorr"] = train_projection(H[gen_mask], lab_all[gen_mask], src[gen_mask], G[gen_mask], dev, hard_w=4.0, lam=1.0)
    out["e19_fs_only_hard_decorr"] = train_projection(H[fs_mask], lab_all[fs_mask], src[fs_mask], G[fs_mask], dev, hard_w=4.0, lam=1.0)
    return {k: (lambda f, p=p: p(f["hybrid"])) for k, p in out.items()}
