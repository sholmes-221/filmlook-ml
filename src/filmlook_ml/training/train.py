"""Head training on cached inputs (Stage A), seed ensembles, calibration, and optional Stage B.

Stage A trains small heads on cached DINOv2 embeddings and/or measured statistics (seconds on GPU).
Stage B (partial fine-tune of the last DINOv2 blocks) is only worth running if held-out evaluation shows
the DINOv2 variants beat texture-only; it is off by default.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from filmlook_ml.labeling.store import Targets
from filmlook_ml.models.filmlook_model import DinoEmbedder, FilmLookHeads


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_free_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


# ------------------------------------------------------------------ embeddings
@torch.no_grad()
def extract_embeddings(manifest: list[dict], backbone: str, cache_path: Path, size: int = 224, batch_size: int = 64,
                       max_side: int = 1280, dev: torch.device | None = None) -> dict[str, np.ndarray]:
    """Cache keyed by sha256. Images are loaded with the SAME bounded loader as inference."""
    from filmlook_infer.imageio import load_image
    from filmlook_infer.pixels import dinov2_pixels

    cache_path = Path(cache_path)
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        z = np.load(cache_path)
        cache = dict(zip(z["sha"].tolist(), z["emb"]))
    todo = [r for r in manifest if r["sha256"] not in cache]
    if todo:
        dev = dev or device()
        model = DinoEmbedder(backbone).to(dev).eval()
        use_amp = dev.type == "cuda"
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            px = np.stack([dinov2_pixels(load_image(r["path"], max_side=max_side), size) for r in chunk])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                e = model(torch.from_numpy(px).to(dev)).float().cpu().numpy()
            for r, v in zip(chunk, e):
                cache[r["sha256"]] = v
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, sha=np.array(list(cache)), emb=np.stack(list(cache.values())))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return {r["image_id"]: cache[r["sha256"]] for r in manifest}


# ------------------------------------------------------------------ data
@dataclass
class TrainData:
    ids: list[str]
    emb: np.ndarray | None
    feats: np.ndarray
    t: Targets
    split: np.ndarray            # str array
    pairs: tuple                 # (ia, ib, attr, sign, w)

    def tensors(self, dev):
        T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
        t = self.t
        return dict(emb=T(self.emb) if self.emb is not None else None, feats=T(self.feats), score=T(t.score),
                    score_w=T(t.score_w), attrs=T(t.attrs), attrs_w=T(t.attrs_w), medium=T(t.medium),
                    medium_w=T(t.medium_w))


def _wmean(x, w):
    return (x * w).sum() / w.sum().clamp_min(1e-6)


def multitask_loss(out, tg, idx, pairs, lw, margin):
    """out: outputs for ALL rows (pairs index globally); idx: rows of this batch/split.

    Attribute loss = mean over attributes of per-attribute weighted means, so densely-supervised
    attributes (or the overall score) cannot dominate rarely-supervised ones."""
    l_score = _wmean(F.smooth_l1_loss(out["score"][idx], tg["score"][idx], reduction="none", beta=0.1),
                     tg["score_w"][idx])
    la = F.smooth_l1_loss(out["attrs"][idx], tg["attrs"][idx], reduction="none", beta=0.1)
    aw = tg["attrs_w"][idx]
    per_attr = (la * aw).sum(0) / aw.sum(0).clamp_min(1e-6)
    l_attr = per_attr[aw.sum(0) > 0].mean() if (aw.sum(0) > 0).any() else per_attr.sum() * 0
    l_med = _wmean(F.binary_cross_entropy_with_logits(out["medium_logit"][idx], tg["medium"][idx], reduction="none"),
                   tg["medium_w"][idx])
    l_rank = torch.zeros((), device=l_score.device)
    ia, ib, at, sign, w = pairs
    if len(ia):
        def val(rows, attr):
            return torch.where(attr < 0, out["score"][rows], out["attrs"][rows, attr.clamp_min(0)])
        l_rank = _wmean(F.relu(margin - sign * (val(ia, at) - val(ib, at))), w)
    total = lw["score"] * l_score + lw["attributes"] * l_attr + lw["ranking"] * l_rank + lw["medium"] * l_med
    return total, {"score": l_score.item(), "attrs": l_attr.item(), "rank": l_rank.item(), "medium": l_med.item(),
                   "total": total.item()}


def _pairs_for(data: TrainData, split: str, dev):
    ia, ib, at, sign, w = data.pairs
    if len(ia) == 0:
        return tuple(torch.as_tensor(x, device=dev) for x in data.pairs)
    m = (data.split[ia] == split) & (data.split[ib] == split)
    return tuple(torch.as_tensor(x[m], device=dev) for x in (ia, ib, at, sign, w))


def ranking_accuracy(out, pairs) -> float | None:
    ia, ib, at, sign, _ = pairs
    if len(ia) == 0:
        return None
    sc_a = torch.where(at < 0, out["score"][ia], out["attrs"][ia, at.clamp_min(0)])
    sc_b = torch.where(at < 0, out["score"][ib], out["attrs"][ib, at.clamp_min(0)])
    return float(((sc_a - sc_b) * sign > 0).float().mean())


# ------------------------------------------------------------------ calibration
@torch.no_grad()
def fit_temperature(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> float:
    """Grid-search one temperature minimising weighted BCE on validation data."""
    if weight.sum() == 0:
        return 1.0
    best, best_t = float("inf"), 1.0
    for t in np.exp(np.linspace(np.log(0.2), np.log(5.0), 61)):
        loss = _wmean(F.binary_cross_entropy_with_logits(logits / t, target, reduction="none"), weight).item()
        if loss < best:
            best, best_t = loss, float(t)
    return best_t


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    prob, target = prob.ravel(), (target.ravel() >= 0.5).astype(np.float32)
    if prob.size == 0:
        return float("nan")
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(prob[bins == b].mean() - target[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


# ------------------------------------------------------------------ stage A
def train_member(data: TrainData, kind: str, cfg: dict, seed: int, ckpt: Path | None = None,
                 resume: bool = False, log=print) -> tuple[FilmLookHeads, list]:
    """Train one head on cached inputs. ``ckpt`` enables resume (last state saved every epoch)."""
    set_seed(seed)
    dev = device()
    c = cfg["stage_a"]
    heads = FilmLookHeads(kind=kind, hidden=c["hidden"], dropout=c["dropout"],
                          emb_dim=data.emb.shape[1] if data.emb is not None else 768).to(dev)
    tr = data.split == "train"
    mu, sd = data.feats[tr].mean(0), data.feats[tr].std(0) + 1e-6
    heads.feat_mean.copy_(torch.as_tensor(mu)), heads.feat_std.copy_(torch.as_tensor(sd))
    opt = torch.optim.AdamW(heads.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    tg = data.tensors(dev)
    idx_tr = torch.as_tensor(np.where(tr)[0], device=dev)
    idx_va = torch.as_tensor(np.where(data.split == "val")[0], device=dev)
    p_tr, p_va = _pairs_for(data, "train", dev), _pairs_for(data, "val", dev)
    lw = c["loss_weights"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    start, best, bad, hist, best_state = 0, float("inf"), 0, [], None
    if ckpt is not None and resume and Path(ckpt).exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        heads.load_state_dict(s["heads"]); opt.load_state_dict(s["opt"])  # noqa: E702
        start, best, bad, hist, best_state = s["epoch"] + 1, s["best"], s["bad"], s["hist"], s["best_state"]
        torch.set_rng_state(s["rng"]); g.set_state(s["g"])  # noqa: E702
        log(f"resumed {kind} seed {seed} from epoch {start}")
    for ep in range(start, c["epochs"]):
        heads.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=g).to(dev)]
        for i in range(0, len(perm), c["batch_size"]):
            b = perm[i:i + c["batch_size"]]
            out = heads(tg["emb"], tg["feats"])  # full pass: cheap and keeps pair indices valid
            loss, _ = multitask_loss(out, tg, b, p_tr, lw, c["rank_margin"])
            opt.zero_grad(); loss.backward(); opt.step()  # noqa: E702
        heads.eval()
        with torch.no_grad():
            out = heads(tg["emb"], tg["feats"])
            _, lva = multitask_loss(out, tg, idx_va, p_va, lw, c["rank_margin"])
        hist.append({"epoch": ep, "val": lva, "val_rank_acc": ranking_accuracy(out, p_va)})
        if lva["total"] < best - 1e-5:
            best, bad = lva["total"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
        else:
            bad += 1
        if ckpt is not None:
            Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"heads": heads.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best, "bad": bad,
                        "hist": hist, "best_state": best_state, "rng": torch.get_rng_state(), "g": g.get_state()}, ckpt)
        if bad >= c["early_stopping_patience"]:
            break
    heads.load_state_dict(best_state)
    heads.eval()
    with torch.no_grad():  # calibrate medium head on validation
        out = heads(tg["emb"], tg["feats"])
        heads.t_medium.fill_(fit_temperature(out["medium_logit"][idx_va], tg["medium"][idx_va], tg["medium_w"][idx_va]))
    return heads.cpu(), hist


def train_ensemble(data: TrainData, kind: str, cfg: dict, seeds: list[int], ckpt_dir: Path | None = None,
                   resume: bool = False, log=print) -> list[FilmLookHeads]:
    members = []
    for s in seeds:
        t0 = time.time()
        ck = Path(ckpt_dir) / f"last_{kind}_seed{s}.pt" if ckpt_dir else None
        m, hist = train_member(data, kind, cfg, s, ck, resume, log)
        members.append(m)
        log(f"[{kind}] seed {s}: {len(hist)} epochs, best val {min(h['val']['total'] for h in hist):.4f} "
            f"({time.time() - t0:.1f}s)")
    return members


@torch.no_grad()
def predict(members: list[FilmLookHeads], data: TrainData) -> dict[str, np.ndarray]:
    """Per-member and ensemble predictions for ALL rows."""
    emb = torch.as_tensor(data.emb) if data.emb is not None else None
    feats = torch.as_tensor(data.feats)
    outs = [m.eval()(emb, feats) for m in members]
    s = torch.stack([o["score"] for o in outs]).numpy()
    a = torch.stack([o["attrs"] for o in outs]).numpy()
    med = torch.stack([torch.sigmoid(o["medium_logit"] / m.t_medium) for o, m in zip(outs, members)]).numpy()
    return {"score": s.mean(0), "attrs": a.mean(0), "score_std": s.std(0), "film_capture": med.mean(0),
            "score_members": s, "attrs_members": a, "film_capture_members": med}


# ------------------------------------------------------------------ stage B
def train_stage_b(data: TrainData, paths: dict[str, str], cfg: dict, init: FilmLookHeads, seed: int,
                  size: int = 224, log=print) -> dict:
    """Unfreeze the last N DINOv2 blocks (kind dino/fused only). Horizontal flip only: colour, tone and
    grain ARE the labels, so no colour jitter / blur / resize noise. Returns val losses before/after."""
    from filmlook_infer.imageio import load_image
    from filmlook_infer.pixels import dinov2_pixels

    assert init.kind in ("dino", "fused")
    set_seed(seed)
    dev = device()
    c = cfg["stage_b"]
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(c.get("max_gpu_mem_gb", 8) * 2**30, free * 0.9) / total)
    heads = init.to(dev)
    emb = DinoEmbedder(cfg["backbone"]).to(dev)
    emb.freeze(c["unfreeze_last_blocks"])
    params = [{"params": [p for p in emb.parameters() if p.requires_grad], "lr": c["lr_backbone"]},
              {"params": heads.parameters(), "lr": c["lr_heads"]}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    tg = data.tensors(dev)
    tr_idx, va_idx = np.where(data.split == "train")[0], np.where(data.split == "val")[0]
    lw = cfg["stage_a"]["loss_weights"]
    no_pairs = tuple(torch.zeros(0, dtype=dt, device=dev) for dt in (torch.long,) * 3 + (torch.float,) * 2)
    amp = c["amp"] and dev.type == "cuda"
    n = len(data.ids)

    def batch_px(ix, flip_rng=None):
        return torch.from_numpy(np.stack([dinov2_pixels(load_image(paths[data.ids[i]], max_side=cfg["max_side"]), size,
                                                        hflip=bool(flip_rng and flip_rng.random() < 0.5))
                                          for i in ix])).to(dev)

    def full_out(o, ix):
        full = {k: torch.zeros((n,) + v.shape[1:], device=dev, dtype=torch.float32) for k, v in o.items()}
        for k in o:
            full[k] = full[k].index_put((torch.as_tensor(ix, device=dev),), o[k].float())
        return full

    def run_eval():
        emb.eval(); heads.eval()  # noqa: E702
        losses = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            for i in range(0, len(va_idx), c["batch_size"]):
                ix = va_idx[i:i + c["batch_size"]]
                o = heads(emb(batch_px(ix)).float(), tg["feats"][ix])
                losses.append(multitask_loss(full_out(o, ix), tg, torch.as_tensor(ix, device=dev), no_pairs, lw,
                                             0.05)[1]["total"])
        return float(np.mean(losses)) if losses else float("nan")

    initial = best = run_eval()
    bad, hist = 0, []
    log(f"[B] initial val {best:.4f}")
    rng = random.Random(seed)
    for ep in range(c["epochs"]):
        emb.train(); heads.train()  # noqa: E702
        order = list(tr_idx)
        rng.shuffle(order)
        opt.zero_grad()
        for step, i in enumerate(range(0, len(order), c["batch_size"])):
            ix = np.array(order[i:i + c["batch_size"]])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                o = heads(emb(batch_px(ix, rng)).float(), tg["feats"][ix])
            loss, _ = multitask_loss(full_out(o, ix), tg, torch.as_tensor(ix, device=dev), no_pairs, lw, 0.05)
            (loss / c["grad_accum"]).backward()
            if (step + 1) % c["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                opt.step(); opt.zero_grad()  # noqa: E702
        v = run_eval()
        hist.append({"epoch": ep, "val": v})
        log(f"[B] ep {ep} val {v:.4f}")
        if v < best - 1e-5:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= c["early_stopping_patience"]:
                break
    return {"initial_val_loss_no_pairs": initial, "best_val_loss_no_pairs": best, "history": hist,
            "improved": best < initial - 1e-5}


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def load_rgb(path) -> Image.Image:  # small helper kept for scripts
    return Image.open(path).convert("RGB")
