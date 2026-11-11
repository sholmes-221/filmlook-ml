"""C08 experiments on the controlled pairs (research/c08/pairs.py). Every number is on sources never used for training.

    python -m research.c08.experiments delta        # difference-embedding consistency (C08d)
    python -m research.c08.experiments presets      # single-image models vs the live engine, 13 presets + none (C08c)
    python -m research.c08.experiments stocks       # stock classes across unseen LUT implementations, and real scans
    -> reports/c08/*.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PAIRS, REPORTS = Path("data/c08/pairs"), Path("reports/c08")
SLICES = {"L4cls": slice(0, 384), "L4mean": slice(384, 768), "L8cls": slice(768, 1152), "L8mean": slice(1152, 1536),
          "L12cls": slice(1536, 1920), "L12mean": slice(1920, 2304), "L12": slice(1536, 2304), "L4": slice(0, 768), "L8": slice(768, 1536), "all": slice(0, 2304)}
RNG = np.random.default_rng(0)


def norm_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load(prefix: Path = PAIRS):
    meta = [json.loads(l) for l in open(prefix / "meta.jsonl")]
    dino = np.load(prefix / "dino.npy").astype(np.float32)
    colour = np.load(prefix / "colour.npy").astype(np.float32)
    return meta, dino, colour


def bootstrap(correct: np.ndarray, groups: np.ndarray, n: int = 1000) -> tuple[float, float]:
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = RNG.choice(ug, len(ug))
        sel = np.concatenate([idx[g] for g in pick])
        vals.append(correct[sel].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.r_[pos, neg]; lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------- difference embeddings

def cmd_delta(a) -> None:
    meta, dino, colour = load()
    luts = json.loads((PAIRS / "luts.json").read_text())
    lut_ids = [L["id"] for L in luts if L["id"] != "none"]
    idx = {(m["source_id"], m["lut"], m["variant"]): i for i, m in enumerate(meta)}
    split = {m["source_id"]: m["split"] for m in meta}
    cat = {m["source_id"]: m["category"] for m in meta}
    srcs = sorted(split)
    reps = {k: dino[:, s] for k, s in SLICES.items() if k in ("L4cls", "L4mean", "L8cls", "L8mean", "L12cls", "L12mean", "L12", "all")}
    reps["colour312"] = (colour - colour.mean(0)) / (colour.std(0) + 1e-6)
    lut_dist = np.load("data/c08/lut_distance.npy"); lut_order = json.loads(Path("data/c08/lut_pairs.json").read_text())["ids"]
    out = {}
    for name, X in reps.items():
        res = {}
        for mode in ("clean", "varied"):
            D, y, s_of = [], [], []
            for s in srcs:
                o = idx.get((s, "none", 0))
                if o is None:
                    continue
                for li, L in enumerate(lut_ids):
                    j = idx.get((s, L, 0 if mode == "clean" else 1))
                    if j is not None:
                        D.append(X[j] - X[o]); y.append(li); s_of.append(s)
            D, y, s_of = np.asarray(D), np.asarray(y), np.asarray(s_of)
            tr = np.array([split[s] == "train" for s in s_of]); te = np.array([split[s] == "test" for s in s_of])
            protos = np.stack([D[tr & (y == k)].mean(0) for k in range(len(lut_ids))])
            Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)
            pred = (Dn[te] @ Pn.T).argmax(1)
            correct = pred == y[te]
            r = {"prototype_top1": float(correct.mean()), "ci": bootstrap(correct, s_of[te]), "chance": 1 / len(lut_ids)}
            if mode == "clean":
                sub = RNG.choice(np.where(te)[0], min(3000, te.sum()), replace=False)
                C = Dn[sub] @ Dn[sub].T
                same = y[sub][:, None] == y[sub][None]
                diffsrc = s_of[sub][:, None] != s_of[sub][None]
                r["within_lut_cos"] = float(C[same & diffsrc].mean())
                r["between_lut_cos"] = float(C[~same & diffsrc].mean())
                r["between_lut_same_source_cos"] = float(C[~same & ~diffsrc].mean())
                r["delta_to_embedding_norm"] = float(np.median(np.linalg.norm(D, axis=1) / (np.linalg.norm(X[[idx[(s, 'none', 0)] for s in s_of]], axis=1) + 1e-9)))
                # do visually similar LUTs have similar signatures? Spearman between prototype distance and patch dE00
                pos = [lut_order.index(L) for L in lut_ids if L in lut_order]
                keep = [k for k, L in enumerate(lut_ids) if L in lut_order]
                pd = 1 - Pn[keep] @ Pn[keep].T
                ld = lut_dist[np.ix_(pos, pos)]
                iu = np.triu_indices(len(keep), 1)
                from scipy.stats import spearmanr
                r["spearman_signature_vs_patch_de00"] = float(spearmanr(pd[iu], ld[iu]).correlation)
                per_cat = defaultdict(list)
                for ok, s in zip(correct, s_of[te]):
                    per_cat[cat[s]].append(ok)
                r["per_category_top1"] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in per_cat.items() if len(v) >= 50}
            res[mode] = r
        out[name] = res
        print(f"{name:9s} clean top1 {res['clean']['prototype_top1']:.3f} varied {res['varied']['prototype_top1']:.3f} "
              f"within {res['clean']['within_lut_cos']:.3f} between {res['clean']['between_lut_cos']:.3f} |d|/|e| {res['clean']['delta_to_embedding_norm']:.3f} "
              f"rho {res['clean']['spearman_signature_vs_patch_de00']:.2f}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "delta.json").write_text(json.dumps({"n_luts": len(lut_ids), "results": out}, indent=1))


# ------------------------------------------------------------------------------------------- single-image models

class Trainer:
    """Small MLP heads on GPU. Options: auxiliary regression of a teacher signal (difference embedding: the student),
    supervised-contrastive term on batches that hold several LUTs of the same source (hard negatives)."""

    def __init__(self, n_in, n_out, hidden=512, aux_dim=0, supcon=0.0, epochs=40, lr=1e-3, wd=1e-4, seed=0):
        import torch
        self.t = torch
        torch.manual_seed(seed)
        nn = torch.nn
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2)).cuda()
        self.head = nn.Linear(hidden, n_out).cuda()
        self.aux = nn.Linear(hidden, aux_dim).cuda() if aux_dim else None
        self.proj = nn.Sequential(nn.Linear(hidden, 128)).cuda() if supcon else None
        self.supcon, self.epochs, self.lr, self.wd = supcon, epochs, lr, wd
        self.T = 1.0

    def params(self):
        ps = list(self.body.parameters()) + list(self.head.parameters())
        for m in (self.aux, self.proj):
            if m is not None:
                ps += list(m.parameters())
        return ps

    def _supcon(self, z, y, tau=0.1):
        t = self.t
        z = t.nn.functional.normalize(z, dim=1)
        sim = z @ z.T / tau
        eye = t.eye(len(y), device=z.device, dtype=t.bool)
        sim = sim.masked_fill(eye, -1e9)
        pos = (y[:, None] == y[None]) & ~eye
        logp = sim - t.logsumexp(sim, 1, keepdim=True)
        n = pos.sum(1).clamp(min=1)
        return -((logp * pos).sum(1) / n).mean()

    def fit(self, X, y, Xv, yv, aux=None, groups=None):
        t = self.t
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = t.tensor((X - self.mu) / self.sd, dtype=t.float32).cuda(); yt = t.tensor(y).cuda()
        At = t.tensor(aux, dtype=t.float32).cuda() if aux is not None else None
        Xvt = t.tensor((Xv - self.mu) / self.sd, dtype=t.float32).cuda(); yvt = t.tensor(yv).cuda()
        opt = t.optim.AdamW(self.params(), lr=self.lr, weight_decay=self.wd)
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best, best_state = 1e9, None
        n = len(Xt)
        if groups is not None:  # batches made of whole sources: every LUT of a source sits in the same batch (hard negatives)
            ug = np.unique(groups); members = {g: np.where(groups == g)[0] for g in ug}
        for ep in range(self.epochs):
            self.body.train()
            if groups is not None:
                order = RNG.permutation(ug); batches, cur = [], []
                for g in order:
                    cur += list(members[g])
                    if len(cur) >= 512:
                        batches.append(np.array(cur)); cur = []
                if cur:
                    batches.append(np.array(cur))
            else:
                perm = RNG.permutation(n); batches = [perm[i:i + 512] for i in range(0, n, 512)]
            for b in batches:
                bt = t.tensor(b).cuda()
                h = self.body(Xt[bt])
                loss = t.nn.functional.cross_entropy(self.head(h), yt[bt], label_smoothing=0.05)
                if self.aux is not None:
                    loss = loss + t.nn.functional.mse_loss(self.aux(h), At[bt])
                if self.proj is not None:
                    loss = loss + self.supcon * self._supcon(self.proj(h), yt[bt])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            self.body.eval()
            with t.no_grad():
                vl = t.nn.functional.cross_entropy(self.head(self.body(Xvt)), yvt).item()
            if vl < best:
                best = vl; best_state = [p.detach().clone() for p in self.params()]
        for p, s in zip(self.params(), best_state):
            p.data.copy_(s)
        # temperature scaling on validation
        with t.no_grad():
            lv = self.head(self.body(Xvt))
        Ts = np.linspace(0.5, 4, 36)
        nll = [t.nn.functional.cross_entropy(lv / T, yvt).item() for T in Ts]
        self.T = float(Ts[int(np.argmin(nll))])
        return self

    def proba(self, X):
        t = self.t
        self.body.eval()
        with t.no_grad():
            lg = self.head(self.body(t.tensor((X - self.mu) / self.sd, dtype=t.float32).cuda())) / self.T
            return t.softmax(lg, 1).cpu().numpy()


def evaluate(P, y, groups, classes, name):
    top1 = P.argmax(1) == y
    top3 = (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1)
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, P.argmax(1)):
        cm[a, b] += 1
    prec = {classes[k]: round(float(cm[k, k] / max(cm[:, k].sum(), 1)), 3) for k in range(len(classes))}
    rec = {classes[k]: round(float(cm[k, k] / max(cm[k].sum(), 1)), 3) for k in range(len(classes))}
    off = [(cm[i, j] + cm[j, i], classes[i], classes[j]) for i in range(len(classes)) for j in range(i + 1, len(classes))]
    off.sort(reverse=True)
    return {"name": name, "n": int(len(y)), "top1": float(top1.mean()), "top1_ci": bootstrap(top1, groups), "top3": float(top3.mean()),
            "ece": ece(P, y), "precision": prec, "recall": rec, "most_confused": [{"pair": [a, b], "count": int(c)} for c, a, b in off[:6]],
            "confusion": cm.tolist()}


def scene_normaliser(Xdino_orig, Xcol_orig):
    """Ridge map from the content embedding of untouched originals to their colour statistics: what colour this scene
    would have without a look. Residual = observed colour - predicted colour."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=10.0).fit(Xdino_orig, Xcol_orig)


def cmd_presets(a) -> None:
    meta, dino, colour = load()
    classes = [norm_id(L["name"]) for L in json.loads((PAIRS / "luts.json").read_text()) if L["source"] in ("owner_presets", "identity")]
    cls_of = {}
    for L in json.loads((PAIRS / "luts.json").read_text()):
        if L["source"] in ("owner_presets", "identity"):
            cls_of[L["id"]] = classes.index(norm_id(L["name"]))
    keep = np.array([m["lut"] in cls_of for m in meta])
    meta = [m for m, k in zip(meta, keep) if k]; dino, colour = dino[keep], colour[keep]
    y = np.array([cls_of[m["lut"]] for m in meta]); split = np.array([m["split"] for m in meta]); src = np.array([m["source_id"] for m in meta])
    varied = np.array([not m["clean"] for m in meta])
    orig = np.array([m["lut"] == "none" for m in meta])
    # scene normaliser fitted on train originals only
    sn = scene_normaliser(dino[(split == "train") & orig][:, SLICES["L12"]], colour[(split == "train") & orig])
    resid = colour - sn.predict(dino[:, SLICES["L12"]])
    # teacher: difference embedding of each render against its own untouched original (offline only)
    o_idx = {(m["source_id"], m["variant"]): i for i, m in enumerate(meta) if m["lut"] == "none"}
    o_clean = {m["source_id"]: i for i, m in enumerate(meta) if m["lut"] == "none" and m["clean"]}
    teacher = np.stack([np.r_[dino[i, SLICES["all"]] - dino[o_clean[m["source_id"]], SLICES["all"]], colour[i] - colour[o_clean[m["source_id"]]]] for i, m in enumerate(meta)])
    teacher = (teacher - teacher[split == "train"].mean(0)) / (teacher[split == "train"].std(0) + 1e-6)
    from sklearn.decomposition import PCA
    teacher = PCA(64, random_state=0).fit(teacher[split == "train"]).transform(teacher)
    feats = {"A_dino_L12 (production embedding)": dino[:, SLICES["L12"]], "A2_dino_all_layers": dino[:, SLICES["all"]],
             "C_colour312": colour, "B_dino_L12+colour": np.c_[dino[:, SLICES["L12"]], colour],
             "E_dino_all+colour": np.c_[dino[:, SLICES["all"]], colour], "SN_colour+scene_residual": np.c_[colour, resid],
             "SN2_all+colour+residual": np.c_[dino[:, SLICES["all"]], colour, resid]}
    tr, va, te = split == "train", split == "val", (split == "test") & varied
    results, models = [], {}
    # unseen image pool: 13-preset renders of the Common Catalog test scenes
    em, ed, ec = load(PAIRS / "extra_synth_test")
    ey = np.array([classes.index(norm_id(m["preset"])) for m in em]); eg = np.array([m["scene_id"] for m in em])
    eresid = ec - sn.predict(ed[:, SLICES["L12"]])
    efeats = {"A_dino_L12 (production embedding)": ed[:, SLICES["L12"]], "A2_dino_all_layers": ed[:, SLICES["all"]], "C_colour312": ec,
              "B_dino_L12+colour": np.c_[ed[:, SLICES["L12"]], ec], "E_dino_all+colour": np.c_[ed[:, SLICES["all"]], ec],
              "SN_colour+scene_residual": np.c_[ec, eresid], "SN2_all+colour+residual": np.c_[ed[:, SLICES["all"]], ec, eresid]}
    xm, xd, xc = load(PAIRS / "extra_edited")
    xresid = xc - sn.predict(xd[:, SLICES["L12"]])
    xfeats = {"A_dino_L12 (production embedding)": xd[:, SLICES["L12"]], "A2_dino_all_layers": xd[:, SLICES["all"]], "C_colour312": xc,
              "B_dino_L12+colour": np.c_[xd[:, SLICES["L12"]], xc], "E_dino_all+colour": np.c_[xd[:, SLICES["all"]], xc],
              "SN_colour+scene_residual": np.c_[xc, xresid], "SN2_all+colour+residual": np.c_[xd[:, SLICES["all"]], xc, xresid]}
    runs = [(k, {}) for k in feats] + [("SN2_all+colour+residual", {"aux": True}), ("SN2_all+colour+residual", {"supcon": 0.5}),
                                       ("SN2_all+colour+residual", {"aux": True, "supcon": 0.5}), ("C_colour312", {"supcon": 0.5})]
    for fname, opt in runs:
        X = feats[fname]
        label = fname + ("" if not opt else " +" + "+".join(("student(dE teacher)" if k == "aux" else "supcon") for k in opt))
        m = Trainer(X.shape[1], len(classes), aux_dim=teacher.shape[1] if opt.get("aux") else 0, supcon=opt.get("supcon", 0.0)).fit(
            X[tr], y[tr], X[va], y[va], aux=teacher[tr] if opt.get("aux") else None, groups=src[tr] if opt.get("supcon") else None)
        P = m.proba(X[te]); r = evaluate(P, y[te], src[te], classes, label)
        P2 = m.proba(efeats[fname]); r2 = evaluate(P2, ey, eg, classes, label)
        Px = m.proba(xfeats[fname])
        r["unknown_edit_auroc"] = auroc(m.proba(X[te & ~orig]).max(1), Px.max(1))  # confidence on known looks vs edited uploads
        res = {"model": label, "test_sources": r, "unseen_image_pool": r2}
        results.append(res); models[label] = m
        print(f"{label:62s} test {r['top1']:.3f} [{r['top1_ci'][0]:.3f},{r['top1_ci'][1]:.3f}] top3 {r['top3']:.3f} ece {r['ece']:.3f} | pool {r2['top1']:.3f} [{r2['top1_ci'][0]:.3f},{r2['top1_ci'][1]:.3f}] | unk {r['unknown_edit_auroc']:.2f}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "presets.json").write_text(json.dumps({"classes": classes, "n_train": int(tr.sum()), "n_test": int(te.sum()), "results": results}, indent=1))


PRESET_STOCK = {"Kodak_Portra_160": "kodak_portra_160", "Kodak_Portra_400": "kodak_portra_400", "Kodak_Ektar_100": "kodak_ektar_100", "Kodak_Gold_200": "kodak_gold_200",
                "Kodak_Ultramax_400": "kodak_ultramax_400", "Kodak_Color_Plus_200": "kodak_colorplus_200", "Cinestill_50D": "cinestill_50d"}


def line_of(stock: str) -> str:
    for pre, line in (("kodak_portra", "kodak_portra"), ("fuji_superia", "fuji_superia"), ("kodak_ektar", "kodak_ektar"), ("fuji_provia", "fuji_provia"),
                      ("fuji_velvia", "fuji_velvia"), ("kodak_gold", "kodak_gold"), ("kodak_ultramax", "kodak_ultramax"), ("kodak_colorplus", "kodak_colorplus"),
                      ("fuji_pro400h", "fuji_pro400h"), ("kodak_ektachrome", "kodak_ektachrome"), ("cinestill_50d", "cinestill_50d")):
        if stock.startswith(pre):
            return line
    return stock


def cmd_stocks(a) -> None:
    """Stock classes that two independent LUT families implement. Fold A trains on RawTherapee + owner presets and tests on
    GlassMeter; fold B the reverse; always on test sources only. Then a model trained on every implementation is applied
    to the real scans, collapsed to stock lines, against the majority line of the scans it can be scored on."""
    meta, dino, colour = load()
    stock, fam = [], []
    for m in meta:
        st = PRESET_STOCK.get(m["preset"]) if m["lut_source"] == "owner_presets" else m["stock"]
        stock.append(st); fam.append("GM" if m["lut_source"] == "glassmeter_film_looks" else "OTHER" if m["lut_source"] in ("owner_presets", "rawtherapee_film_simulation") else None)
    stock, fam = np.array(stock, dtype=object), np.array(fam, dtype=object)
    both = sorted({s for s in set(stock[fam == "GM"]) & set(stock[fam == "OTHER"]) if s})
    classes = both
    keep = np.isin(stock, classes) & (fam != None)  # noqa: E711
    split = np.array([m["split"] for m in meta]); src = np.array([m["source_id"] for m in meta]); varied = np.array([not m["clean"] for m in meta])
    orig = np.array([m["lut"] == "none" for m in meta])
    sn = scene_normaliser(dino[(split == "train") & orig][:, SLICES["L12"]], colour[(split == "train") & orig])
    resid = colour - sn.predict(dino[:, SLICES["L12"]])
    feats = {"A_dino_L12": dino[:, SLICES["L12"]], "C_colour312": colour, "E_dino_all+colour": np.c_[dino[:, SLICES["all"]], colour],
             "SN2_all+colour+residual": np.c_[dino[:, SLICES["all"]], colour, resid]}
    y = np.array([classes.index(s) if s in classes else -1 for s in stock])
    out = {"classes": classes, "folds": {}, "real": {}}
    for fold, (trf, tef) in {"A_train_RT+presets_test_GlassMeter": ("OTHER", "GM"), "B_train_GlassMeter_test_RT+presets": ("GM", "OTHER")}.items():
        tr = keep & (fam == trf) & (split == "train"); va = keep & (fam == trf) & (split == "val"); te = keep & (fam == tef) & (split == "test") & varied
        res = []
        for fname, X in feats.items():
            for opt in ({}, {"supcon": 0.5}):
                m = Trainer(X.shape[1], len(classes), supcon=opt.get("supcon", 0.0)).fit(X[tr], y[tr], X[va], y[va], groups=src[tr] if opt else None)
                r = evaluate(m.proba(X[te]), y[te], src[te], classes, fname + (" +supcon" if opt else ""))
                res.append(r)
                print(f"{fold[:1]} {r['name']:34s} top1 {r['top1']:.3f} [{r['top1_ci'][0]:.3f},{r['top1_ci'][1]:.3f}] top3 {r['top3']:.3f} (chance {1/len(classes):.3f})", flush=True)
        out["folds"][fold] = res
    # real scans: train on every implementation (train sources), score the scans whose line the model can name
    rm, rd, rc = load(PAIRS / "extra_real")
    rresid = rc - sn.predict(rd[:, SLICES["L12"]])
    rfeats = {"A_dino_L12": rd[:, SLICES["L12"]], "C_colour312": rc, "E_dino_all+colour": np.c_[rd[:, SLICES["all"]], rc], "SN2_all+colour+residual": np.c_[rd[:, SLICES["all"]], rc, rresid]}
    lines = sorted({line_of(c) for c in classes})
    M = np.zeros((len(classes), len(lines)))
    for k, c in enumerate(classes):
        M[k, lines.index(line_of(c))] = 1
    sel = np.array([r.get("line") in lines for r in rm]); ry = np.array([lines.index(r["line"]) if r.get("line") in lines else -1 for r in rm])
    ru = np.array([r["uploader"] for r in rm])
    cnt = Counter(ry[sel]); maj = cnt.most_common(1)[0]
    out["real"]["n_scored"] = int(sel.sum()); out["real"]["lines"] = {lines[k]: int(v) for k, v in cnt.items()}
    out["real"]["majority_line"] = lines[maj[0]]; out["real"]["majority_rate"] = maj[1] / sel.sum()
    tr = keep & (split == "train"); va = keep & (split == "val")
    for fname, X in feats.items():
        m = Trainer(X.shape[1], len(classes)).fit(X[tr], y[tr], X[va], y[va])
        L = m.proba(rfeats[fname][sel]) @ M
        ok = L.argmax(1) == ry[sel]
        out["real"][fname] = {"line_top1": float(ok.mean()), "ci": bootstrap(ok, ru[sel]), "top3": float((np.argsort(-L, 1)[:, :3] == ry[sel][:, None]).any(1).mean()),
                              "predicted_line_distribution": {lines[k]: int(v) for k, v in Counter(L.argmax(1)).items()}}
        print(f"real {fname:28s} line top1 {ok.mean():.3f} [{out['real'][fname]['ci'][0]:.3f},{out['real'][fname]['ci'][1]:.3f}]  majority {out['real']['majority_rate']:.3f} ({out['real']['majority_line']}) n={sel.sum()}", flush=True)
    (REPORTS / "stocks.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("delta", "presets", "stocks"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"delta": cmd_delta, "presets": cmd_presets, "stocks": cmd_stocks}[a.cmd](a)


if __name__ == "__main__":
    main()
