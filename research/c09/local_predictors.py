"""C09: can a small local model predict the Claude style annotation well enough to recover its matching gain?

    python -m research.c09.local_predictors --seed 0     # -> reports/c09/local_predictors_seed0.json (+ dist/c09/style_head.npz)
    python -m research.c09.local_predictors --pool       # pooled paired CIs over the two seed reports

Targets: the style-annotation-v2 vector (Haiku, data/embeddings/photographers/claude_v1.npz) without any subject field
(subject_primary, subjects_secondary, people_count, human_presence, setting, urban_setting), as in subject_audit.py.
They are Claude outputs used as training targets for a small categorisation head, under the owner's recorded
permission (docs/AI_DATA_POLICY.md, training_authorization: user_confirmed; preflight PASS). Nothing calls an API.

Inputs, all computable in production from the one DINOv2-small pass (data/c09/bank_inputs.npz, descriptors.npz):
  A  linear multinomial / logistic heads on L12 [CLS, mean] (768)            - the old baseline
  B  MLP (2 x 256, dropout) on the six DINOv2 parts (2304)
  C  the same MLP on 2304 + F36 + axes17 + film205
Training uses BANK-split works only; hyper-parameters and the predictor are chosen on VAL; TEST is read once.
Single-choice fields: softmax cross-entropy per field; multi-label fields and the 0..4 axes: binary cross-entropy.

Matching (same split, classes, scorer and cross-subject protocol as subject_audit.py), on TEST queries:
  ref        prod hybrid: sims [1.0, 0.5, 0.5] over prod, axes17, film205
  oracle     ref + w * cos(real annotation)          (bank and query real; offline upper reference)
  predicted  ref + w * cos(predicted annotation)     (bank and query predicted)
  asymmetric ref + w * cos(real bank, predicted query)   (the bank index is built offline; the query never sees Claude)
Annotation space: centred by the real-annotation bank mean, cosine. w in {0, .25, .5, .75, 1} chosen on VAL under the
cross-subject protocol. The query's subject label (Claude) only defines the cross-subject protocol; it is never an input.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from research.photographers.evaluate import EMB, HELD_OUT, l2, leakage_groups, load_rows, load_vectors, split_groups, topm_scores
from research.photographers.subject_audit import subjects

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports" / "c09"; DIST = ROOT / "dist" / "c09"
SUBJECT_FIELDS = ("subject_primary", "subjects_secondary", "people_count", "human_presence", "setting", "urban_setting")
GROUPS = {
    "film_rendering": ("film_rendering",), "composition": ("composition",), "motifs": ("motifs",), "style_traits": ("style_traits",),
    "mood": ("mood",),
    "palette_tone": ("palette", "colour_temperature", "saturation", "shadows", "highlights", "tonal_range", "tonal_treatment", "key",
                     "contrast", "dominant_colours"),
    "light": ("light_source", "light_direction", "light_quality"),
    "camera_space": ("framing", "vantage", "depth", "focus", "sharpness", "texture", "moment", "people_awareness"),
}
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------------------------- data
def field_of(name: str) -> str:
    for sep in ("=", ":"):
        if sep in name:
            return name.split(sep)[0]
    return name


def load(seed: int) -> dict:
    rows = load_rows(); ids = [r["work_id"] for r in rows]
    Xp = load_vectors("prod", ids)
    groups, _ = leakage_groups(rows, l2(Xp[:, :384])); split = split_groups(rows, groups, seed)
    pid = np.array([r["photographer_id"] for r in rows]); counts = Counter(pid.tolist())
    classes = sorted(p for p, n in counts.items() if n >= 40 and p not in HELD_OUT); cidx = {p: i for i, p in enumerate(classes)}
    y = np.array([cidx.get(p, -1) for p in pid]); s = subjects(ids)

    d = np.load(EMB / "descriptors.npz", allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
    order = [pos[w] for w in ids]
    ax = np.nan_to_num(d["axes"][order], nan=0.5).astype(np.float32); f205 = d["film205"][order].astype(np.float32)
    bi = np.load(ROOT / "data" / "c09" / "bank_inputs.npz", allow_pickle=False); pos = {w: i for i, w in enumerate(bi["work_ids"].tolist())}
    order = [pos[w] for w in ids]
    layers = bi["layers"][order].astype(np.float32); F36 = bi["F36"][order].astype(np.float32)
    an = np.load(EMB / "claude_v1.npz", allow_pickle=False); pos = {w: i for i, w in enumerate(an["work_ids"].tolist())}
    names_all = [str(n) for n in an["feature_names"]]
    keep = [i for i, n in enumerate(names_all) if not n.startswith(SUBJECT_FIELDS)]
    A = an["X"][[pos[w] for w in ids]][:, keep].astype(np.float32)
    names = [names_all[i] for i in keep]
    return dict(rows=rows, ids=ids, Xp=Xp, groups=groups, split=split, y=y, s=s, n_class=len(classes), classes=classes,
                ax=ax, f205=f205, layers=layers, F36=F36, A=A, names=names)


def layout(names: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    """Column groups of the single-choice fields (softmax) and the indices of every binary / ordinal column (sigmoid)."""
    single, other = {}, []
    for j, n in enumerate(names):
        if "=" in n:
            single.setdefault(n.split("=")[0], []).append(j)
        else:
            other.append(j)
    return [np.array(v) for v in single.values()], np.array(other)


# ------------------------------------------------------------------------------------------------------ predictors
class Head(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, dropout: float):
        super().__init__()
        if hidden:
            self.net = torch.nn.Sequential(torch.nn.Dropout(dropout / 2), torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                                           torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, d_out))
        else:
            self.net = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.net(x)


def probs(logits: torch.Tensor, single: list, other: np.ndarray) -> torch.Tensor:
    out = torch.empty_like(logits)
    for g in single:
        out[:, g] = torch.softmax(logits[:, g], 1)
    out[:, other] = torch.sigmoid(logits[:, other])
    return out


def loss_fn(logits, T, single, other):
    l = sum(-(T[:, g] * torch.log_softmax(logits[:, g], 1)).sum(1).mean() for g in single)
    return l + torch.nn.functional.binary_cross_entropy_with_logits(logits[:, other], T[:, other], reduction="none").sum(1).mean()


def train(X, T, tr, va, single, other, hidden, dropout, wd, seed, max_epochs=400, patience=25):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(X, device=DEV); Tt = torch.tensor(T, device=DEV)
    m = Head(X.shape[1], T.shape[1], hidden, dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    tri = torch.tensor(tr, device=DEV); vai = torch.tensor(va, device=DEV)
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(max_epochs):
        m.train()
        perm = tri[torch.randperm(len(tri), generator=g).to(DEV)]
        for k in range(0, len(perm), 128):
            b = perm[k:k + 128]
            loss = loss_fn(m(Xt[b]), Tt[b], single, other)
            if not hidden:  # explicit L2 for the logistic baseline (AdamW decay is relative to lr)
                loss = loss + wd * (m.net.weight ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(loss_fn(m(Xt[vai]), Tt[vai], single, other))
        if vl < best - 1e-4:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        P = probs(m(Xt), single, other).cpu().numpy()
    return m, P, best, best_ep


def quality(P, A, idx, names, single, other) -> dict:
    """Per-group mean AUROC (every binary / one-hot column with both classes present) and Spearman for the 0..4 axes."""
    out = {}
    fields = np.array([field_of(n) for n in names])
    ordinal = [j for j, n in enumerate(names) if field_of(n) == n]
    for gname, fs in GROUPS.items():
        aucs = []
        for j in np.flatnonzero(np.isin(fields, fs)):
            t = A[idx, j]
            if 0 < t.sum() < len(t):
                aucs.append(roc_auc_score(t, P[idx, j]))
        out[gname] = {"auroc": float(np.mean(aucs)) if aucs else None, "n_columns": len(aucs)}
    rho = {names[j]: float(spearmanr(A[idx, j], P[idx, j]).statistic) for j in ordinal}
    out["ordinal_axes"] = {"spearman_mean": float(np.mean(list(rho.values()))), "per_axis": rho}
    all_auc = [roc_auc_score(A[idx, j], P[idx, j]) for j in range(len(names)) if j not in ordinal and 0 < A[idx, j].sum() < len(idx)]
    out["all_binary_auroc"] = float(np.mean(all_auc))
    return out


# --------------------------------------------------------------------------------------------------------- matching
def rank_metrics(S, yb, yq, n_class):
    sc = topm_scores(S, yb, n_class, 3)
    rank = (sc > sc[np.arange(len(yq)), yq][:, None]).sum(1)
    return {"top1": (rank == 0).astype(float), "top3": (rank < 3).astype(float), "mrr": 1.0 / (rank + 1)}


def run_match(parts, ws, y, s, q, b, n_class, cross):
    S = sum(w * P[np.ix_(q, b)] for w, P in zip(ws, parts) if w)
    if cross:
        S = np.where(s[q][:, None] == s[b][None, :], -1e9, S)
    return rank_metrics(S, y[b], y[q], n_class)


def paired_ci(diff, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed); ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(ug)); cnt = np.bincount(inv, minlength=len(ug))
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        vals.append((w * sums).sum() / (w * cnt).sum())
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# ------------------------------------------------------------------------------------------------------------ main
def main_seed(seed: int) -> None:
    D = load(seed); split, y, s, A, names = D["split"], D["y"], D["s"], D["A"], D["names"]
    single, other = layout(names)
    tr = np.flatnonzero(split == "bank"); va = np.flatnonzero(split == "val"); te = np.flatnonzero(split == "test")
    L = D["layers"]
    feats = {"A_linear_L12": L[:, 1536:], "B_mlp_layers": L, "C_mlp_layers_desc": np.hstack([L, D["F36"], D["ax"], D["f205"]])}
    grids = {"A_linear_L12": [(0, 0.0, wd) for wd in (1e-3, 1e-2, 3e-2, 1e-1)],
             "B_mlp_layers": [(256, dr, wd) for dr in (0.3, 0.5) for wd in (1e-4, 1e-2)],
             "C_mlp_layers_desc": [(256, dr, wd) for dr in (0.3, 0.5) for wd in (1e-4, 1e-2)]}
    rep = {"seed": seed, "n_train_bank": len(tr), "n_val": len(va), "n_test_all": len(te), "n_targets": len(names), "device": DEV,
           "predictors": {}}
    preds, models = {}, {}
    for name, X in feats.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        best = None
        for hidden, dr, wd in grids[name]:
            m, P, vl, ep = train(Xn, A, tr, va, single, other, hidden, dr, wd, seed)
            vq = quality(P, A, va, names, single, other)["all_binary_auroc"]
            print(f"seed{seed} {name} h={hidden} dr={dr} wd={wd}: val loss {vl:.3f} (ep {ep}) val AUROC {vq:.4f}", flush=True)
            if best is None or vl < best[0]:
                best = (vl, (hidden, dr, wd), m, P, ep)
        vl, hp, m, P, ep = best
        n_par = sum(p.numel() for p in m.parameters())
        # CPU latency, single image, numpy float32 forward of the exported weights
        Ws = [(l.weight.detach().cpu().numpy().T.astype(np.float16).astype(np.float32), l.bias.detach().cpu().numpy().astype(np.float32))
              for l in m.net.modules() if isinstance(l, torch.nn.Linear)]
        x = Xn[te[:1]]
        def fwd(x):
            h = x
            for k, (W, b) in enumerate(Ws):
                h = h @ W + b
                if k < len(Ws) - 1:
                    h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
            return h
        for _ in range(20):
            fwd(x)
        t0 = time.perf_counter(); n_rep = 500
        for _ in range(n_rep):
            fwd(x)
        lat_ms = (time.perf_counter() - t0) / n_rep * 1e3
        preds[name] = P; models[name] = (m, mu, sd, hp)
        rep["predictors"][name] = {"hyper": {"hidden": hp[0], "dropout": hp[1], "weight_decay": hp[2], "epochs": ep + 1}, "val_loss": vl,
                                   "val_quality": quality(P, A, va, names, single, other),
                                   "test_quality": quality(P, A, te, names, single, other),
                                   "params": int(n_par), "fp16_bytes": int(n_par * 2), "cpu_latency_ms_batch1_numpy": round(lat_ms, 4),
                                   "input_dim": int(X.shape[1])}
        print(name, "test", {k: (v["auroc"] if isinstance(v, dict) and "auroc" in v else v) for k, v in rep["predictors"][name]["test_quality"].items() if k != "ordinal_axes"},
              "spearman", round(rep["predictors"][name]["test_quality"]["ordinal_axes"]["spearman_mean"], 3), flush=True)
    best_name = min(rep["predictors"], key=lambda k: rep["predictors"][k]["val_loss"])
    rep["selected_predictor"] = best_name; rep["selection_rule"] = "lowest validation loss (same loss for all three predictors)"

    # ---------------- matching
    ok = (y >= 0) & (s != None)  # noqa: E711
    b, v, t = np.flatnonzero(ok & (split == "bank")), np.flatnonzero(ok & (split == "val")), np.flatnonzero(ok & (split == "test"))
    rep["matching_n"] = {"bank": len(b), "val": len(v), "test": len(t), "classes": D["n_class"]}
    axes = l2(D["ax"] - D["ax"][b].mean(0)); film = l2((D["f205"] - D["f205"][b].mean(0)) / (D["f205"][b].std(0) + 1e-6))
    Xp = l2(D["Xp"])
    base = [Xp @ Xp.T, axes @ axes.T, film @ film.T]
    centre = A[b].mean(0)
    real = l2(A - centre)
    def predicted(P):
        Z = l2(P - centre); return Z @ Z.T
    def asym(P):
        Z = l2(P - centre); return Z @ real.T  # rows: query predicted; columns: bank real
    variants = {"oracle": lambda P: real @ real.T, "predicted": predicted, "asymmetric": asym}

    def evaluate_variant(Sann, q, cross, w):
        return run_match(base + [Sann], [1.0, 0.5, 0.5, w], y, s, q, b, D["n_class"], cross)

    # validation matching for every predictor (transparency), test for the selected one only
    rep["val_matching"] = {}
    ref_val = {c: run_match(base, [1.0, 0.5, 0.5], y, s, v, b, D["n_class"], c)["top1"].mean() for c in (False, True)}
    rep["val_matching"]["reference"] = {"full": ref_val[False], "cross": ref_val[True]}
    chosen_w = {}
    for pname, P in preds.items():
        for vname, f in variants.items():
            if vname == "oracle" and pname != best_name:
                continue
            Sann = f(P)
            accs = {w: float(evaluate_variant(Sann, v, True, w)["top1"].mean()) for w in W_GRID}
            wbest = max(W_GRID, key=lambda w: (accs[w], -w))
            key = "oracle" if vname == "oracle" else f"{vname}:{pname}"
            rep["val_matching"][key] = {"cross_by_w": accs, "w": wbest,
                                        "full_at_w": float(evaluate_variant(Sann, v, False, wbest)["top1"].mean())}
            if pname == best_name:
                chosen_w[vname] = wbest
    print("val matching", json.dumps(rep["val_matching"], indent=None)[:1500], flush=True)

    P = preds[best_name]
    rep["test_matching"] = {}; per_query = {"groups": D["groups"][t].tolist()}
    for cross in (False, True):
        proto = "cross" if cross else "full"
        ref = run_match(base, [1.0, 0.5, 0.5], y, s, t, b, D["n_class"], cross)
        out = {"reference": {k: float(x.mean()) for k, x in ref.items()}}
        per_query[f"{proto}:reference"] = {k: x.tolist() for k, x in ref.items()}
        for vname, f in variants.items():
            r = evaluate_variant(f(P), t, cross, chosen_w[vname])
            out[vname] = {"w": chosen_w[vname], **{k: float(x.mean()) for k, x in r.items()},
                          "diff_vs_ref": {k: float((r[k] - ref[k]).mean()) for k in r},
                          "diff_ci95": {k: paired_ci(r[k] - ref[k], D["groups"][t]) for k in r}}
            per_query[f"{proto}:{vname}"] = {k: x.tolist() for k, x in r.items()}
        rep["test_matching"][proto] = out
        print(proto, json.dumps(out), flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(REP / f"local_predictors_seed{seed}.json", "w"), indent=1)
    np.savez_compressed(REP / f"per_query_seed{seed}.npz", **{k.replace(":", "__") + ("__" + m if isinstance(vv, dict) else ""): np.array(x)
                        for k, vv in per_query.items() for m, x in (vv.items() if isinstance(vv, dict) else [("", vv)])})

    # ---------------- export (seed 0 defines the shipped candidate)
    if seed == 0:
        m, mu, sd, hp = models[best_name]
        lin = [l for l in m.net.modules() if isinstance(l, torch.nn.Linear)]
        arrs = {f"W{k}": l.weight.detach().cpu().numpy().T.astype(np.float16) for k, l in enumerate(lin)}
        arrs.update({f"b{k}": l.bias.detach().cpu().numpy().astype(np.float16) for k, l in enumerate(lin)})
        sg = np.zeros(len(names), dtype=np.int16) - 1
        for k, g in enumerate(single):
            sg[g] = k
        DIST.mkdir(parents=True, exist_ok=True)
        path = DIST / "style_head.npz"
        np.savez(path, **arrs, input_mean=mu.astype(np.float32), input_std=sd.astype(np.float32), target_bank_mean=centre.astype(np.float32),
                 feature_names=np.array(names), softmax_group=sg, activation=np.array("gelu"), predictor=np.array(best_name),
                 inputs=np.array({"A_linear_L12": "layers[1536:2304] (L12 CLS, L12 mean)", "B_mlp_layers": "layers[0:2304]",
                                  "C_mlp_layers_desc": "layers 2304 + F36 + axes17 (nan->0.5) + film205"}[best_name]),
                 note=np.array("C09 research export, seed-0 bank split; trained on Claude style annotations (owner permission user_confirmed)"))
        rep["export"] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "predictor": best_name}
        json.dump(rep, open(REP / f"local_predictors_seed{seed}.json", "w"), indent=1)
        print("exported", rep["export"])


def pool() -> None:
    """Pooled paired difference over both seeds, bootstrap clusters = (seed, leakage group)."""
    out = {}
    zs = {sd: np.load(REP / f"per_query_seed{sd}.npz") for sd in (0, 1)}
    for proto in ("full", "cross"):
        for vname in ("oracle", "predicted", "asymmetric"):
            for met in ("top1", "top3", "mrr"):
                diffs, groups = [], []
                for sd, z in zs.items():
                    diffs.append(z[f"{proto}__{vname}__{met}"] - z[f"{proto}__reference__{met}"])
                    groups.append(np.array([f"{sd}:{g}" for g in z["groups"]]))
                d = np.concatenate(diffs); g = np.concatenate(groups)
                out.setdefault(proto, {}).setdefault(vname, {})[met] = {"diff": round(float(d.mean()), 4), "ci95": paired_ci(d, g)}
    json.dump(out, open(REP / "local_predictors_pooled.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    pool() if a.pool else main_seed(a.seed)
