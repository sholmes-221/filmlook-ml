"""C12b: open-set look ranker (13 looks + untouched + unknown edit) vs the live engine. Protocol: PREREGISTRATION.md.

    python -m research.c12.openset select     # validation only: every candidate, the selection rule -> reports/c12/select.json
    python -m research.c12.openset test       # final 5-seed model, test metrics, gate -> reports/c12/test.json, dist/c12/look_openset.npz
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from research.c08.experiments import auroc, ece

DATA, REPORTS, DIST = Path("data/c12"), Path("reports/c12"), Path("dist/c12")
RNG = np.random.default_rng(0)
CHROMA_P50_IDX = 177  # filmlook.filmstyle.descriptors.NAMES.index("chroma_p50")
MIN_CHROMA = 0.04
TARGET_ACCEPT = 0.90


# ------------------------------------------------------------------------------------------------ data
def load(group: str) -> dict:
    p = DATA / group
    meta = [json.loads(l) for l in open(p / "meta.jsonl")]
    dino = np.load(p / "dino.npy").astype(np.float32)
    l48 = dino[:, :1536].reshape(len(dino), 4, 384)
    l48 = (l48 / (np.linalg.norm(l48, axis=2, keepdims=True) + 1e-9)).reshape(len(dino), -1)
    desc = np.nan_to_num(np.load(p / "desc.npy"), nan=0.0, posinf=0.0, neginf=0.0)
    col = np.nan_to_num(np.load(p / "col312.npy"))
    return {"meta": meta, "desc": desc, "dino48": l48, "col312": col, "engine_d": np.load(p / "engine_d.npy"),
            "eng_unc": np.array([m["engine_uncertain"] for m in meta])}


def feats(g: dict, name: str) -> np.ndarray:
    return {"desc": g["desc"], "desc+dino48": np.c_[g["desc"], g["dino48"]], "col312": g["col312"], "col312+dino48": np.c_[g["col312"], g["dino48"]]}[name]


DEPLOYABLE = ("desc", "desc+dino48")
ABLATION = ("col312", "col312+dino48")


def classes() -> list[str]:
    return [L["cls"] for L in json.loads((DATA / "luts.json").read_text())["known"]]


def mask(g, **kw):
    m = np.ones(len(g["meta"]), bool)
    for k, v in kw.items():
        vals = np.array([r.get(k) for r in g["meta"]], dtype=object)
        m &= np.array([x in v for x in vals]) if isinstance(v, (set, list, tuple)) else (vals == v)
    return m


def labels(g, C):
    return np.array([C.index(r["cls"]) if r.get("cls") in C else len(C) for r in g["meta"]])


# ------------------------------------------------------------------------------------------------ model
class MLP:
    """2-hidden-layer MLP (GELU), trained on GPU, exported to numpy. n_out = 14 (known only) or 15 (+ unknown)."""

    def __init__(self, n_in, n_out, hidden=256, epochs=40, lr=1e-3, wd=1e-4, seed=0):
        import torch
        torch.manual_seed(seed)
        nn = torch.nn
        self.t = torch
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_out)).cuda()
        self.epochs, self.lr, self.wd, self.seed = epochs, lr, wd, seed
        self.T = 1.0

    def fit(self, X, y, Xv, yv):
        t = self.t
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = t.tensor((X - self.mu) / self.sd, dtype=t.float32).cuda(); yt = t.tensor(y).cuda()
        Xvt = t.tensor((Xv - self.mu) / self.sd, dtype=t.float32).cuda(); yvt = t.tensor(yv).cuda()
        opt = t.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        rng = np.random.default_rng(self.seed)
        best, state = 1e9, None
        for _ in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(Xt))
            for i in range(0, len(perm), 512):
                b = t.tensor(perm[i:i + 512]).cuda()
                loss = t.nn.functional.cross_entropy(self.net(Xt[b]), yt[b], label_smoothing=0.05)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            self.net.eval()
            with t.no_grad():
                vl = t.nn.functional.cross_entropy(self.net(Xvt), yvt).item()
            if vl < best:
                best, state = vl, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(state)
        return self

    def logits(self, X):
        t = self.t
        self.net.eval()
        with t.no_grad():
            return self.net(t.tensor((X - self.mu) / self.sd, dtype=t.float32).cuda()).cpu().numpy()

    def penult(self, X):
        t = self.t
        self.net.eval()
        with t.no_grad():
            return self.net[:-1](t.tensor((X - self.mu) / self.sd, dtype=t.float32).cuda()).cpu().numpy()

    def export(self) -> dict:
        sd = {k: v.cpu().numpy().astype(np.float32) for k, v in self.net.state_dict().items()}
        return {"mu": self.mu.astype(np.float32), "sd": self.sd.astype(np.float32), "W1": sd["0.weight"].T, "b1": sd["0.bias"],
                "W2": sd["3.weight"].T, "b2": sd["3.bias"], "W3": sd["6.weight"].T, "b3": sd["6.bias"]}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def logsumexp(z):
    m = z.max(1); return m + np.log(np.exp(z - m[:, None]).sum(1))


def fit_T(logits, y):
    Ts = np.exp(np.linspace(np.log(0.3), np.log(5), 60))
    nll = [-np.log(softmax(logits / T)[np.arange(len(y)), y] + 1e-12).mean() for T in Ts]
    return float(Ts[int(np.argmin(nll))])


class Maha:
    def fit(self, H, y, k):
        from sklearn.covariance import LedoitWolf
        self.mu = np.stack([H[y == c].mean(0) for c in range(k)])
        self.P = LedoitWolf().fit(H - self.mu[y]).precision_
        return self

    def score(self, H):
        d = np.stack([np.einsum("ij,jk,ik->i", H - m, self.P, H - m) for m in self.mu], 1)
        return -d.min(1)


def scores(head: str, Z: np.ndarray, T: float, k: int, maha=None, H=None) -> dict:
    """Known-ness scores (higher = more like a known look) from logits Z."""
    out = {}
    P = softmax(Z / T)
    if head == "unk":
        out["1-p_unknown"] = 1 - P[:, k]
        out["max_known_p"] = P[:, :k].max(1)
        out["energy_known"] = logsumexp(Z[:, :k] / T)
    else:
        out["msp"] = P.max(1)
        out["energy"] = logsumexp(Z / T)
        if maha is not None:
            out["mahalanobis"] = maha.score(H)
    return out


def known_probs(Z, T, k):
    return softmax(Z[:, :k] / T)


# ------------------------------------------------------------------------------------------------ engine
def engine_scores(g):
    d = g["engine_d"].astype(np.float64)
    s = np.sort(d, 1)
    scale = np.maximum(np.median(d, 1) - s[:, 0], 1e-6)
    return {"-best_distance": -s[:, 0], "margin": (s[:, 1] - s[:, 0]) / scale}


# ------------------------------------------------------------------------------------------------ metrics
def boot_ci(v, groups, n=1000, rng=None):
    rng = rng or np.random.default_rng(1)
    ug = np.unique(groups); idx = {u: np.where(groups == u)[0] for u in ug}
    vals = [v[np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def topk(P, y, k=1):
    return (np.argsort(-P, 1)[:, :k] == y[:, None]).any(1)


def acc_cov(conf_known: np.ndarray, correct: np.ndarray, is_unknown: np.ndarray, points=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)):
    """Accept the most confident fraction c; accuracy = accepted known with correct top-1 / accepted (accepted unknown = wrong)."""
    order = np.argsort(-conf_known)
    ok = (correct & ~is_unknown)[order]
    out = {}
    for c in points:
        n = max(1, int(round(c * len(order))))
        out[f"{c:.1f}"] = round(float(ok[:n].mean()), 3)
    return out


# ------------------------------------------------------------------------------------------------ training helpers
def train_sets(P, fname, head, C):
    k = len(C)
    y = labels(P, C)
    tr = mask(P, split="train", perturb=None); va = mask(P, split="val", perturb=None)
    if head == "known":
        tr &= y < k; va &= y < k
    X = feats(P, fname)
    return X, y, tr, va


def fit_one(P, fname, head, C, seed=0, hidden=256):
    k = len(C)
    X, y, tr, va = train_sets(P, fname, head, C)
    m = MLP(X.shape[1], k + (1 if head == "unk" else 0), hidden=hidden, seed=seed).fit(X[tr], y[tr], X[va], y[va])
    return m


def evaluate_val(P, m, fname, head, C):
    k = len(C)
    X = feats(P, fname); y = labels(P, C)
    va_all = mask(P, split="val", perturb=None)
    vk = va_all & (y < k) & mask(P, clean=False)
    vu = va_all & (y == k)
    nonbw = np.array([r.get("kind") != "film_emulation_bw" for r in P["meta"]])
    Z = m.logits(X)
    T = fit_T(Z[va_all if head == "unk" else va_all & (y < k)], y[va_all if head == "unk" else va_all & (y < k)])
    maha = H = None
    if head == "known":
        trk = mask(P, split="train", perturb=None) & (y < k)
        maha = Maha().fit(m.penult(X[trk]), y[trk], k); H = m.penult(X)
    S = scores(head, Z, T, k, maha, H)
    Pk = known_probs(Z, T, k)
    top1 = float(topk(Pk[vk], y[vk]).mean())
    res = {}
    for sname, s in S.items():
        res[sname] = {"val_top1": top1, "val_auroc_nonbw": auroc(s[vk], s[vu & nonbw]), "val_auroc_all": auroc(s[vk], s[vu]), "T": T,
                      "val_ece": ece(Pk[vk], y[vk])}
    return res


def cmd_select(a):
    P = load("pairs"); C = classes(); k = len(C)
    y = labels(P, C)
    va = mask(P, split="val", perturb=None, clean=False) & (y < k)
    vu = mask(P, split="val", perturb=None) & (y == k)
    nonbw = np.array([r.get("kind") != "film_emulation_bw" for r in P["meta"]])
    eng = {"val_top1": float((P["engine_d"][va].argmin(1) == y[va]).mean())}
    for sname, s in engine_scores(P).items():
        eng[f"val_auroc_nonbw[{sname}]"] = auroc(s[va], s[vu & nonbw])
    print("engine", eng, flush=True)
    rows = []
    for fname in DEPLOYABLE + ABLATION:
        for head in ("unk", "known"):
            m = fit_one(P, fname, head, C)
            for sname, r in evaluate_val(P, m, fname, head, C).items():
                r.update({"features": fname, "head": head, "score": sname, "deployable": fname in DEPLOYABLE})
                rows.append(r)
                print(f"{fname:14s} {head:5s} {sname:13s} top1 {r['val_top1']:.3f} auroc(nonBW) {r['val_auroc_nonbw']:.3f} all {r['val_auroc_all']:.3f} ece {r['val_ece']:.3f}", flush=True)
    dep = [r for r in rows if r["deployable"]]
    best = max(dep, key=lambda r: r["val_top1"] + r["val_auroc_nonbw"])
    print("selected", best, flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "select.json").write_text(json.dumps({"engine_val": eng, "candidates": rows, "selected": best}, indent=1))


# ------------------------------------------------------------------------------------------------ final model
class Ensemble:
    def __init__(self, models, T, head, k, score, thr=None, maha=None):
        self.models, self.T, self.head, self.k, self.score, self.thr, self.maha = models, T, head, k, score, thr, maha

    def logits(self, X):
        return np.mean([m.logits(X) for m in self.models], 0)

    def penult(self, X):
        return np.concatenate([m.penult(X) for m in self.models], 1)

    def known_score(self, X):
        Z = self.logits(X)
        H = self.penult(X) if self.maha is not None else None
        return scores(self.head, Z, self.T, self.k, self.maha, H)[self.score], Z

    def decide(self, X, chroma):
        """-> (known-look probabilities, known score, abstain flag)."""
        s, Z = self.known_score(X)
        Pk = known_probs(Z, self.T, self.k)
        ab = (s < self.thr) | (chroma < MIN_CHROMA)
        if self.head == "unk":
            ab |= Z.argmax(1) == self.k
        return Pk, s, ab


def build_final(P, sel, C, seeds=5):
    k = len(C); fname, head = sel["features"], sel["head"]
    models = [fit_one(P, fname, head, C, seed=s) for s in range(seeds)]
    X = feats(P, fname); y = labels(P, C)
    va_all = mask(P, split="val", perturb=None)
    vsel = va_all if head == "unk" else va_all & (y < k)
    ens = Ensemble(models, 1.0, head, k, sel["score"])
    ens.T = fit_T(ens.logits(X[vsel]), y[vsel])
    if sel["score"] == "mahalanobis":
        trk = mask(P, split="train", perturb=None) & (y < k)
        ens.maha = Maha().fit(ens.penult(X[trk]), y[trk], k)
    vk = va_all & (y < k) & mask(P, clean=False)
    s, _ = ens.known_score(X[vk])
    ens.thr = float(np.quantile(s, 1 - TARGET_ACCEPT))
    return ens


def export(ens: Ensemble, C, fname, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for i, m in enumerate(ens.models):
        for kk, v in m.export().items():
            arrs[f"m{i}_{kk}"] = v
    meta = {"classes": C + (["unknown"] if ens.head == "unk" else []), "features": fname, "head": ens.head, "score": ens.score, "T": ens.T,
            "threshold": ens.thr, "min_chroma_p50": MIN_CHROMA, "n_models": len(ens.models)}
    np.savez(path, meta=json.dumps(meta), **arrs)
    return meta


def np_forward(npz, x):
    """Pure-numpy inference: what the website would run (x = one feature vector)."""
    meta = json.loads(str(npz["meta"]))
    Z = 0
    for i in range(meta["n_models"]):
        g = lambda k: npz[f"m{i}_{k}"]  # noqa: E731
        h = (x - g("mu")) / g("sd")
        for j in (1, 2):
            h = h @ g(f"W{j}") + g(f"b{j}")
            h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))  # tanh-GELU (max abs diff to erf-GELU ~1e-3)
        Z = Z + (h @ g("W3") + g("b3"))
    return Z / meta["n_models"], meta


def cmd_test(a):
    sel = json.loads((REPORTS / "select.json").read_text())["selected"]
    P = load("pairs"); C = classes(); k = len(C); fname = sel["features"]
    ens = build_final(P, sel, C)
    meta_art = export(ens, C, fname, DIST / "look_openset.npz")
    rep = {"selected": sel, "artefact": meta_art}
    y = labels(P, C)
    nonbw_of = lambda g: np.array([r.get("kind") != "film_emulation_bw" for r in g["meta"]])  # noqa: E731
    src = np.array([r.get("source_id") for r in P["meta"]])

    def chroma(g):
        return g["desc"][:, CHROMA_P50_IDX]

    # ---------------- test sources
    tk = mask(P, split="test", perturb=None, clean=False) & (y < k)
    tu = mask(P, split="test", perturb=None) & (y == k)
    X = feats(P, fname)
    Pk, s, ab = ens.decide(X, chroma(P))
    es = engine_scores(P)
    eng_pred = P["engine_d"].argmin(1)
    nb = nonbw_of(P)
    kinds = np.array([r.get("kind") for r in P["meta"]])

    def block(mk, mu, Pk, s, ab, eng_pred, es, eng_unc, groups, y, kinds, nb):
        c1 = topk(Pk[mk], y[mk]); e1 = eng_pred[mk] == y[mk]
        diff = c1.astype(float) - e1.astype(float)
        out = {"n_known": int(mk.sum()), "n_unknown": int(mu.sum()),
               "candidate": {"top1": float(c1.mean()), "top1_ci": boot_ci(c1.astype(float), groups[mk]), "top3": float(topk(Pk[mk], y[mk], 3).mean()),
                             "ece": ece(Pk[mk], y[mk]), "abstain_known": float(ab[mk].mean()), "abstain_unknown": float(ab[mu].mean()) if mu.any() else None},
               "engine": {"top1": float(e1.mean()), "top1_ci": boot_ci(e1.astype(float), groups[mk]), "uncertain_known": float(eng_unc[mk].mean()), "uncertain_unknown": float(eng_unc[mu].mean()) if mu.any() else None},
               "paired_diff_top1": {"mean": float(diff.mean()), "ci": boot_ci(diff, groups[mk])}}
        if mu.any():
            au = {}
            for name, sel_u in {"nonbw": mu & nb, "all": mu, "film_emulation_colour": mu & (kinds == "film_emulation_colour"), "creative": mu & (kinds == "creative"),
                                "film_emulation_bw": mu & (kinds == "film_emulation_bw")}.items():
                if sel_u.any():
                    au[name] = {"candidate": auroc(s[mk], s[sel_u]), **{f"engine[{n}]": auroc(v[mk], v[sel_u]) for n, v in es.items()},
                                "n": int(sel_u.sum())}
            out["unknown_auroc"] = au
            corr = np.zeros(len(y), bool); corr[mk] = c1
            m_all = mk | (mu & nb)
            out["accuracy_coverage_nonbw"] = {"candidate": acc_cov(s[m_all], corr[m_all], (y == k)[m_all])}
            ecorr = np.zeros(len(y), bool); ecorr[mk] = e1
            best_e = max(es, key=lambda n: auroc(es[n][mk], es[n][mu & nb]))
            out["accuracy_coverage_nonbw"][f"engine[{best_e}]"] = acc_cov(es[best_e][m_all], ecorr[m_all], (y == k)[m_all])
            acc_cand = (corr & ~ab)[m_all].sum() / max((~ab)[m_all].sum(), 1)
            acc_eng = (ecorr & ~eng_unc)[m_all].sum() / max((~eng_unc)[m_all].sum(), 1)
            out["operating_point_nonbw"] = {"candidate": {"coverage": float((~ab)[m_all].mean()), "accuracy": float(acc_cand)},
                                            "engine_uncertain_rule": {"coverage": float((~eng_unc)[m_all].mean()), "accuracy": float(acc_eng)}}
        return out

    P_d = P["engine_d"]
    eng_top3 = lambda D, yy, mk: float(np.mean((np.argsort(D[mk], 1)[:, :3] == yy[mk][:, None]).any(1)))  # noqa: E731
    rep["test_sources"] = block(tk, tu, Pk, s, ab, eng_pred, es, P["eng_unc"], src, y, kinds, nb)
    rep["test_sources"]["engine"]["top3"] = eng_top3(P_d, y, tk)
    # robustness
    rob = {}
    base_c = topk(Pk[tk], y[tk]).mean(); base_e = (eng_pred[tk] == y[tk]).mean()
    for p in ("jpeg50", "half", "crop60", "wb+8", "wb-8", "ev+1", "ev-1"):
        mp = mask(P, split="test", perturb=p, clean=False) & (y < k)
        mup = mask(P, split="test", perturb=p) & (y == k) & nb
        c = topk(Pk[mp], y[mp]).mean(); e = (eng_pred[mp] == y[mp]).mean()
        rob[p] = {"candidate_top1": float(c), "candidate_delta": float(c - base_c), "engine_top1": float(e), "engine_delta": float(e - base_e),
                  "candidate_auroc_nonbw": auroc(s[mp], s[mup]), "engine_auroc_nonbw[-best_distance]": auroc(es["-best_distance"][mp], es["-best_distance"][mup]),
                  "pass": bool(c - base_c >= e - base_e - 0.03)}
    rep["robustness_test_sources"] = rob
    # ---------------- unseen pool
    G = load("pool"); yg = labels(G, C); Xg = feats(G, fname)
    Pg, sg, abg = ens.decide(Xg, chroma(G))
    gk = yg < k; gu = yg == k
    scene = np.array([r["scene_id"] for r in G["meta"]])
    kg = np.array([r.get("kind") for r in G["meta"]]); nbg = nonbw_of(G)
    rep["unseen_pool"] = block(gk, gu, Pg, sg, abg, G["engine_d"].argmin(1), engine_scores(G), G["eng_unc"], scene, yg, kg, nbg)
    rep["unseen_pool"]["engine"]["top3"] = eng_top3(G["engine_d"], yg, gk)
    rep["unseen_pool"]["n_scenes"] = int(len(np.unique(scene)))
    # ---------------- edited uploads
    E = load("edited"); se, _ = ens.known_score(feats(E, fname)); _, _, abe = ens.decide(feats(E, fname), chroma(E))
    rep["edited_uploads"] = {"n": len(E["meta"]), "candidate_auroc_vs_test_known": auroc(s[tk], se), "candidate_abstain": float(abe.mean()),
                             "engine_uncertain": float(E["eng_unc"].mean()),
                             **{f"engine_auroc[{n}]": auroc(es[n][tk], v) for n, v in engine_scores(E).items()}}
    # ---------------- real photos
    R = load("real"); Xr = feats(R, fname)
    Pr, sr, abr = ens.decide(Xr, chroma(R))
    pr = np.array([r["perturb"] for r in R["meta"]], dtype=object)
    path = np.array([r["path"] for r in R["meta"]]); lab = np.array([r["label"] for r in R["meta"]]); col = np.array([r["colour"] for r in R["meta"]])
    base = pr == None  # noqa: E711
    top_c = Pr.argmax(1); top_e = R["engine_d"].argmin(1); unc_e = R["eng_unc"]
    real = {}
    for name, sub in {"colour_all": base & col, "colour_film": base & col & (lab == 1), "colour_digital": base & col & (lab == 0), "all_rows": base}.items():
        def dist(top, abst):
            kept = top[sub & ~abst]
            cnt = Counter(C[i] for i in kept)
            return {"abstain_rate": float(abst[sub].mean()), "n": int(sub.sum()), "n_answered": int(len(kept)),
                    "max_look_share": float(max(cnt.values()) / len(kept)) if len(kept) else None,
                    "distribution": {c: round(v / max(len(kept), 1), 3) for c, v in cnt.most_common()}}
        real[name] = {"candidate": dist(top_c, abr), "engine": dist(top_e, unc_e)}
    idx_base = {p: i for i, p in enumerate(path) if base[i]}
    stab = {}
    for p in ("jpeg50", "half", "crop60", "wb+8", "wb-8", "ev+1", "ev-1"):
        ii = np.where((pr == p) & col)[0]; jj = np.array([idx_base[path[i]] for i in ii])
        both_c = ~abr[ii] & ~abr[jj]; both_e = ~unc_e[ii] & ~unc_e[jj]
        stab[p] = {"candidate_top1_agree": float((top_c[ii] == top_c[jj])[both_c].mean()) if both_c.any() else None,
                   "candidate_abstain_agree": float((abr[ii] == abr[jj]).mean()), "candidate_abstain_rate": float(abr[ii].mean()),
                   "engine_top1_agree": float((top_e[ii] == top_e[jj])[both_e].mean()) if both_e.any() else None,
                   "engine_uncertain_agree": float((unc_e[ii] == unc_e[jj]).mean())}
    real["stability_colour"] = stab
    rep["real_photos"] = real
    # ---------------- size and latency
    npz = np.load(DIST / "look_openset.npz")
    size_mb = (DIST / "look_openset.npz").stat().st_size / 1e6
    from threadpoolctl import threadpool_limits
    xs = Xr[base][:200].astype(np.float32)
    with threadpool_limits(1):
        Zt, _ = np_forward(npz, xs[:5])
        t0 = time.perf_counter()
        for x in xs:
            np_forward(npz, x[None])
        lat = 1000 * (time.perf_counter() - t0) / len(xs)
    ref = ens.logits(xs[:5])
    rep["artefact_check"] = {"size_mb": size_mb, "latency_ms_single_thread": lat, "numpy_vs_torch_max_abs_logit_diff": float(np.abs(Zt - ref).max())}
    # ---------------- gate
    ts, pool = rep["test_sources"], rep["unseen_pool"]
    au = ts["unknown_auroc"]["nonbw"]
    eng_au = max(v for kk, v in au.items() if kk.startswith("engine"))
    rc = real["colour_all"]["candidate"]
    gate = {"1_top1_ci_above_engine_test_sources": ts["paired_diff_top1"]["ci"][0] > 0,
            "1_top1_ci_above_engine_unseen_pool": pool["paired_diff_top1"]["ci"][0] > 0,
            "2_unknown_auroc_ge_0.80_and_above_engine": bool(au["candidate"] >= 0.80 and au["candidate"] > eng_au),
            "3_ece_le_0.05": ts["candidate"]["ece"] <= 0.05,
            "4_robustness_within_0.03_of_engine": all(v["pass"] for v in rob.values()),
            "5_real_abstention_20_90": 0.20 <= rc["abstain_rate"] <= 0.90,
            "5_real_no_look_over_50pct": rc["max_look_share"] is not None and rc["max_look_share"] <= 0.50,
            "6_size_lt_10MB": size_mb < 10, "6_latency_lt_50ms": lat < 50}
    gate = {kk: bool(v) for kk, v in gate.items()}
    rep["gate"] = gate; rep["decision"] = "SHIP" if all(gate.values()) else "DO NOT SHIP"
    (REPORTS / "test.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"test_sources": {kk: ts[kk] for kk in ("candidate", "engine", "paired_diff_top1")}, "auroc": ts["unknown_auroc"],
                      "pool": {kk: pool[kk] for kk in ("candidate", "engine", "paired_diff_top1")}, "pool_auroc": pool.get("unknown_auroc"),
                      "robust": rob, "real": {kk: real[kk] for kk in ("colour_all", "colour_film", "colour_digital")}, "stab": stab,
                      "edited": rep["edited_uploads"], "artefact": rep["artefact_check"], "gate": gate, "decision": rep["decision"]}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "test"])
    a = ap.parse_args()
    {"select": cmd_select, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
