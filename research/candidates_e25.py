"""E25 candidates: content residualisation before the style projection.

  python scripts/style_benchmark.py --split val --extra research.candidates_e25:build --candidates hybrid_supcon_family,e25_resid_a10,...

Hypothesis: part of the content and generator signal left in the hybrid features is linearly predictable from
DINOv2-global content. Fit a ridge regression content -> hybrid on TRAIN rows only (filmsynth train variants, their
sources, CommonCatalog train originals: no labels needed), subtract the prediction, and train the usual family SupCon
projection on the residuals. Variants: ridge alpha in {1, 10, 100} (fit on standardised features), and a control
that residualises against a random projection of the same dimension (checks that it is the content that matters).
"""
from __future__ import annotations

import numpy as np

import synth_experiment as X


def _fit_resid(H, C, mask, alpha, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc_h = StandardScaler().fit(H[mask]); sc_c = StandardScaler().fit(C[mask])
    Hs, Cs = sc_h.transform(H), sc_c.transform(C)
    m = Ridge(alpha=alpha).fit(Cs[mask], Hs[mask])
    R = Hs - m.predict(Cs)
    r2 = 1 - ((Hs[mask] - m.predict(Cs[mask])) ** 2).mean() / Hs[mask].var(0).mean()
    return R, float(r2)


def build(F, rows, dev):
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    train = (sp == "train") & np.isin(st, ["filmsynth", "source", "cc_original"])
    fs_tr = (st == "filmsynth") & (sp == "train")
    H, C = F["hybrid"], F["dino_global"]
    out, r2s = {}, {}
    for alpha in (1.0, 10.0, 100.0):
        R, r2 = _fit_resid(H, C, train, alpha); r2s[f"a{int(alpha)}"] = round(r2, 3)
        proj = X.supcon(R[fs_tr], fam[fs_tr], dev)
        out[f"e25_resid_a{int(alpha)}"] = (lambda f, Z=proj(R): Z)
    rng = np.random.default_rng(0); Crand = H @ rng.standard_normal((H.shape[1], C.shape[1])).astype(np.float32) / np.sqrt(H.shape[1])
    R, r2 = _fit_resid(H, Crand, train, 10.0); r2s["random_control_a10"] = round(r2, 3)
    proj = X.supcon(R[fs_tr], fam[fs_tr], dev); out["e25_resid_random_ctrl"] = (lambda f, Z=proj(R): Z)
    print("E25 ridge R^2 of hybrid from content (train):", r2s)
    return out
