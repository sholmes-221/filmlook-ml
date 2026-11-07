#!/usr/bin/env python
"""E11: style embedding from continuous parameter similarity (soft positives) vs family-label SupCon.

  python scripts/param_similarity_embedding.py    # -> reports/param_similarity_embedding.json

Training data: filmsynth v2 train sources, hybrid features (explicit ++ DINOv2 L4+L8 statistics), standardised
images. Pair weight w_ij = exp(-||p_i - p_j||^2 / tau) over the applied parameter vector (temperature, saturation,
contrast, grain, shadow lift, highlight compression, monochrome flag), excluding same-source pairs so the embedding
cannot use scene identity. Loss: soft-label contrastive (weights normalised per anchor). Baselines: family SupCon
(same recipe, hard positives = same family), raw features. Metrics: BlueNeg same-stock p@5 (real, owner labels,
same roll excluded); cross-generator treated retrieval with SDXL queries (synthetic proxy); own-source rank-1 on
filmsynth test (content leakage); pairwise-parameter Spearman: embedding distance vs parameter distance on test.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

PARAMS = ["temperature_shift", "saturation_adjustment", "contrast_adjustment", "grain_strength", "shadow_lift", "highlight_compression"]


def pvec(a):
    p = a["parameters"]
    mono = 1.0 if p.get("monochrome") else 0.0
    v = [float(p[k]) for k in PARAMS]
    if mono:
        v[0] = 0.0; v[1] = -1.0                       # monochrome: no temperature; saturation at the floor
    return np.array(v + [mono], dtype=np.float32)


def soft_supcon(Xtr, Pv, src, dev, tau_p=0.35, dim=64, epochs=400, temp=0.1, seed=0):
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    Pt = torch.tensor(Pv, device=dev)
    D2 = ((Pt[:, None] - Pt[None]) ** 2).sum(-1)
    W = torch.exp(-D2 / tau_p)
    same = torch.tensor(src[:, None] == src[None], device=dev)
    W = W.masked_fill(same, 0.0)                    # never a positive with the same scene
    eye = torch.eye(len(Xt), device=dev, dtype=torch.bool)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dim)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        z = torch.nn.functional.normalize(net(Xt + 0.05 * torch.randn_like(Xt)), dim=1)
        logits = (z @ z.T / temp).masked_fill(eye | same, -1e9)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -(W * logp).sum(1) / W.sum(1).clamp(min=1e-6)
        opt.zero_grad(); loss.mean().backward(); opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor(sc.transform(Xn), dtype=torch.float32, device=dev)).detach().cpu().numpy()


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    fs = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx})
    sp = {s: by_id[s]["split"] for s in srcs}
    fs = [a for a in fs if a["source_image_id"] in sp]
    sdx_t = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True)]
    sdx_c = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True) is False]
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    feats = V3.Feats(dev)
    Fs = feats([str(P.ROOT / a["path"]) for a in fs], [a["variant_id"] for a in fs], True)["hybrid_style"]
    Fx = feats([str(P.ROOT / a["path"]) for a in sdx_t], [a["variant_id"] for a in sdx_t], True)["hybrid_style"]
    Fc = feats([str(P.ROOT / a["path"]) for a in sdx_c], [a["variant_id"] for a in sdx_c], True)["hybrid_style"]
    Fo = feats([by_id[s]["path"] for s in srcs], srcs, True, real=True)["hybrid_style"]
    Fb = feats([r["path"] for r in bn], [r["image_id"] for r in bn], True, real=True)["hybrid_style"]
    src_s = np.array([a["source_image_id"] for a in fs]); sp_s = np.array([sp[s] for s in src_s])
    fam = np.array([a["augmentation_family"] for a in fs]); Pv = np.stack([pvec(a) for a in fs])
    treated_s = fam != "identity_v2"
    src_x = np.array([a["source_image_id"] for a in sdx_t]); sp_x = np.array([sp[s] for s in src_x]); src_c = np.array([a["source_image_id"] for a in sdx_c])
    intact = np.array([r["partition"] == "blue-intact" for r in bn])
    stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
    tr = sp_s == "train"; te = sp_s == "test"
    from scipy.stats import spearmanr
    embs = {"raw_hybrid": lambda Z: Z,
            "family_supcon": X.supcon(Fs[tr], fam[tr], dev)}
    for tau in (0.2, 0.35, 0.7):
        embs[f"param_soft_tau{tau}"] = soft_supcon(Fs[tr], Pv[tr], src_s[tr], dev, tau_p=tau)
    res = {}
    for name, f in embs.items():
        Zs, Zx, Zc, Zo, Zb = f(Fs), f(Fx), f(Fc), f(Fo), f(Fb)
        bank = np.vstack([Zs, Zc]); bsrc = np.r_[src_s, src_c]; btr = np.r_[treated_s, np.zeros(len(src_c), bool)]
        te_x = sp_x == "test"
        # parameter-distance correlation on test variants (pairs from different sources)
        ii, jj = np.triu_indices(te.sum(), 1)
        Zt = Zs[te] / (np.linalg.norm(Zs[te], axis=1, keepdims=True) + 1e-8); st = src_s[te]
        keep = st[ii] != st[jj]
        ed = 1 - (Zt[ii[keep]] * Zt[jj[keep]]).sum(1); pd_ = np.linalg.norm(Pv[te][ii[keep]] - Pv[te][jj[keep]], axis=1)
        res[name] = {"blueneg_same_stock": V3.same_stock(Zb[intact], stock, roll),
                     "xgen_treated_retrieval_sdxl_query": V3.treated_retrieval(Zx[te_x], src_x[te_x], bank, bsrc, btr),
                     "own_source_rank1_filmsynth_test": V3.rank1(Zs[te & treated_s], src_s[te & treated_s], Zo, np.array(srcs)),
                     "spearman_embdist_vs_paramdist_test": round(float(spearmanr(ed, pd_).statistic), 4)}
        print(name, json.dumps(res[name]), file=sys.stderr)
    best = max((k for k in res if k.startswith("param")), key=lambda k: res[k]["blueneg_same_stock"]["precision@5"])
    rep = {"results": res, "best_param": {"name": best, **res[best]["blueneg_same_stock"]},
           "family_supcon_same_stock": res["family_supcon"]["blueneg_same_stock"], "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "param_similarity_embedding.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
