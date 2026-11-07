#!/usr/bin/env python
"""Three-generator leakage benchmark v4: filmsynth v2 (grain randomised), SDXL img2img, Z-Image-Turbo img2img.

  python scripts/synth_experiment_v4.py      # -> reports/synth_experiment_v4.json

All images standardised (384 px short side, seeded JPEG q80-95); diffusion controls are the VAE round-trips;
filmsynth control is identity_v2. Hybrid features (explicit + DINOv2 intermediate) and explicit alone.
Reported, each with a source-level bootstrap 95 % CI where applicable:
  pairwise LOGO     train on generator A, test on B, all six ordered pairs (treated vs untreated);
  held-out LOGO     train on two generators, test on the third (three cells);
  generator identity  4-way (filmsynth / SDXL / Z-Image treated, controls pooled), chance stated;
  content leakage   own-source rank-1 of treated images among the untreated sources;
  cross-photo style retrieval  SupCon projection trained on the two training generators of each held-out cell:
                    treated-precision@5 among other sources of the held-out generator, and BlueNeg same-stock p@5;
  real film         BlueNeg vs CommonCatalog digital AUROC from each held-out-cell probe.
Baseline for comparison: v3 cell C3 (filmsynth v2 <-> SDXL only).
"""
import json
import sys
import time
from collections import Counter

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

GENS = ("filmsynth", "sdxl", "zimage")


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    fs = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    zim = read_jsonl(P.DATA / "synthetic" / "zimage_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx} & {a["source_image_id"] for a in zim})
    sp = {s: by_id[s]["split"] for s in srcs}
    sets = {}
    sets["filmsynth"] = ([a for a in fs if a["source_image_id"] in sp and a["augmentation_family"] != "identity_v2"],
                         [a for a in fs if a["source_image_id"] in sp and a["augmentation_family"] == "identity_v2"])
    for g, rows in (("sdxl", sdx), ("zimage", zim)):
        sets[g] = ([a for a in rows if a["source_image_id"] in sp and a.get("treated", True)],
                   [a for a in rows if a["source_image_id"] in sp and a.get("treated", True) is False])
    for g, (t, c) in sets.items():
        assert len(c) == len(srcs), f"{g}: {len(c)} controls for {len(srcs)} sources"
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    dig = [r for r in eligible if r["image_id"] not in set(srcs) and r.get("medium") in ("digital_camera", "smartphone") and r["split"] in ("val", "test")]
    rep = {"sources": len(srcs), "splits": dict(Counter(sp.values())), "n": {g: {"treated": len(t), "control": len(c)} for g, (t, c) in sets.items()},
           "n_blueneg": len(bn), "n_digital": len(dig), "standardised": True}
    feats = V3.Feats(dev)
    F = {}
    for g, (t, c) in sets.items():
        F[g] = {"t": feats([str(P.ROOT / a["path"]) for a in t], [a["variant_id"] for a in t], True),
                "c": feats([str(P.ROOT / a["path"]) for a in c], [a["variant_id"] for a in c], True)}
    Fo = feats([by_id[s]["path"] for s in srcs], srcs, True, real=True)
    Fb = feats([r["path"] for r in bn], [r["image_id"] for r in bn], True, real=True)
    Fd = feats([r["path"] for r in dig], [r["image_id"] for r in dig], True, real=True)
    meta = {g: {"t": (np.array([a["source_image_id"] for a in t]), np.array([sp[a["source_image_id"]] for a in t])),
                "c": (np.array([a["source_image_id"] for a in c]), np.array([sp[a["source_image_id"]] for a in c]))} for g, (t, c) in sets.items()}
    rep["feature_seconds"] = round(time.time() - t0, 1)

    def xy(g, name, split):
        (st, pt), (sc_, pc) = meta[g]["t"], meta[g]["c"]
        Xg = np.vstack([F[g]["t"][name][pt == split], F[g]["c"][name][pc == split]])
        yg = np.r_[np.ones((pt == split).sum()), np.zeros((pc == split).sum())].astype(int)
        sg = np.r_[st[pt == split], sc_[pc == split]]
        return Xg, yg, sg

    out = {"pairwise_logo": {}, "heldout_logo": {}, "generator_identity": {}, "content_leakage": {}, "retrieval": {}, "real_film": {}}
    intact = np.array([r["partition"] == "blue-intact" for r in bn])
    stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
    for name in ("hybrid_style", "explicit"):
        pw = {}
        for a in GENS:
            for b in GENS:
                if a == b:
                    continue
                Xtr, ytr, _ = xy(a, name, "train"); Xte, yte, ste = xy(b, name, "test")
                pw[f"{a}->{b}"], _ = V3.probe_auroc(Xtr, ytr, Xte, yte, ste)
        out["pairwise_logo"][name] = pw
        ho = {}
        for held in GENS:
            others = [g for g in GENS if g != held]
            Xtr = np.vstack([xy(g, name, "train")[0] for g in others]); ytr = np.concatenate([xy(g, name, "train")[1] for g in others])
            Xte, yte, ste = xy(held, name, "test")
            r_, (sc, m) = V3.probe_auroc(Xtr, ytr, Xte, yte, ste)
            from sklearn.metrics import roc_auc_score
            pb = m.predict_proba(sc.transform(Fb[name]))[:, 1]; pd = m.predict_proba(sc.transform(Fd[name]))[:, 1]
            ho[f"heldout_{held}"] = {**r_, "train_on": others,
                                    "real_film": {"blueneg_vs_digital_auroc": round(float(roc_auc_score(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd])), 4),
                                                  "blue_intact_vs_digital_auroc": round(float(roc_auc_score(np.r_[np.ones(intact.sum()), np.zeros(len(dig))], np.r_[pb[intact], pd])), 4)}}
        out["heldout_logo"][name] = ho
        # 4-way generator identity (treated of each generator + pooled controls)
        Xg = np.vstack([F[g]["t"][name] for g in GENS] + [F[g]["c"][name] for g in GENS])
        yg = np.concatenate([np.full(len(F[g]["t"][name]), i) for i, g in enumerate(GENS)] + [np.full(len(F[g]["c"][name]), 3) for g in GENS])
        spg = np.concatenate([meta[g]["t"][1] for g in GENS] + [meta[g]["c"][1] for g in GENS])
        out["generator_identity"][name] = V3.gen_id(Xg, yg, spg, 4)
    # content leakage (dino_global and hybrid)
    for g in GENS:
        st, pt = meta[g]["t"]
        out["content_leakage"][g] = {n: V3.rank1(F[g]["t"][n][pt == "test"], st[pt == "test"], Fo[n], np.array(srcs)) for n in ("dino_global", "hybrid_style")}
    # retrieval with SupCon trained on the two training generators of each held-out cell (labels: generator+treated)
    for held in GENS:
        others = [g for g in GENS if g != held]
        Xtr = np.vstack([F[g]["t"]["hybrid_style"][meta[g]["t"][1] == "train"] for g in others] + [F[g]["c"]["hybrid_style"][meta[g]["c"][1] == "train"] for g in others])
        lab = np.concatenate([np.full((meta[g]["t"][1] == "train").sum(), f"{g}_treated") for g in others] + [np.full((meta[g]["c"][1] == "train").sum(), "control") for g in others])
        proj = X.supcon(Xtr, lab, dev)
        st, pt = meta[held]["t"]; sc_, pc = meta[held]["c"]
        bank_F = np.vstack([F[g]["t"]["hybrid_style"] for g in others] + [F[g]["c"]["hybrid_style"] for g in others] + [F[held]["c"]["hybrid_style"]])
        bank_src = np.concatenate([meta[g]["t"][0] for g in others] + [meta[g]["c"][0] for g in others] + [sc_])
        bank_tr = np.concatenate([np.ones(len(meta[g]["t"][0]), bool) for g in others] + [np.zeros(len(meta[g]["c"][0]), bool) for g in others] + [np.zeros(len(sc_), bool)])
        q = F[held]["t"]["hybrid_style"][pt == "test"]
        out["retrieval"][f"heldout_{held}"] = {"treated_retrieval_hybrid": V3.treated_retrieval(q, st[pt == "test"], bank_F, bank_src, bank_tr),
                                               "treated_retrieval_supcon": V3.treated_retrieval(proj(q), st[pt == "test"], proj(bank_F), bank_src, bank_tr),
                                               "blueneg_same_stock_supcon": V3.same_stock(proj(Fb["hybrid_style"])[intact], stock, roll)}
    out["retrieval"]["blueneg_same_stock_hybrid_raw"] = V3.same_stock(Fb["hybrid_style"][intact], stock, roll)
    rep.update(out)
    h = out["heldout_logo"]["hybrid_style"]
    rep["summary"] = {"pairwise_hybrid": {k: v["auroc"] for k, v in out["pairwise_logo"]["hybrid_style"].items()},
                      "heldout_hybrid": {k: (v["auroc"], v["ci95"], v["real_film"]["blueneg_vs_digital_auroc"]) for k, v in h.items()},
                      "generator_identity_hybrid": out["generator_identity"]["hybrid_style"],
                      "v3_C3_baseline": {"filmsynth->sdxl": 0.5763, "sdxl->filmsynth": 0.4806, "gen_id_3way": 0.7944, "blueneg": 0.6975, "same_stock_supcon": 0.2482}}
    rep["total_seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / "synth_experiment_v4.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"n": rep["n"], "summary": rep["summary"], "content_leakage": rep["content_leakage"], "retrieval": rep["retrieval"], "seconds": rep["total_seconds"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
