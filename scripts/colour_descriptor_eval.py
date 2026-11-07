#!/usr/bin/env python
"""E17: global colour-rendition descriptor (80-d, no texture) as the generator-agnostic aesthetics representation.

  python scripts/colour_descriptor_eval.py    # -> reports/colour_descriptor_eval.json   (CPU only)

Representations: cr80 (new), colour_tone16 (E15 subset), cr80++colour_tone16, texture12 (reference).
Metrics: six pairwise treated-vs-untreated transfers with source CIs (cross-type mean), BlueNeg same-stock raw and
family-SupCon with roll CI, BlueNeg-vs-digital AUROC from the filmsynth probe, SDXL own-source rank-1.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402

from filmlook_infer.colour_rendition import colour_rendition  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402


def _feat(args):
    arr = V3._load(args)
    im = Image.fromarray(arr)
    return np.r_[colour_rendition(im), X.explicit_of(im)]


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # only for the tiny SupCon projection
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    fs = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    zim = read_jsonl(P.DATA / "synthetic" / "zimage_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx} & {a["source_image_id"] for a in zim})
    sp = {s: by_id[s]["split"] for s in srcs}
    gens = {"filmsynth": ([a for a in fs if a["source_image_id"] in sp and a["augmentation_family"] != "identity_v2"], [a for a in fs if a["source_image_id"] in sp and a["augmentation_family"] == "identity_v2"]),
            "sdxl": ([a for a in sdx if a["source_image_id"] in sp and a.get("treated", True)], [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True) is False]),
            "zimage": ([a for a in zim if a["source_image_id"] in sp and a.get("treated", True)], [a for a in zim if a["source_image_id"] in sp and a.get("treated", True) is False])}
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    dig = [r for r in eligible if r["image_id"] not in set(srcs) and r.get("medium") in ("digital_camera", "smartphone") and r["split"] in ("val", "test")]

    def featurize(paths, keys, real=False):
        with Pool(16) as pool:
            M = np.stack(pool.map(_feat, [(p, k, True, real) for p, k in zip(paths, keys)], chunksize=8))
        cr, ex = M[:, :80], M[:, 80:]
        return {"cr80": cr, "colour_tone16": ex[:, IDX_C], "cr80+ct16": np.hstack([cr, ex[:, IDX_C]]), "texture12": ex[:, IDX_T]}
    Fg = {g: {"t": featurize([str(P.ROOT / a["path"]) for a in t], [a["variant_id"] for a in t]),
              "c": featurize([str(P.ROOT / a["path"]) for a in c], [a["variant_id"] for a in c])} for g, (t, c) in gens.items()}
    Fo = featurize([by_id[s]["path"] for s in srcs], srcs, real=True)
    Fb = featurize([r["path"] for r in bn], [r["image_id"] for r in bn], real=True)
    Fd = featurize([r["path"] for r in dig], [r["image_id"] for r in dig], real=True)
    meta = {g: {"t": (np.array([a["source_image_id"] for a in t]), np.array([sp[a["source_image_id"]] for a in t])),
                "c": (np.array([a["source_image_id"] for a in c]), np.array([sp[a["source_image_id"]] for a in c]))} for g, (t, c) in gens.items()}
    fam = np.array([a["augmentation_family"] for a in gens["filmsynth"][0]]); sp_fs = meta["filmsynth"]["t"][1]
    intact = np.array([r["partition"] == "blue-intact" for r in bn]); stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
    from sklearn.metrics import roc_auc_score
    res = {}
    for name in ("cr80", "colour_tone16", "cr80+ct16", "texture12"):
        def xy(g, split):
            (st, pt), (sc_, pc) = meta[g]["t"], meta[g]["c"]
            return (np.vstack([Fg[g]["t"][name][pt == split], Fg[g]["c"][name][pc == split]]),
                    np.r_[np.ones((pt == split).sum()), np.zeros((pc == split).sum())].astype(int), np.r_[st[pt == split], sc_[pc == split]])
        pw = {}
        for a in gens:
            for b in gens:
                if a != b:
                    Xtr, ytr, _ = xy(a, "train"); pw[f"{a}->{b}"], _ = V3.probe_auroc(Xtr, ytr, *xy(b, "test"))
        Xtr, ytr, _ = xy("filmsynth", "train"); _, (sc, m) = V3.probe_auroc(Xtr, ytr, *xy("sdxl", "test"))
        pb = m.predict_proba(sc.transform(Fb[name]))[:, 1]; pd = m.predict_proba(sc.transform(Fd[name]))[:, 1]
        Xf = np.vstack([Fg["filmsynth"]["t"][name][sp_fs == "train"], Fg["filmsynth"]["c"][name][meta["filmsynth"]["c"][1] == "train"]])
        lf = np.r_[fam[sp_fs == "train"], np.full((meta["filmsynth"]["c"][1] == "train").sum(), "identity_v2")]
        proj = X.supcon(Xf, lf, dev, dim=min(64, Xf.shape[1] * 2))
        te_x = meta["sdxl"]["t"][1] == "test"
        res[name] = {"dim": int(Fb[name].shape[1]), "pairwise_logo": {k: (v["auroc"], v["ci95"]) for k, v in pw.items()},
                     "cross_type_mean": round(float(np.mean([pw[k]["auroc"] for k in ("filmsynth->sdxl", "filmsynth->zimage", "sdxl->filmsynth", "zimage->filmsynth")])), 4),
                     "blueneg_vs_digital_auroc": round(float(roc_auc_score(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd])), 4),
                     "same_stock_raw": V3.same_stock(Fb[name][intact], stock, roll), "same_stock_supcon": V3.same_stock(proj(Fb[name])[intact], stock, roll),
                     "own_source_rank1_sdxl_test": V3.rank1(Fg["sdxl"]["t"][name][te_x], meta["sdxl"]["t"][0][te_x], Fo[name], np.array(srcs))}
    rep = {"results": res, "seconds": round(time.time() - t0, 1),
           "cr80": {"cross_type_mean": res["cr80"]["cross_type_mean"], "same_stock_supcon": res["cr80"]["same_stock_supcon"], "same_stock_raw": res["cr80"]["same_stock_raw"]}}
    (P.REPORTS / "colour_descriptor_eval.json").write_text(json.dumps(rep, indent=1, default=str))
    for k, v in res.items():
        pw = v["pairwise_logo"]
        print(f"{k:14s} xtype {v['cross_type_mean']:.3f} | fs->sdxl {pw['filmsynth->sdxl'][0]:.2f} fs->zim {pw['filmsynth->zimage'][0]:.2f} sdxl->fs {pw['sdxl->filmsynth'][0]:.2f} zim->fs {pw['zimage->filmsynth'][0]:.2f} "
              f"sdxl<->zim {pw['sdxl->zimage'][0]:.2f}/{pw['zimage->sdxl'][0]:.2f} | blueneg {v['blueneg_vs_digital_auroc']:.2f} | ss raw {v['same_stock_raw']['precision@5']:.3f} {v['same_stock_raw']['ci95']} "
              f"supcon {v['same_stock_supcon']['precision@5']:.3f} {v['same_stock_supcon']['ci95']} | rank1 {v['own_source_rank1_sdxl_test']:.2f}")


if __name__ == "__main__":
    main()
