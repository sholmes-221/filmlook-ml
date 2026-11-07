#!/usr/bin/env python
"""E16: texture-neutralised DINOv2 features: kill grain and sharpening before encoding, keep colour, tone, layout.

  python scripts/texture_neutral_dino.py    # -> reports/texture_neutral_dino.json

Each standardised image is downsampled (area) to a short side of S px and upsampled back (bicubic) before DINOv2
L4+L8 statistics are taken. Variants: S in {64, 96, 128}; "none" is the unmodified control. Representations:
tn_dino_S, tn_hybrid_S (= colour/tone explicit 16-d ++ tn_dino_S). Metrics as in E15: six pairwise treated-vs-untreated
AUROCs with CIs; BlueNeg same-stock (raw and family SupCon) with roll CI; BlueNeg-vs-digital from the filmsynth probe
(expected to fall toward chance once texture is gone); own-source rank-1 for content leakage.
"""
import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from multiprocessing import Pool

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from colour_vs_texture import IDX_C  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

SIZES = (64, 96, 128)


def neutralise(im: Image.Image, s: int) -> Image.Image:
    f = s / min(im.size)
    small = im.resize((max(8, round(im.width * f)), max(8, round(im.height * f))), Image.BOX)
    return small.resize(im.size, Image.BICUBIC)


def _load(args):
    return V3._load(args)


def main():
    t0 = time.time()
    dev = torch.device("cuda")
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
    dino = X.Dino(dev)

    def featurize(paths, keys, real=False):
        with Pool(16) as pool:
            arrs = pool.map(_load, [(p, k, True, real) for p, k in zip(paths, keys)], chunksize=8)
        ims = [Image.fromarray(a) for a in arrs]
        ex = np.stack([X.explicit_of(im) for im in ims])
        out = {"colour_tone": ex[:, IDX_C]}
        dg, ds = dino(ims); out["dino_none"] = ds; out["hybrid_none"] = np.hstack([ex, ds]); out["global_none"] = dg
        for s in SIZES:
            tn = [neutralise(im, s) for im in ims]
            dg_s, ds_s = dino(tn); out[f"tn_dino_{s}"] = ds_s; out[f"tn_hybrid_{s}"] = np.hstack([ex[:, IDX_C], ds_s]); out[f"tn_global_{s}"] = dg_s
        return out
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
    names = ["colour_tone", "dino_none", "hybrid_none"] + [f"tn_dino_{s}" for s in SIZES] + [f"tn_hybrid_{s}" for s in SIZES]
    res = {}
    for name in names:
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
        print(name, json.dumps({k: res[name][k] for k in ("cross_type_mean", "blueneg_vs_digital_auroc")}), res[name]["same_stock_supcon"]["precision@5"], file=sys.stderr)
    rep = {"sizes": SIZES, "results": res, "seconds": round(time.time() - t0, 1)}
    best = max((n for n in names if n.startswith("tn_hybrid")), key=lambda n: res[n]["cross_type_mean"])
    rep["best_tn_hybrid"] = {"name": best, "cross_type_mean": res[best]["cross_type_mean"], "same_stock_supcon": res[best]["same_stock_supcon"], "pairwise": res[best]["pairwise_logo"]}
    (P.REPORTS / "texture_neutral_dino.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({n: {"cross_type_mean": v["cross_type_mean"], "ss_supcon": v["same_stock_supcon"]["precision@5"], "ci": v["same_stock_supcon"]["ci95"], "blueneg": v["blueneg_vs_digital_auroc"], "rank1": v["own_source_rank1_sdxl_test"]} for n, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
