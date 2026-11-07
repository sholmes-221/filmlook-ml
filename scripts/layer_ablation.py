#!/usr/bin/env python
"""E9: which DINOv2-small layers carry transferable style? Per-layer mean++std patch statistics (768-d each).

  python scripts/layer_ablation.py    # -> reports/layer_ablation.json

Per layer 1..12 (and the 4+8 pair, and explicit statistics): treated-vs-untreated probe trained on filmsynth v2
(identity_v2 control) tested on SDXL (VAE control), and the reverse, with source bootstrap CIs; BlueNeg vs digital
AUROC from the filmsynth probe; BlueNeg same-stock p@5 with the raw layer features and with a SupCon projection
trained on filmsynth train sources. All images standardised as in v3.
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

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402

import os
COMBOS = os.environ.get("LAYER_COMBOS", "L4+L8").split(",")
OUT_NAME = os.environ.get("LAYER_ABLATION_OUT", "layer_ablation.json")


class LayerDino:
    def __init__(self, dev):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval(); self.dev = dev

    @torch.no_grad()
    def __call__(self, ims):
        per = {f"L{i}": [] for i in range(1, 13)}
        for i in range(0, len(ims), 64):
            px = torch.from_numpy(np.stack([dinov2_pixels(im, 224) for im in ims[i:i + 64]])).to(self.dev)
            hs = self.m(pixel_values=px, output_hidden_states=True).hidden_states
            for li in range(1, 13):
                h = hs[li][:, 1:]
                per[f"L{li}"].append(torch.cat([h.mean(1), h.std(1)], 1).float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in per.items()}


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
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx})
    sp = {s: by_id[s]["split"] for s in srcs}
    fs = [a for a in fs if a["source_image_id"] in sp]
    sdx_t = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True)]
    sdx_c = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True) is False]
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    dig = [r for r in eligible if r["image_id"] not in set(srcs) and r.get("medium") in ("digital_camera", "smartphone") and r["split"] in ("val", "test")]
    ld = LayerDino(dev)

    def feats(paths, keys, real=False):
        with Pool(16) as pool:
            arrs = pool.map(_load, [(p, k, True, real) for p, k in zip(paths, keys)], chunksize=8)
        ims = [Image.fromarray(a) for a in arrs]
        out = ld(ims); out["explicit"] = np.stack([X.explicit_of(im) for im in ims])
        for combo in COMBOS:
            out[combo] = np.hstack([out[c] for c in combo.split("+")])
        return out
    Fs = feats([str(P.ROOT / a["path"]) for a in fs], [a["variant_id"] for a in fs])
    Fx = feats([str(P.ROOT / a["path"]) for a in sdx_t], [a["variant_id"] for a in sdx_t])
    Fc = feats([str(P.ROOT / a["path"]) for a in sdx_c], [a["variant_id"] for a in sdx_c])
    Fb = feats([r["path"] for r in bn], [r["image_id"] for r in bn], real=True)
    Fd = feats([r["path"] for r in dig], [r["image_id"] for r in dig], real=True)
    src_s = np.array([a["source_image_id"] for a in fs]); sp_s = np.array([sp[s] for s in src_s])
    y_s = np.array([a["augmentation_family"] != "identity_v2" for a in fs]).astype(int)
    fam = np.array([a["augmentation_family"] for a in fs])
    src_x = np.array([a["source_image_id"] for a in sdx_t]); sp_x = np.array([sp[s] for s in src_x])
    src_c = np.array([a["source_image_id"] for a in sdx_c]); sp_c = np.array([sp[s] for s in src_c])
    intact = np.array([r["partition"] == "blue-intact" for r in bn])
    stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
    from sklearn.metrics import roc_auc_score
    res = {}
    names = [f"L{i}" for i in range(1, 13)] + COMBOS + ["explicit"] if os.environ.get("LAYER_SINGLES", "1") == "1" else COMBOS + ["explicit"]
    for name in names:
        Xte = np.vstack([Fx[name][sp_x == "test"], Fc[name][sp_c == "test"]]); yte = np.r_[np.ones((sp_x == "test").sum()), np.zeros((sp_c == "test").sum())].astype(int)
        ste = np.r_[src_x[sp_x == "test"], src_c[sp_c == "test"]]
        a, (sc, m) = V3.probe_auroc(Fs[name][sp_s == "train"], y_s[sp_s == "train"], Xte, yte, ste)
        Xtr2 = np.vstack([Fx[name][sp_x == "train"], Fc[name][sp_c == "train"]]); ytr2 = np.r_[np.ones((sp_x == "train").sum()), np.zeros((sp_c == "train").sum())].astype(int)
        b, _ = V3.probe_auroc(Xtr2, ytr2, Fs[name][sp_s == "test"], y_s[sp_s == "test"], src_s[sp_s == "test"])
        pb = m.predict_proba(sc.transform(Fb[name]))[:, 1]; pd = m.predict_proba(sc.transform(Fd[name]))[:, 1]
        real_auc = round(float(roc_auc_score(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd])), 4)
        proj = X.supcon(Fs[name][sp_s == "train"], fam[sp_s == "train"], dev, epochs=300)
        res[name] = {"dim": int(Fs[name].shape[1]), "fs_to_sdxl": a, "sdxl_to_fs": b, "blueneg_auroc": real_auc,
                     "same_stock_raw": V3.same_stock(Fb[name][intact], stock, roll),
                     "same_stock_supcon": V3.same_stock(proj(Fb[name])[intact], stock, roll)}
        print(name, a["auroc"], b["auroc"], real_auc, res[name]["same_stock_supcon"]["precision@5"], file=sys.stderr)
    layers = [k for k in res if k.startswith("L")]
    best_fs = max(layers, key=lambda k: res[k]["fs_to_sdxl"]["auroc"]); best_ss = max(layers, key=lambda k: res[k]["same_stock_supcon"]["precision@5"])
    rep = {"per_layer": res, "best": {"fs_to_sdxl": {"layer": best_fs, **res[best_fs]["fs_to_sdxl"]},
                                       "sdxl_to_fs": {"layer": max(layers, key=lambda k: res[k]["sdxl_to_fs"]["auroc"])},
                                       "same_stock": {"layer": best_ss, **res[best_ss]["same_stock_supcon"]}},
           "reference_L4+L8": {k: res["L4+L8"][k] for k in ("fs_to_sdxl", "sdxl_to_fs", "blueneg_auroc", "same_stock_supcon")} if "L4+L8" in res else None,
           "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / OUT_NAME).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: {"fs->sdxl": v["fs_to_sdxl"]["auroc"], "sdxl->fs": v["sdxl_to_fs"]["auroc"], "blueneg": v["blueneg_auroc"],
                          "same_stock_supcon": v["same_stock_supcon"]["precision@5"], "same_stock_raw": v["same_stock_raw"]["precision@5"]} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
