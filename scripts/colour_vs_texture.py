#!/usr/bin/env python
"""E15: does colour/tone alone carry film-stock identity on real scans, separately from texture?

  python scripts/colour_vs_texture.py    # -> reports/colour_vs_texture.json

Representations: explicit colour/tone statistics only (16), explicit texture statistics only (12), all 28,
DINOv2 L4+L8 statistics, hybrid. For each: BlueNeg same-stock p@5 raw and with family SupCon trained on filmsynth
v2 (roll-bootstrap CI); pairwise treated-vs-untreated transfer across the three generators; BlueNeg-vs-digital
AUROC from the filmsynth probe. Reads: if colour/tone-only retrieval matches the full feature set, stock identity
on these scans is a colour-rendition property (aesthetics) rather than grain (authenticity).
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402

from filmlook_infer.features import FEATURE_NAMES  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

TEXTURE = ["local_contrast_ratio", "lap_var", "hf_ratio", "overshoot", "noise_std", "noise_kurtosis", "noise_shadow_ratio",
           "noise_mid_ratio", "chroma_noise_ratio", "noise_autocorr1", "noise_autocorr2", "blockiness"]
COLOUR_TONE = [n for n in FEATURE_NAMES if n not in TEXTURE]
IDX_C = [FEATURE_NAMES.index(n) for n in COLOUR_TONE]; IDX_T = [FEATURE_NAMES.index(n) for n in TEXTURE]


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
    feats = V3.Feats(dev)

    def expand(F):
        ex = F["explicit"]
        return {"colour_tone": ex[:, IDX_C], "texture": ex[:, IDX_T], "explicit_all": ex, "dino_L4L8": F["hybrid_style"][:, 28:], "hybrid": F["hybrid_style"]}
    Fg = {g: {"t": expand(feats([str(P.ROOT / a["path"]) for a in t], [a["variant_id"] for a in t], True)),
              "c": expand(feats([str(P.ROOT / a["path"]) for a in c], [a["variant_id"] for a in c], True))} for g, (t, c) in gens.items()}
    Fb = expand(feats([r["path"] for r in bn], [r["image_id"] for r in bn], True, real=True))
    Fd = expand(feats([r["path"] for r in dig], [r["image_id"] for r in dig], True, real=True))
    meta = {g: {"t": (np.array([a["source_image_id"] for a in t]), np.array([sp[a["source_image_id"]] for a in t])),
                "c": (np.array([a["source_image_id"] for a in c]), np.array([sp[a["source_image_id"]] for a in c]))} for g, (t, c) in gens.items()}
    fam = np.array([a["augmentation_family"] for a in gens["filmsynth"][0]]); sp_fs = meta["filmsynth"]["t"][1]
    intact = np.array([r["partition"] == "blue-intact" for r in bn]); stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
    from sklearn.metrics import roc_auc_score
    res = {}
    for name in ("colour_tone", "texture", "explicit_all", "dino_L4L8", "hybrid"):
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
        # SupCon on filmsynth families (train sources), treated only (identity has no family) plus identity as its own class
        Xf = np.vstack([Fg["filmsynth"]["t"][name][sp_fs == "train"], Fg["filmsynth"]["c"][name][meta["filmsynth"]["c"][1] == "train"]])
        lf = np.r_[fam[sp_fs == "train"], np.full((meta["filmsynth"]["c"][1] == "train").sum(), "identity_v2")]
        proj = X.supcon(Xf, lf, dev, dim=min(64, Xf.shape[1] * 2))
        res[name] = {"dim": int(Fb[name].shape[1]), "pairwise_logo": {k: (v["auroc"], v["ci95"]) for k, v in pw.items()},
                     "blueneg_vs_digital_auroc_filmsynth_probe": round(float(roc_auc_score(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd])), 4),
                     "same_stock_raw": V3.same_stock(Fb[name][intact], stock, roll), "same_stock_supcon": V3.same_stock(proj(Fb[name])[intact], stock, roll)}
        print(name, json.dumps({k: res[name][k] for k in ("blueneg_vs_digital_auroc_filmsynth_probe", "same_stock_raw", "same_stock_supcon")}), file=sys.stderr)
    rep = {"colour_tone_features": COLOUR_TONE, "texture_features": TEXTURE, "results": res, "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "colour_vs_texture.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: {"same_stock_raw": v["same_stock_raw"]["precision@5"], "same_stock_supcon": v["same_stock_supcon"]["precision@5"], "ci": v["same_stock_supcon"]["ci95"],
                          "blueneg_auroc": v["blueneg_vs_digital_auroc_filmsynth_probe"], "fs->sdxl": v["pairwise_logo"]["filmsynth->sdxl"][0], "sdxl->zimage": v["pairwise_logo"]["sdxl->zimage"][0]} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
