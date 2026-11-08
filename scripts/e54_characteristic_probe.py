#!/usr/bin/env python
"""E54: does the frozen representation encode photographic characteristics (colour response, tonal transitions, contrast,
texture, grain) as decodable, consistent directions, independent of the scene?

  python scripts/e54_characteristic_probe.py        # -> reports/e54_characteristic_probe.json  (CPU for the students; one DINOv2 pass)

Data: the 113 commoncatalog_ccby VALIDATION originals (the E49/E50 scenes; never a holdout). Each scene receives seven
single-parameter edits at five magnitudes (the middle magnitude is the identity for the symmetric ones):
  temperature (R/B gain, colour response), saturation, gamma (midtone / tonal transitions), contrast (linear around
  mid-grey), lift (shadow fade), grain (additive Gaussian noise), blur (Gaussian radius; micro-contrast / texture).
Representations (frozen; checkpoints/frozen_session7): the candidate seeds' PCA-32 int8, one E23b seed, ImageNet-init
PCA-64, DINOv2-Small CLS++mean PCA-64 and L4+L8 PCA-64, explicit-28 standardised.
Measures per representation and attribute, all scene-grouped (no scene in both folds):
  decodability   ridge R^2 of the parameter value from the unit embedding (GroupKFold over scenes, 5 folds);
  monotonicity   mean Spearman between the parameter and the projection on the attribute direction fitted on the other scenes;
  consistency    mean cosine between the displacement vectors (max magnitude minus identity) of different scenes, against
                 the same statistic for pairs of different attributes (null);
  sensitivity    mean cosine distance at the largest magnitude relative to the centre-crop-80 displacement;
and, over all attributes, the accuracy of a scene-grouped logistic classifier that names the attribute from the
displacement vector (chance 1/7).
Pre-registered rule (research/backlog.yaml E54): supported iff for every candidate seed at least five of the seven
attributes reach decodability R^2 >= 0.5 and the attribute classifier reaches accuracy >= 0.7. DINOv2 and ImageNet-init
are expected to decode the edits as well (they are strong encoders); the claim about the candidate is conditional on
its content invariance already established (E48-E50), which this probe does not re-measure.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import FIT  # noqa: E402
from e53_fresh_holdout import perturb  # noqa: E402
from freeze_candidate import CANDIDATE, Frozen  # noqa: E402
from style_benchmark import unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

OUT = P.REPORTS / "e54_characteristic_probe.json"
ATTR = {"temperature": [-0.15, -0.075, 0.0, 0.075, 0.15], "saturation": [0.4, 0.7, 1.0, 1.3, 1.6], "gamma": [0.6, 0.8, 1.0, 1.25, 1.6], "contrast": [-0.4, -0.2, 0.0, 0.2, 0.4],
        "lift": [0.0, 0.05, 0.1, 0.15, 0.2], "grain": [0.0, 4.0, 8.0, 12.0, 16.0], "blur": [0.0, 0.5, 1.0, 1.5, 2.0]}
IDENTITY = {"temperature": 2, "saturation": 2, "gamma": 2, "contrast": 2, "lift": 0, "grain": 0, "blur": 0}
REPS = [f"{t}:pca32_int8" for t in CANDIDATE] + ["style_fs_v2_real_f576:pca32_int8", "imagenet_init:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std"]


def edit(a, attr, v, seed=0):
    from PIL import Image, ImageFilter
    x = a.astype(np.float32) / 255.0
    if attr == "temperature":
        x = x * np.array([1 + v, 1.0, 1 - v], np.float32)
    elif attr == "saturation":
        g = x @ np.array([0.299, 0.587, 0.114], np.float32); x = g[..., None] + v * (x - g[..., None])
    elif attr == "gamma":
        x = np.clip(x, 0, 1) ** v
    elif attr == "contrast":
        x = 0.5 + (x - 0.5) * (1 + v)
    elif attr == "lift":
        x = v + (1 - v) * x
    elif attr == "grain":
        x = x + np.random.default_rng(seed).normal(0, v / 255.0, x.shape).astype(np.float32)
    elif attr == "blur":
        if v > 0:
            return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(v)))
        return a
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def grouped_r2(E, y, g):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(E, y, g):
        pred[te] = Ridge(alpha=1.0).fit(E[tr], y[tr]).predict(E[te])
    return round(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)


def grouped_acc(D, y, g):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(D, y, g):
        sc = StandardScaler().fit(D[tr]); pred[te] = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(D[tr]), y[tr]).predict(sc.transform(D[te]))
    return round(float((pred == y).mean()), 4)


def monotonicity(E, param, scene, vals):
    """Projection on the attribute direction fitted on the other scenes (mean of e(max) - e(identity)); Spearman per scene."""
    from scipy.stats import spearmanr
    rho = []
    for s in np.unique(scene):
        o = scene != s; d = (E[o & (param == vals[-1])].mean(0) - E[o & (param == vals[0])].mean(0)); d /= np.linalg.norm(d) + 1e-9
        m = scene == s; r = spearmanr(param[m], E[m] @ d).correlation; rho.append(0.0 if np.isnan(r) else r)
    return round(float(np.mean(rho)), 4)


def main():
    import torch
    from PIL import Image
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = [r for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby"]
    orig = sorted([r for r in man if r["split"] == "val" and r["edit_type"] == "original"], key=lambda r: r["image_id"])
    with Pool(16) as pool:
        base = pool.map(V3._load, [(r["path"], r["image_id"], True, True) for r in orig], chunksize=8)
    items = [("original", 0, i, 0.0) for i in range(len(base))] + [("crop80", 0, i, 0.0) for i in range(len(base))]
    arrays = list(base) + [perturb(a, "crop80") for a in base]
    for attr, vals in ATTR.items():
        for k, v in enumerate(vals):
            for i, a in enumerate(base):
                items.append((attr, k, i, float(v))); arrays.append(edit(a, attr, v, seed=i))
    kind = np.array([it[0] for it in items]); level = np.array([it[1] for it in items]); scene = np.array([it[2] for it in items]); param = np.array([it[3] for it in items])
    print(f"{len(base)} scenes, {len(arrays)} images", flush=True)
    frozen = {t: Frozen(t) for t in CANDIDATE + ["style_fs_v2_real_f576", "imagenet_init"]}
    reps = {}
    for t, fz in frozen.items():
        F = fz.features(arrays)
        if t == "imagenet_init":
            reps[f"{t}:pca64"] = fz.reduce(F, 64, False)
        else:
            reps[f"{t}:pca32_int8"] = fz.reduce(F, 32, True)
    Fit = dict(np.load(FIT.with_suffix(".npz"))); mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6
    ims = [Image.fromarray(a) for a in arrays]; reps["explicit28_std"] = (np.stack([X.explicit_of(im) for im in ims]).astype(np.float32) - mu) / sd
    dg, dsty = X.Dino(dev)(ims); reps["dino_global:pca64"] = PCA64(Fit["dino_global"], 64)(np.asarray(dg, np.float32)); reps["dino_style:pca64"] = PCA64(Fit["dino_style"], 64)(np.asarray(dsty, np.float32))
    out = {"experiment": "E54_characteristic_probe", "validation_data_only": True, "n_scenes": len(base), "n_images": len(arrays), "attributes": ATTR, "representations": {}}
    for name in REPS:
        E = unit(np.nan_to_num(reps[name].astype(np.float32))); o = kind == "original"; E0 = E[o][np.argsort(scene[o])]
        crop = 1 - (E[kind == "crop80"][np.argsort(scene[kind == "crop80"])] * E0).sum(1); r = {"attributes": {}}
        deltas, dlab, dscene, dirs = [], [], [], {}
        for attr, vals in ATTR.items():
            m = kind == attr; Ea, pa, sa, la = E[m], param[m], scene[m], level[m]
            ident = la == IDENTITY[attr]; Eid = Ea[ident][np.argsort(sa[ident])]
            far = la == (len(vals) - 1 if IDENTITY[attr] == 0 else 0); Efar = Ea[far][np.argsort(sa[far])]; Efar2 = Ea[la == len(vals) - 1][np.argsort(sa[la == len(vals) - 1])]
            D = Efar2 - Eid if IDENTITY[attr] == 0 else np.vstack([Efar2 - Eid, Efar - Eid]); Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            same = (Dn @ Dn.T)[np.triu_indices(len(Dn), 1)].mean()
            sens = float((1 - (Efar2 * Eid).sum(1)).mean()); dirs[attr] = Dn
            for j in range(len(Eid)):
                for lv in range(len(vals)):
                    if lv == IDENTITY[attr]:
                        continue
                    e_lv = Ea[(la == lv) & (sa == j)]
                    if len(e_lv):
                        deltas.append(e_lv[0] - Eid[j]); dlab.append(attr); dscene.append(j)
            r["attributes"][attr] = {"decodability_r2": grouped_r2(Ea, pa, sa), "monotonicity": monotonicity(Ea, pa, sa, vals), "consistency_same": round(float(same), 4),
                                     "sensitivity_vs_crop80": round(float(sens / (crop.mean() + 1e-9)), 3), "displacement_max": round(sens, 4)}
        allD = np.vstack(list(dirs.values())); labs = np.concatenate([[a] * len(d) for a, d in dirs.items()]); C = allD @ allD.T; diff = C[labs[:, None] != labs[None]].mean()
        r["consistency_null_cross_attribute"] = round(float(diff), 4); r["attribute_from_displacement_acc"] = grouped_acc(np.array(deltas), np.array(dlab), np.array(dscene)); r["chance"] = round(1 / len(ATTR), 4)
        r["n_attributes_r2_ge_0.5"] = int(sum(v["decodability_r2"] >= 0.5 for v in r["attributes"].values())); r["crop80_displacement"] = round(float(crop.mean()), 4)
        out["representations"][name] = r
        print(f"{name:40s} r2>=0.5: {r['n_attributes_r2_ge_0.5']}/7  acc {r['attribute_from_displacement_acc']}  " + " ".join(f"{a[:4]} {v['decodability_r2']:.2f}/{v['monotonicity']:.2f}/{v['consistency_same']:.2f}/x{v['sensitivity_vs_crop80']:.1f}" for a, v in r["attributes"].items()), flush=True)
    cands = [out["representations"][f"{t}:pca32_int8"] for t in CANDIDATE]
    out["summary"] = {"supported": int(all(c["n_attributes_r2_ge_0.5"] >= 5 and c["attribute_from_displacement_acc"] >= 0.7 for c in cands)),
                      "candidate_min_attributes_r2": min(c["n_attributes_r2_ge_0.5"] for c in cands), "candidate_min_attr_acc": min(c["attribute_from_displacement_acc"] for c in cands),
                      "candidate_r2_seed_mean": {a: round(float(np.mean([c["attributes"][a]["decodability_r2"] for c in cands])), 4) for a in ATTR},
                      "explicit_attributes_r2": out["representations"]["explicit28_std"]["n_attributes_r2_ge_0.5"], "dino_global_attributes_r2": out["representations"]["dino_global:pca64"]["n_attributes_r2_ge_0.5"],
                      "init_attributes_r2": out["representations"]["imagenet_init:pca64"]["n_attributes_r2_ge_0.5"], "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); OUT.write_text(json.dumps(out, indent=1)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
