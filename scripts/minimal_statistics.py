#!/usr/bin/env python
"""E18: which of the 16 coarse colour/tone statistics carry cross-generator-type transfer? (CPU only)

  python scripts/minimal_statistics.py   # -> reports/minimal_statistics.json

Cross-type mean = mean AUROC over filmsynth->sdxl, filmsynth->zimage, sdxl->filmsynth, zimage->filmsynth with the
standard probe. Leave-one-out drop, then greedy forward selection (add the statistic that most raises the cross-type
mean; stop when no gain). Reports the chosen set, its CIs on each pair, same-stock p@5 and own-source rank-1.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from colour_vs_texture import COLOUR_TONE, IDX_C  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

PAIRS = (("filmsynth", "sdxl"), ("filmsynth", "zimage"), ("sdxl", "filmsynth"), ("zimage", "filmsynth"))


def _feat(args):
    from PIL import Image
    return X.explicit_of(Image.fromarray(V3._load(args)))[IDX_C]


def main():
    t0 = time.time()
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

    def featurize(paths, keys, real=False):
        with Pool(16) as pool:
            return np.stack(pool.map(_feat, [(p, k, True, real) for p, k in zip(paths, keys)], chunksize=8))
    F = {g: {"t": featurize([str(P.ROOT / a["path"]) for a in t], [a["variant_id"] for a in t]), "c": featurize([str(P.ROOT / a["path"]) for a in c], [a["variant_id"] for a in c])} for g, (t, c) in gens.items()}
    Fo = featurize([by_id[s]["path"] for s in srcs], srcs, real=True); Fb = featurize([r["path"] for r in bn], [r["image_id"] for r in bn], real=True)
    meta = {g: {"t": (np.array([a["source_image_id"] for a in t]), np.array([sp[a["source_image_id"]] for a in t])),
                "c": (np.array([a["source_image_id"] for a in c]), np.array([sp[a["source_image_id"]] for a in c]))} for g, (t, c) in gens.items()}
    intact = np.array([r["partition"] == "blue-intact" for r in bn]); stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]

    def xy(g, split, cols):
        (st, pt), (sc_, pc) = meta[g]["t"], meta[g]["c"]
        return (np.vstack([F[g]["t"][pt == split][:, cols], F[g]["c"][pc == split][:, cols]]),
                np.r_[np.ones((pt == split).sum()), np.zeros((pc == split).sum())].astype(int), np.r_[st[pt == split], sc_[pc == split]])

    def cross(cols, n_boot=0):
        out = {}
        for a, b in PAIRS:
            Xtr, ytr, _ = xy(a, "train", cols); Xte, yte, ste = xy(b, "test", cols)
            r, _ = V3.probe_auroc(Xtr, ytr, Xte, yte, ste) if n_boot else (None, None)
            if not n_boot:
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import roc_auc_score
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced").fit(sc.transform(Xtr), ytr)
                r = {"auroc": round(float(roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])), 4)}
            out[f"{a}->{b}"] = r
        return round(float(np.mean([v["auroc"] for v in out.values()])), 4), out
    all_cols = list(range(len(COLOUR_TONE)))
    full, _ = cross(all_cols)
    loo = {COLOUR_TONE[i]: round(full - cross([c for c in all_cols if c != i])[0], 4) for i in all_cols}
    chosen, best = [], 0.0
    trace = []
    while True:
        cands = [(cross(chosen + [i])[0], i) for i in all_cols if i not in chosen]
        score, i = max(cands)
        if score <= best + 0.005 or len(chosen) >= 8:
            break
        chosen.append(i); best = score; trace.append((COLOUR_TONE[i], score))
    final_mean, final = cross(chosen, n_boot=V3.N_BOOT)
    ss = V3.same_stock(Fb[intact][:, chosen], stock, roll)
    te_x = meta["sdxl"]["t"][1] == "test"
    rank1 = V3.rank1(F["sdxl"]["t"][te_x][:, chosen], meta["sdxl"]["t"][0][te_x], Fo[:, chosen], np.array(srcs))
    rep = {"full16_cross_type_mean": full, "leave_one_out_drop": dict(sorted(loo.items(), key=lambda kv: -kv[1])),
           "greedy_selection": trace, "chosen": [COLOUR_TONE[i] for i in chosen], "chosen_cross_type_mean": final_mean,
           "chosen_pairs": final, "chosen_same_stock_raw": ss, "chosen_own_source_rank1_sdxl_test": rank1, "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "minimal_statistics.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("full16_cross_type_mean", "leave_one_out_drop", "greedy_selection", "chosen_cross_type_mean", "chosen_pairs", "chosen_same_stock_raw", "chosen_own_source_rank1_sdxl_test")}, indent=1, default=str))


if __name__ == "__main__":
    main()
