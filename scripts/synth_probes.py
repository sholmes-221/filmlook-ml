#!/usr/bin/env python
"""Follow-up probes for the synthetic study: pairwise relative-attribute ranking and composition sensitivity.

  python scripts/synth_probes.py    # -> reports/synth_probes.json

1. Pairwise ranking (objective C). For two versions of the SAME source, is A warmer / more contrasty / grainier /
   more saturated / with softer highlights than B? A linear ranker is trained on feature differences of synthetic
   pairs (train sources), with labels from the applied parameters. Evaluation:
   (a) synthetic pairs from test sources;
   (b) REAL transfer: pairs (original, edit) from the dataset's independently implemented edit pipeline, on photos
       never used as sources, judged by the sign of the measured-statistic difference (warmth, y_std, noise_std,
       sat_mean). That judgement is independent of the learner, but it is a deterministic proxy, not human perception.
2. Composition probe. Controlled crops of real test photos (zoom 1.0-2.0, horizontal offset -1..1, where the crop
   window slides toward one side). Ridge regression from each representation predicts zoom and offset on held-out
   sources. This asks which representation encodes framing; colour-only statistics should not.
"""
import json
import sys
import time
from itertools import combinations

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402

from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

PAIR_ATTRS = {"warmer": ("temperature_shift", "warmth"), "more_contrast": ("contrast_adjustment", "y_std"),
              "grainier": ("grain_strength", "noise_std"), "more_saturated": ("saturation_adjustment", "sat_mean"),
              "softer_highlights": ("highlight_compression", None)}
MIN_GAP = 0.15


def pair_set(idx, src, params, mono):
    by = {}
    for i in idx:
        by.setdefault(src[i], []).append(i)
    out = {a: [] for a in PAIR_ATTRS}
    for items in by.values():
        for i, j in combinations(items, 2):
            for a, (p, _) in PAIR_ATTRS.items():
                if p in ("temperature_shift", "saturation_adjustment") and (mono[i] or mono[j]):
                    continue
                d = params[i][p] - params[j][p]
                if abs(d) >= MIN_GAP:
                    out[a].append((i, j, 1 if d > 0 else 0))
    return out


def fit_ranker(F, pairs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D = np.array([F[i] - F[j] for i, j, _ in pairs] + [F[j] - F[i] for i, j, _ in pairs])
    y = np.array([y for *_, y in pairs] + [1 - y for *_, y in pairs])
    sc = StandardScaler(with_mean=False).fit(D)
    return sc, LogisticRegression(max_iter=4000, C=0.1, fit_intercept=False).fit(sc.transform(D), y)


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anns = read_jsonl(X.SYN / "annotations.jsonl")
    paths = [str(P.ROOT / a["path"]) for a in anns]
    src = np.array([a["source_image_id"] for a in anns])
    split = np.array([a["split"] for a in anns])
    params = [a["parameters"] for a in anns]
    mono = [bool(a["parameters"].get("monochrome")) for a in anns]
    from multiprocessing import Pool
    with Pool(16) as pool:
        ex = np.stack(pool.map(X._explicit_path, paths, chunksize=8))
    dino = X.Dino(dev)
    dg, ds = dino([Image.open(p).convert("RGB") for p in paths])
    R = X.reps_from(ex, dg, ds)
    trp = pair_set(np.where(split == "train")[0], src, params, mono)
    tep = pair_set(np.where(split == "test")[0], src, params, mono)

    # real transfer pairs: original vs independent edit, on photos never used as sources
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    used = set(src)
    real = [r for r in eligible if r["image_id"] not in used and r["split"] in ("val", "test")]
    fams = {r["family_id"] for r in real}
    rows = [r for r in man if r["family_id"] in fams]
    feats = json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text())
    rim = [X.to_short(load_image(r["path"], max_side=1280)) for r in rows]
    with Pool(16) as pool:
        rex = np.stack(pool.map(X._explicit_real, [r["path"] for r in rows], chunksize=8))
    rdg, rds = dino(rim)
    RR = X.reps_from(rex, rdg, rds)
    orig = {r["family_id"]: i for i, r in enumerate(rows) if r["edit_type"] == "original"}
    real_pairs = {a: [] for a in PAIR_ATTRS}
    for i, r in enumerate(rows):
        if r["edit_type"] == "original":
            continue
        o = orig[r["family_id"]]
        fo, fe = feats[rows[o]["sha256"]], feats[r["sha256"]]
        for a, (_, stat) in PAIR_ATTRS.items():
            if stat is None or (stat in ("warmth", "sat_mean") and (fo["is_mono"] or fe["is_mono"])):
                continue
            d = fe[stat] - fo[stat]
            scale = np.std([feats[x["sha256"]][stat] for x in rows[:400]]) + 1e-9
            if abs(d) / scale >= 0.25:                        # only clear measured differences
                real_pairs[a].append((i, o, 1 if d > 0 else 0, r["edit_type"]))

    rank = {}
    for name in ("explicit", "dino_global", "dino_style", "hybrid_style"):
        rank[name] = {}
        for a in PAIR_ATTRS:
            if len(trp[a]) < 20:
                continue
            sc, m = fit_ranker(R[name], trp[a])
            acc = lambda F, ps: float(np.mean([(m.decision_function(sc.transform((F[i] - F[j])[None]))[0] > 0) == bool(y)  # noqa: E731
                                               for i, j, y, *_ in ps])) if ps else None
            rank[name][a] = {"synthetic_test_pairs": len(tep[a]), "synthetic_test_acc": round(acc(R[name], tep[a]), 4),
                             "real_edit_pairs": len(real_pairs[a]),
                             "real_edit_acc": round(acc(RR[name], real_pairs[a]), 4) if real_pairs[a] else None}

    # composition probe on real test photos
    comp_src = [r for r in eligible if r["split"] in ("train", "test") and r["image_id"] not in used][:240]
    rng = np.random.default_rng(0)
    cim, zoom, off, grp, csplit = [], [], [], [], []
    for r in comp_src:
        im = X.to_short(load_image(r["path"], max_side=1280), 512)
        for _ in range(4):
            z, o = float(rng.uniform(1.0, 2.0)), float(rng.uniform(-1, 1))
            w, h = im.width / z, im.height / z
            cx = im.width / 2 + o * (im.width - w) / 2
            box = (cx - w / 2, (im.height - h) / 2, cx + w / 2, (im.height + h) / 2)
            cim.append(X.to_short(im.crop(tuple(int(round(v)) for v in box))))
            zoom.append(z)
            off.append(o)
            grp.append(r["image_id"])
            csplit.append(r["split"])
    csplit = np.array(csplit)
    cex = np.stack([X.explicit_of(i) for i in cim])
    cdg, cds = dino(cim)
    CR = X.reps_from(cex, cdg, cds)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    comp = {}
    for name, F in CR.items():
        tr, te = csplit == "train", csplit == "test"
        sc = StandardScaler().fit(F[tr])
        res = {}
        for tname, t in (("zoom", np.array(zoom)), ("abs_offset", np.abs(off)), ("signed_offset", np.array(off))):
            m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(sc.transform(F[tr]), t[tr])
            p = m.predict(sc.transform(F[te]))
            res[tname] = round(float(1 - ((p - t[te]) ** 2).sum() / ((t[te] - t[te].mean()) ** 2).sum()), 4)
        comp[name] = {"test_r2": res}
    rep = {"pairwise_ranking": rank, "pair_min_gap": MIN_GAP,
           "composition_probe": {"n_sources": len(comp_src), "crops_per_source": 4,
                                 "test_sources": int(len(set(np.array(grp)[csplit == "test"]))), "results": comp},
           "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "synth_probes.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
