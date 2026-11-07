#!/usr/bin/env python
"""E10 evaluation: are diffusion treatments at a given strength distinct beyond nuisance, and do they keep content?

  SDXL_SET=sdxl_i2i_s050 python scripts/eval_sdxl_strength.py   # -> reports/sdxl_strength_<set>.json

For the chosen SDXL set (treated + vae_identity controls, standardised): hybrid features, DINOv2 global, the family
SupCon projection trained on filmsynth v2 train sources (from the benchmark cache), and the 16 colour/tone statistics.
Metrics on val+test sources: treatment-vs-treatment distance (same source) as a ratio to the JPEG+rescale nuisance
distance; own-source rank-1 of treated outputs against raw sources (DINOv2 global; content preservation); same-treatment
precision@5 across other sources (chance stated, source CI); 4-way treatment probe accuracy on val from train.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
import build_bench_features as BB  # noqa: E402
from colour_vs_texture import IDX_C  # noqa: E402
from style_benchmark import load_cache, unit, boot  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

SET = os.environ.get("SDXL_SET", "sdxl_i2i_s050")


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    Fc, rows_c = load_cache()
    st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam_c = np.array([r["family"] for r in rows_c])
    tr_fs = (st == "filmsynth") & (sp == "train")
    proj = X.supcon(Fc["hybrid"][tr_fs], fam_c[tr_fs], dev)
    src_rows = [r for r in rows_c if r["set"] == "source"]; sp_of = {r["source"]: r["split"] for r in src_rows}
    Fo_global = Fc["dino_global"][st == "source"]; src_ids = np.array([r["source"] for r in src_rows])
    sdx = read_jsonl(P.DATA / "synthetic" / SET / "annotations.jsonl")
    sdx = [a for a in sdx if a["source_image_id"] in sp_of]
    from multiprocessing import Pool
    with Pool(16) as pool:
        arrs = pool.map(BB._load, [(str(P.ROOT / a["path"]), a["variant_id"], False, False) for a in sdx] + [(str(P.ROOT / a["path"]), a["variant_id"] + "__n", False, True) for a in sdx], chunksize=8)
    n = len(sdx); ims = [Image.fromarray(a) for a in arrs]
    ex = np.stack([X.explicit_of(im) for im in ims]).astype(np.float32); dino = X.Dino(dev); dg, ds = dino(ims)
    hyb = np.hstack([ex, ds])
    reps = {"raw_hybrid": hyb, "hybrid_supcon_family": proj(hyb), "colour_tone16": ex[:, IDX_C], "dino_global": dg}
    src = np.array([a["source_image_id"] for a in sdx]); fam = np.array([a["treatment"] for a in sdx]); treated = np.array([a.get("treated", True) for a in sdx]); spl = np.array([sp_of[s] for s in src])
    ev = np.isin(spl, ["val", "test"])
    out = {"set": SET, "n": n, "strength": sdx[0]["strength"]}
    for name, R in reps.items():
        E = unit(R[:n].astype(np.float32)); En = unit(R[n:].astype(np.float32))
        d_nuis = np.median([1 - E[i] @ En[i] for i in range(n) if ev[i]])
        by = {}
        for i in np.where(ev)[0]: by.setdefault(src[i], []).append(i)
        d_tt = []
        for s, mem in by.items():
            tr = [i for i in mem if treated[i]]
            for a in range(len(tr)):
                for b in range(a + 1, len(tr)): d_tt.append(1 - E[tr[a]] @ E[tr[b]])
        qi = np.where(ev & treated)[0]; S = E[qi] @ E[qi].T; prec, prior, gq = [], [], []
        for a, i in enumerate(qi):
            other = np.array([b for b, j in enumerate(qi) if src[j] != src[i]]); order = other[np.argsort(-S[a, other])]
            rel = (fam[qi[order]] == fam[i]).astype(float); prec.append(rel[:5].mean()); prior.append(rel.mean()); gq.append(src[i])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        trm = (spl == "train") & treated; vam = (spl == "val") & treated
        sc = StandardScaler().fit(R[:n][trm]); m = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(R[:n][trm]), fam[trm])
        out[name] = {"treatment_distinctness_ratio_to_nuisance": round(float(np.median(d_tt) / (d_nuis + 1e-9)), 2), "d_treatment": round(float(np.median(d_tt)), 4), "d_nuisance": round(float(d_nuis), 4),
                     "same_treatment_precision@5_other_sources": round(float(np.mean(prec)), 4), "ci95": boot(prec, gq), "chance": round(float(np.mean(prior)), 4),
                     "treatment_probe_val_accuracy": round(float((m.predict(sc.transform(R[:n][vam])) == fam[vam]).mean()), 4), "probe_chance": 0.25}
    Zq = unit(dg[:n][ev & treated]); Zo = unit(Fo_global); Sg = Zq @ Zo.T
    out["content_own_source_rank1_dino_global"] = round(float(np.mean([src_ids[Sg[a].argmax()] == s for a, s in enumerate(src[ev & treated])])), 4)
    out["seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / f"sdxl_strength_{SET}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
