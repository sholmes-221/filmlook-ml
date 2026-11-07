#!/usr/bin/env python
"""E30: is the family SupCon projection less generator-specific when trained on two generators at once?

  python scripts/joint_generator_supcon.py        # -> reports/joint_generator_supcon.json (analysis data only: val split)

E28 showed that the family projection is generator-specific whichever DINOv2 layer feeds it: trained on filmsynth it
retrieves filmsynth treatments (val B 0.467) but not sdxl/zimage ones (0.35/0.34), and trained on sdxl it collapses on
filmsynth (0.19). Hypothesis: training the projection on the union of the filmsynth and sdxl TRAIN families (13 family
labels, two pipelines) yields a projection that transfers to the generator it has never seen (zimage) at least as well
as either single-generator projection, without more content leakage. Candidates: hybrid (explicit28 ++ L4+L8 mean/std)
with projections trained on filmsynth, sdxl, filmsynth+sdxl; and the same for the L3 features (from the E26 cache) as a
secondary reading. Pre-registered rule (research/backlog.yaml E30): `summary.supported` = 1 iff the joint projection's
zimage-val B lower CI exceeds the filmsynth-only projection's point estimate AND its own-source rank-1 (test E) is not
higher by more than 0.05 AND its generator share is not higher by more than 0.02. Zimage is held out from every
projection; the benchmark test split and all real-film sets are untouched.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from style_benchmark import evaluate, load_cache  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

CACHE = P.EMBEDDINGS / "bench_layers_v1.npz"


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load_cache(); st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    feats = {"L4L8_ms": F["dino_style"].astype(np.float32)}
    if CACHE.exists():
        z = np.load(CACHE)
        if (z["ids"] == np.array([r["id"] for r in rows])).all():
            feats["L3_ms"] = z["ms_L3"].astype(np.float32)
    trains = {"filmsynth": (st == "filmsynth") & (sp == "train"), "sdxl": (st == "sdxl") & (sp == "train")}
    trains["filmsynth+sdxl"] = trains["filmsynth"] | trains["sdxl"]
    res = {}
    for fname, Z in feats.items():
        H = np.hstack([F["explicit"], Z])
        for tname, tr in trains.items():
            lab = np.array([f"{s}:{f}" for s, f in zip(st, fam)])                   # family labels namespaced by generator
            proj = X.supcon(H[tr], lab[tr], dev); E = proj(H); r = evaluate(E, F, rows, "val")
            B = r["B_same_treatment_across_scenes"]
            res[f"{fname}|{tname}"] = {"n_train": int(tr.sum()), "n_families": int(len(set(lab[tr]))),
                                       "B": {s: {"precision@5": B[s]["precision@5"], "ci95": B[s]["ci95"], "chance": B[s].get("chance")} for s in ("filmsynth", "sdxl", "zimage")},
                                       "A_p": r["A_treatment_gt_nuisance"]["p"], "C_generator_share": r["C_generator_fingerprint"]["same_generator_share_top5"],
                                       "E_rank1": r["E_semantic_leakage"]["own_source_rank1"], "E_subject_probe": r["E_semantic_leakage"]["subject_probe_accuracy"]}
            b = res[f"{fname}|{tname}"]
            print(f"{fname:7s} proj on {tname:15s} B fs {b['B']['filmsynth']['precision@5']:.3f} sdxl {b['B']['sdxl']['precision@5']:.3f} zimage {b['B']['zimage']['precision@5']:.3f} {b['B']['zimage']['ci95']}  "
                  f"C {b['C_generator_share']:.3f} E r1 {b['E_rank1']:.3f}", flush=True)
    base, joint = res["L4L8_ms|filmsynth"], res["L4L8_ms|filmsynth+sdxl"]
    c1 = joint["B"]["zimage"]["ci95"][0] > base["B"]["zimage"]["precision@5"]
    c2 = joint["E_rank1"] <= base["E_rank1"] + 0.05
    c3 = joint["C_generator_share"] <= base["C_generator_share"] + 0.02
    summary = {"supported": int(c1 and c2 and c3), "zimage_ci_above_baseline": bool(c1), "content_not_worse": bool(c2), "generator_share_not_worse": bool(c3),
               "zimage_joint_minus_filmsynth": round(joint["B"]["zimage"]["precision@5"] - base["B"]["zimage"]["precision@5"], 4),
               "filmsynth_joint_minus_filmsynth": round(joint["B"]["filmsynth"]["precision@5"] - base["B"]["filmsynth"]["precision@5"], 4),
               "sdxl_joint_minus_filmsynth": round(joint["B"]["sdxl"]["precision@5"] - base["B"]["sdxl"]["precision@5"], 4), "seconds": round(time.time() - t0, 1)}
    rep = {"experiment": "E30_joint_generator_supcon", "analysis_data_only": True, "results": res, "summary": summary}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "joint_generator_supcon.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
