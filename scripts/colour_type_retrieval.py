#!/usr/bin/env python
"""E31: does any film-process signal survive across photographers once the monochrome cue is removed?

  python scripts/colour_type_retrieval.py        # -> reports/colour_type_retrieval.json (commoncatalog_film analysis half, cached features)

On `commoncatalog_film` same-stock retrieval is at chance for every embedding (E27b), while film-type retrieval looks
strong but mixes B&W with colour. This test keeps only colour film scans whose uploader claimed a colour type
(colour_negative vs colour_reversal; cine negatives and specials excluded) and asks, with the query's uploader excluded
and an uploader bootstrap, whether negative-vs-reversal is retrieved above its class prior. Pre-registered rule
(research/backlog.yaml E31): `summary.supported` = 1 iff the hybrid teacher's colour-type p@5 lower CI exceeds chance
AND its uploader-share excess over the prior is <= 0.02; candidates are reported alongside for observation. Analysis
half only; no test half is read. Features come from the evaluation pipeline's cache (no GPU).
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from eval_commons_film import retrieval  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

FEAT = P.EMBEDDINGS / "commoncatalog_film_features_v1"


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = dict(np.load(FEAT.with_suffix(".npz"))); rows = json.loads(FEAT.with_suffix(".json").read_text())["rows"]
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev); hyb = np.hstack([F["explicit"], F["dino_style"]])
    cands = {"raw_dino_global": F["dino_global"], "hybrid_supcon_family": proj(hyb), "student_fs_v2_64": F["student64"], "colour_tone16": F["explicit"][:, IDX_C], "texture12": F["explicit"][:, IDX_T]}
    for k in F:
        if k.startswith("student_style_"):
            cands[k + "_64"] = F[k]
    ftype = np.array([r["film_type"] for r in rows]); up = np.array([r["uploader"] for r in rows]); half = np.array([r["split"] for r in rows])
    m = (half == "analysis") & np.isin(ftype, ["colour_negative", "colour_reversal"])
    out = {"experiment": "E31_colour_type_retrieval", "analysis_data_only": True, "n": int(m.sum()), "n_uploaders": int(len(set(up[m]))), "classes": dict(Counter(ftype[m])), "candidates": {}}
    prior = float(np.mean([(up[m] == u).mean() for u in up[m]]))
    for name, Z in cands.items():
        E = unit(Z[m].astype(np.float32)); r = retrieval(E, ftype[m], up[m])
        S = E @ E.T; np.fill_diagonal(S, -np.inf); own = [(up[m][np.argsort(-S[i])[:5]] == up[m][i]).mean() for i in range(len(E))]
        out["candidates"][name] = {"colour_type_p5": r, "uploader_share_top5": round(float(np.mean(own)), 4), "uploader_prior": round(prior, 4)}
        print(f"{name:28s} p@5 {r['precision@5']:.3f} {r['ci95']} chance {r['chance']:.3f} n {r['n']}  uploader share {np.mean(own):.3f} (prior {prior:.3f})", flush=True)
    b = out["candidates"]["hybrid_supcon_family"]
    c1 = b["colour_type_p5"]["ci95"][0] > b["colour_type_p5"]["chance"]; c2 = b["uploader_share_top5"] - prior <= 0.02
    out["summary"] = {"supported": int(c1 and c2), "ci_low_above_chance": bool(c1), "leakage_ok": bool(c2), "baseline_p5": b["colour_type_p5"]["precision@5"], "chance": b["colour_type_p5"]["chance"],
                      "best_candidate": max(out["candidates"], key=lambda n: out["candidates"][n]["colour_type_p5"]["precision@5"]), "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "colour_type_retrieval.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
