#!/usr/bin/env python
"""E44: the single final evaluation of the session-5 candidate on the held-out halves, benchmark v1 test and BlueNeg.

  python scripts/final_holdout_eval.py --final      # -> reports/final_holdout_e44.json  (refuses to run twice)

Candidate (chosen on the analysis halves, E42/E43): PCA-64 of the 576-d pooled features of the real-distilled student
`student_style_fs_v2_real_f576` (seed 0), the PCA fitted on the student's embeddings of the training-allowed CC-BY train
originals. Comparators: the same student's 64-d distilled head, the earlier `student_style_fs_v2_real` head, and the
frozen hybrid teacher's family SupCon projection. Held-out data read here for the first and only time: commoncatalog_film
test half (stock-line p@5 uploader excluded, within-uploader different-day AUC, uploader share, scanner probe),
aic_photographs test half (process p@5 stereo removed and decade matched, artist share), filmsynth test split (same-
treatment-across-scenes p@5 with the source excluded = benchmark B; own-source rank-1 against the 15 test sources),
BlueNeg blue-intact previews (same-stock p@5 with the roll excluded, roll bootstrap = benchmark D).
Pre-registered rule (research/backlog.yaml E44): supported = 1 iff the candidate has a museum lower CI above chance, a
within-uploader AUC lower CI above 0.5, stock-line p@5 >= the same student's head - 0.02, own-source rank-1 <= 0.45,
B >= the head - 0.10 and BlueNeg same-stock p@5 >= the head - 0.03. "Supported" records the candidate as the research
embedding in the model card; nothing is deployed.
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

OUT = P.REPORTS / "final_holdout_e44.json"
CAND = "style_fs_v2_real_f576"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    if not a.final:
        raise SystemExit("this script reads the held-out halves; pass --final (once)")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the final evaluation has already been run; it is not repeated")
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ra, rg, rv, rb, L = M.labels("test")
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    bi = np.where((st == "blueneg") & np.array([r.get("partition") == "blue-intact" for r in rows_c]))[0]; rows_bn = [rows_c[i] for i in bi]; stock = fam[bi]; roll = np.array([rows_c[i]["roll"] for i in bi])
    A = dict(np.load(M.AIC.with_suffix(".npz"))); G = dict(np.load(M.CCF.with_suffix(".npz")))
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    emb = {}
    for tag, output in ((CAND, "features_576"), (CAND, "style_embedding"), ("style_fs_v2_real", "style_embedding")):
        emb[f"{tag}:{output}"] = {"aic": M.embed(tag, [r["path"] for r in ra], True, output), "ccf": M.embed(tag, [r["path"] for r in rg], True, output), "val": M.embed(tag, [r["path"] for r in rv], False, output),
                                  "bank": M.embed(tag, [r["path"] for r in rb], True, output), "blueneg": M.embed(tag, [r["path"] for r in rows_bn], True, output)}
    pca = PCA64(M.embed(CAND, fit_paths, True, "features_576")); e = emb[f"{CAND}:features_576"]
    emb["candidate_pca64"] = {k: pca(v) for k, v in e.items()}
    hyb = lambda D, E_: np.hstack([E_, D])  # noqa: E731
    emb["hybrid_supcon_family"] = {"aic": proj(hyb(A["dino_style"], A["explicit"])[L["ia"]]), "ccf": proj(hyb(G["dino_style"], G["explicit"])[L["ig"]]), "val": proj(Fc["hybrid"][[rows_c.index(r) for r in rv]]),
                                   "bank": proj(Fc["hybrid"][[rows_c.index(r) for r in rb]]), "blueneg": proj(Fc["hybrid"][bi])}
    out = {"experiment": "E44_final_holdout", "held_out_data": True, "n_aic_test": len(ra), "n_ccf_test": len(rg), "n_fs_test": len(rv), "n_blueneg_intact": int(len(bi)), "candidates": {}}
    for name, e in emb.items():
        r = M.matrix(e["aic"], e["ccf"], e["val"], L, e["bank"]); r["blueneg_same_stock_p5"] = V3.same_stock(e["blueneg"], stock, roll); out["candidates"][name] = r
        print(M.fmt(name, r) + f" | blueneg {r['blueneg_same_stock_p5']['precision@5']:.3f} {r['blueneg_same_stock_p5']['ci95']}", flush=True)
    c, h = out["candidates"]["candidate_pca64"], out["candidates"][f"{CAND}:style_embedding"]
    checks = {"museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"], "within_auc_ci_low_above_half": c["within_uploader_auc"]["ci95"][0] > 0.5,
              "stock_line_not_worse": c["stock_line_p5"]["precision@5"] >= h["stock_line_p5"]["precision@5"] - 0.02, "own_source_le_0.45": c["own_source_rank1"] <= 0.45,
              "B_cost_ok": c["synth_family_p5"]["precision@5"] >= h["synth_family_p5"]["precision@5"] - 0.10, "blueneg_not_worse": c["blueneg_same_stock_p5"]["precision@5"] >= h["blueneg_same_stock_p5"]["precision@5"] - 0.03}
    out["summary"] = {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "candidate_museum_p5": c["museum_p5"]["precision@5"], "candidate_museum_ci": c["museum_p5"]["ci95"], "museum_chance": c["museum_p5"]["chance"],
                      "candidate_within_auc": c["within_uploader_auc"]["auc"], "candidate_within_ci": c["within_uploader_auc"]["ci95"], "candidate_stock_line_p5": c["stock_line_p5"]["precision@5"], "head_stock_line_p5": h["stock_line_p5"]["precision@5"],
                      "candidate_own_source_rank1": c["own_source_rank1"], "candidate_B": c["synth_family_p5"]["precision@5"], "head_B": h["synth_family_p5"]["precision@5"], "candidate_blueneg": c["blueneg_same_stock_p5"]["precision@5"],
                      "head_blueneg": h["blueneg_same_stock_p5"]["precision@5"], "teacher_blueneg": out["candidates"]["hybrid_supcon_family"]["blueneg_same_stock_p5"]["precision@5"], "candidate_eff_rank": c["eff_rank"], "candidate_artist_share": c["artist_share"], "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
