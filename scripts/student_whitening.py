#!/usr/bin/env python
"""E36: does an independently fitted whitening of the students' 64-d outputs restore their real-photograph signal?

  python scripts/student_whitening.py       # -> reports/student_whitening.json (cached features, CPU)

E32: every student's museum-print process signal returns after PCA whitening fitted on the evaluated half itself, and the
students' outputs have effective rank 2-3 (they inherit the collapsed SupCon target). A deployable fix must not be fitted on
evaluation data. Here the whitening (64x64, label-free) is fitted on the students' embeddings of the training-allowed
commoncatalog_ccby train originals (601 modern Flickr photographs, uploader-disjoint from commoncatalog_film), and applied
unchanged to the museum prints and to the Flickr film scans. The evaluation-half fit is kept as the diagnostic ceiling.
Metrics (analysis halves, group bootstrap CIs): museum process p@5 (stereo removed, decade matched) and artist share;
commoncatalog_film stock-line p@5 (uploader excluded), uploader share, scanner probe; content-controlled photographer p@5
(same uploader, disjoint subject tags, aspect ratio within 10 %); effective rank.
Pre-registered rule (research/backlog.yaml E36): primary student `student_style_fs_v2_real` with the independent fit:
supported = 1 iff its museum p@5 lower CI is above chance, its stock-line p@5 is not below the raw student's, and its
uploader-share, artist-share (<= +0.02) and scanner-probe (<= +0.05) excesses over the raw student stay within the guards.
No test half, no BlueNeg, no GPU.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from e29_audit import eff_rank, retrieval  # noqa: E402
from eval_commons_film import grouped_probe  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from frozen_head_choice import PCAW, share  # noqa: E402
from style_benchmark import unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
STUDENTS = ["student_style_fs_v2", "student_style_fs", "student_style_both", "student_style_fs_v2_sel", "student_style_fs_v2_kdsdxl", "student_style_fs_v2_real", "student_style_l3_real"]
CACHE_KEY = {"student_style_fs_v2": "student64"}
PRIMARY = "student_style_fs_v2_real"


def main():
    t0 = time.time()
    Fit = dict(np.load(FIT.with_suffix(".npz"))); meta = json.loads(FIT.with_suffix(".json").read_text())
    if not meta.get("training_use_allowed"):
        raise SystemExit("fit set must be training-allowed")
    A = dict(np.load(AIC.with_suffix(".npz"))); rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    ma = np.array([r["split"] == "analysis" for r in rows_a]); mg = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows_g])
    artist = np.array([r["uploader"] for r in rows_a])[ma]; proc = np.array([process_family(r.get("process")) for r in rows_a])[ma]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows_a])[ma]
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows_a])[ma]
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    rg = [r for r, k in zip(rows_g, mg) if k]; up = np.array([r["uploader"] for r in rg]); line = np.array([r.get("film_stock_line") or r.get("film_stock_claim") or "?" for r in rg])
    scanner = np.array([r.get("scanner_category") or "" for r in rg]); has = scanner != ""
    subj = [set(r.get("subjects") or []) for r in rg]; lar = np.array([np.log(r["image_width"] / r["image_height"]) for r in rg])
    photog_ok = np.array([[not (a & b) for b in subj] for a in subj]) & (np.abs(lar[:, None] - lar[None]) < 0.1)
    out = {"experiment": "E36_student_whitening", "analysis_data_only": True, "fit_set": {"n": meta["n"], "dataset": meta["dataset"], "split": meta["split"]}, "n_aic": int(ma.sum()), "n_ccf": int(mg.sum()), "students": {}}
    print(f"{'student':26s} {'variant':10s} {'museum p@5':>24s} {'art':>6s} | {'stock-line p@5':>24s} {'up':>6s} {'scan':>5s} {'photog':>12s} | erank", flush=True)
    for s in STUDENTS:
        k = CACHE_KEY.get(s, s); Za, Zg = A[k][ma], G[k][mg]; w_ind = PCAW(64).fit(Fit[s])
        variants = {"raw": (lambda Z: Z, lambda Z: Z), "whiten_indep": (w_ind, w_ind), "whiten_evalfit": (PCAW(64).fit(Za), PCAW(64).fit(Zg))}   # (museum map, film-scan map)
        out["students"][s] = {}
        for vn, (fa, fg) in variants.items():
            Ea = unit(np.nan_to_num(fa(Za).astype(np.float32))); Eg = unit(np.nan_to_num(fg(Zg).astype(np.float32)))
            r = {"museum_p5": retrieval(Ea, proc, artist, allowed), "artist_share": share(Ea, artist), "stock_line_p5": retrieval(Eg, line, up), "uploader_share": share(Eg, up),
                 "scanner_probe": grouped_probe(Eg[has], scanner[has], up[has]), "photographer_p5_content_aspect_controlled": retrieval(Eg, up, np.arange(len(Eg)), photog_ok), "eff_rank_ccf": eff_rank(Eg), "eff_rank_aic": eff_rank(Ea)}
            out["students"][s][vn] = r; m_, sl, pc = r["museum_p5"], r["stock_line_p5"], r["photographer_p5_content_aspect_controlled"]
            print(f"{s:26s} {vn:10s} {m_['precision@5']:.3f} {str(m_['ci95']):>15s} ch {m_['chance']:.3f} {r['artist_share']:6.3f} | {sl['precision@5']:.3f} {str(sl['ci95']):>15s} ch {sl['chance']:.3f} {r['uploader_share']:6.3f} {r['scanner_probe']['accuracy'] or 0:5.3f} {pc['precision@5'] or 0:.3f}/{pc.get('chance', 0) or 0:.3f} | {r['eff_rank_ccf']}/{r['eff_rank_aic']}", flush=True)
    b, c = out["students"][PRIMARY]["raw"], out["students"][PRIMARY]["whiten_indep"]
    checks = {"museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"], "stock_line_not_worse": c["stock_line_p5"]["precision@5"] >= b["stock_line_p5"]["precision@5"],
              "uploader_leak_ok": c["uploader_share"] - b["uploader_share"] <= 0.02, "artist_leak_ok": c["artist_share"] - b["artist_share"] <= 0.02,
              "scanner_leak_ok": (c["scanner_probe"]["accuracy"] or 0) - (b["scanner_probe"]["accuracy"] or 0) <= 0.05}
    n_rec = sum(1 for s in STUDENTS if out["students"][s]["whiten_indep"]["museum_p5"]["ci95"][0] > out["students"][s]["whiten_indep"]["museum_p5"]["chance"])
    out["summary"] = {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY, "primary_raw_museum_p5": b["museum_p5"]["precision@5"], "primary_whitened_museum_p5": c["museum_p5"]["precision@5"],
                      "primary_whitened_museum_ci": c["museum_p5"]["ci95"], "museum_chance": c["museum_p5"]["chance"], "primary_raw_stock_line_p5": b["stock_line_p5"]["precision@5"], "primary_whitened_stock_line_p5": c["stock_line_p5"]["precision@5"],
                      "primary_raw_uploader_share": b["uploader_share"], "primary_whitened_uploader_share": c["uploader_share"], "primary_raw_artist_share": b["artist_share"], "primary_whitened_artist_share": c["artist_share"],
                      "students_recovered_indep": n_rec, "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "student_whitening.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
