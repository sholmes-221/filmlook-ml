#!/usr/bin/env python
"""E37: within a photographer, does any embedding tell their film stocks apart (content, camera, scanner held roughly fixed)?

  python scripts/within_photographer_stock.py     # -> reports/within_photographer_stock.json (commoncatalog_film analysis half, CPU)

Across photographers, stock retrieval is at chance (E27b) and every informative head also clusters by photographer
(E33-E36). The complementary question is whether stock information exists at all in these embeddings once the photographer
is held fixed. For every uploader with at least two stock lines and four scans, each scan is a query against that
uploader's other scans; the statistic is the query-wise pairwise AUC (same-line candidates ranked above other-line
candidates), averaged, chance 0.5. Roll control: the `different_day` variant keeps only candidates taken on another day
than the query (same-roll frames share scene and lighting). Significance: uploader-bootstrap CI and a permutation null
(stock lines shuffled within each uploader, 1000 draws).
Secondary reading: `colour_only_different_day` keeps only colour scans (removes the monochrome-vs-colour cue).
Pre-registered rule (research/backlog.yaml E37): summary.supported = 1 iff the hybrid teacher's different-day AUC lower CI
is above 0.5 AND its permutation p-value is below 0.05. Other embeddings are reported for observation. Analysis half only.
"""
import json
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from eval_commons_film import boot_groups  # noqa: E402
from frozen_head_choice import PCAW  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
N_PERM = 1000


def within_auc(E, line, up, day, different_day, colour=None):
    """Per query: AUC of same-line vs other-line candidates within the uploader (optionally other days only; optionally
    colour scans only, which removes the monochrome-vs-colour cue)."""
    S = E @ E.T; aucs, qg = [], []
    for i in range(len(E)):
        ok = (up == up[i]) & (np.arange(len(E)) != i)
        if different_day:
            ok &= day != day[i]
        if colour is not None:
            if not colour[i]:
                continue
            ok &= colour
        pos = ok & (line == line[i]); neg = ok & (line != line[i])
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        sp, sn = S[i][pos], S[i][neg]; aucs.append(float((sp[:, None] > sn[None]).mean() + 0.5 * (sp[:, None] == sn[None]).mean())); qg.append(up[i])
    return np.array(aucs), np.array(qg)


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng = np.random.default_rng(0)
    G = dict(np.load(CCF.with_suffix(".npz"))); rows = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    Fit = dict(np.load(FIT.with_suffix(".npz")))
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    m = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows]); rg = [r for r, k in zip(rows, m) if k]
    up = np.array([r["uploader"] for r in rg]); line = np.array([r.get("film_stock_line") or r["film_stock_claim"] for r in rg]); day = np.array([str(r.get("datetaken") or "?")[:10] for r in rg])
    per = {u: Counter(line[up == u]) for u in set(up)}; keep_u = {u for u, c in per.items() if len(c) >= 2 and sum(c.values()) >= 4}; k = np.isin(up, list(keep_u))
    colour = np.array([r.get("film_type") in ("colour_negative", "colour_reversal") for r in rg])
    up, line, day, colour = up[k], line[k], day[k], colour[k]
    hyb = np.hstack([G["explicit"], G["dino_style"]])[m][k]
    cands = {"hybrid_supcon_family": proj(hyb), "L4L8_pcaw64_ccby": PCAW(64).fit(Fit["dino_style"])(G["dino_style"][m][k]), "raw_dino_global": G["dino_global"][m][k],
             "explicit28": G["explicit"][m][k], "colour_tone16": G["explicit"][m][k][:, IDX_C], "texture12": G["explicit"][m][k][:, IDX_T]}
    for s, key in (("student_style_fs_v2", "student64"), ("student_style_fs_v2_real", "student_style_fs_v2_real"), ("student_style_fs", "student_style_fs")):
        cands[s + "_raw"] = G[key][m][k]; cands[s + "_whiten_indep"] = PCAW(64).fit(Fit[s])(G[key][m][k])
    out = {"experiment": "E37_within_photographer_stock", "analysis_data_only": True, "n": int(k.sum()), "n_uploaders": int(len(keep_u)), "lines_per_uploader": {u: dict(per[u]) for u in sorted(keep_u)}, "candidates": {}}
    print(f"n {k.sum()} scans, {len(keep_u)} uploaders with >= 2 stock lines", flush=True)
    for name, Z in cands.items():
        E = unit(np.nan_to_num(Z.astype(np.float32))); r = {}
        for variant, dd, col in (("all_days", False, None), ("different_day", True, None), ("colour_only_different_day", True, colour)):
            a, g = within_auc(E, line, up, day, dd, col)
            if len(a) == 0:
                r[variant] = {"auc": None, "n": 0}; continue
            null = []
            for _ in range(N_PERM):
                lp = line.copy()
                for u in keep_u:
                    idx = np.where(up == u)[0]; lp[idx] = lp[rng.permutation(idx)]
                a0, _ = within_auc(E, lp, up, day, dd, col); null.append(a0.mean() if len(a0) else 0.5)
            null = np.array(null); r[variant] = {"auc": round(float(a.mean()), 4), "ci95": boot_groups(a, g), "n_queries": int(len(a)), "n_uploaders": int(len(set(g))),
                                                "perm_null_mean": round(float(null.mean()), 4), "perm_null_95": [round(float(np.percentile(null, 2.5)), 3), round(float(np.percentile(null, 97.5)), 3)], "perm_p": round(float((null >= a.mean()).mean()), 4)}
        out["candidates"][name] = r; d = r["different_day"]; c = r["colour_only_different_day"]
        print(f"{name:34s} all {r['all_days']['auc']:.3f} p {r['all_days']['perm_p']:.3f} | diff-day {d['auc'] if d['auc'] is not None else 0:.3f} {d.get('ci95')} null {d.get('perm_null_95')} p {d.get('perm_p')} n {d.get('n_queries')} | colour-only {c['auc'] if c['auc'] is not None else 0:.3f} {c.get('ci95')} p {c.get('perm_p')} n {c.get('n_queries')}", flush=True)
    t = out["candidates"]["hybrid_supcon_family"]["different_day"]
    ok = bool(t.get("ci95")) and t["ci95"][0] > 0.5 and t["perm_p"] < 0.05
    out["summary"] = {"supported": int(ok), "teacher_auc_different_day": t.get("auc"), "teacher_ci": t.get("ci95"), "teacher_perm_p": t.get("perm_p"),
                      "best_different_day": max(out["candidates"], key=lambda n: out["candidates"][n]["different_day"]["auc"] or 0),
                      "teacher_colour_only_auc": out["candidates"]["hybrid_supcon_family"]["colour_only_different_day"].get("auc"), "teacher_colour_only_ci": out["candidates"]["hybrid_supcon_family"]["colour_only_different_day"].get("ci95"), "teacher_colour_only_perm_p": out["candidates"]["hybrid_supcon_family"]["colour_only_different_day"].get("perm_p"),
                      "n_significant_different_day": int(sum(1 for c in out["candidates"].values() if c["different_day"].get("perm_p") is not None and c["different_day"]["perm_p"] < 0.05)), "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "within_photographer_stock.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
