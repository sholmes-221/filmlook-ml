#!/usr/bin/env python
"""E55: uncertainty-aware retrieval - when should the product name a process or a stock, and when should it abstain?

  python scripts/e55_abstention.py           # -> reports/e55_abstention.json  (no image is read; stored query records only)

Inputs: the per-query records written by scripts/e53_fresh_holdout.py - for every representation and query, the ten
nearest neighbours (query's artist / uploader excluded) with their cosines - from the self-test on the spent halves
(reports/e53_self_test_queries.npz, calibration) and from the single holdout read (reports/e53_fresh_holdout_queries.npz,
evaluation). The holdout is therefore not re-read and no threshold is tuned on it.
Decision rule: a label claim ("looks like <process>" / "looks like <stock line>") is the plurality label of the top-5
neighbours; it is made only when at least three of the five agree AND the top-1 cosine is at or above a threshold tau;
otherwise the system abstains and shows "similar look" without a label. tau is calibrated on the spent halves as the
smallest threshold at which the answered-query precision reaches a target of twice the class prior (chance), separately
per set; agreement thresholds 3/5 and 4/5 are both reported.
Readings on the holdout: coverage, answered precision with a bootstrap CI over artists / uploaders, the precision of
answering every query with the plurality label (no abstention), the precision of the abstained queries (what would have
been wrong), and the same curve for the baselines.
Pre-registered rule (research/backlog.yaml E55): supported iff, for the candidate seed-mean (PCA-32 int8), the museum
process claim at the calibrated tau and 3/5 agreement has coverage >= 0.2, answered precision >= 1.5 x chance with its
lower CI above chance, and higher answered precision than answering every query. Stock-line claims are reported only
(at chance for every representation on the spent halves); the rule does not depend on them.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eval_commons_film import boot_groups  # noqa: E402
from freeze_candidate import CANDIDATE  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

OUT = P.REPORTS / "e55_abstention.json"
CAL = P.REPORTS / "e53_self_test_queries.npz"; EVAL = P.REPORTS / "e53_fresh_holdout_queries.npz"
REPS = [f"{t}:pca32_int8" for t in CANDIDATE] + ["style_fs_v2_real_f576:pca32_int8", "imagenet_init:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std"]
SETS = {"museum": ("museum_proc", "museum_artist"), "scans": ("scans_line", "scans_up")}


def records(z, name, s):
    lab, grp = SETS[s]; k = f"{name}|{s}|"
    if k + "qi" not in z:
        return None
    qi, top, S = z[k + "qi"], z[k + "top"], z[k + "S_top"]; label = z[lab]; group = z[grp]
    nb = label[top[:, :5]]; plur, agree = [], []
    for row in nb:
        u, c = np.unique(row, return_counts=True); plur.append(u[c.argmax()]); agree.append(c.max() / 5)
    plur, agree = np.array(plur), np.array(agree); truth = label[qi]
    prior = np.array([(label[group != group[i]] == label[i]).mean() for i in qi])
    return {"correct": (plur == truth).astype(float), "agree": agree, "top1": S[:, 0], "groups": group[qi], "prior": prior, "n": len(qi)}


def curve(r, taus, min_agree):
    out = []
    for t in taus:
        ans = (r["agree"] >= min_agree) & (r["top1"] >= t)
        out.append({"tau": round(float(t), 4), "coverage": round(float(ans.mean()), 4), "precision": round(float(r["correct"][ans].mean()), 4) if ans.any() else None, "n_answered": int(ans.sum())})
    return out


def calibrate(r, min_agree, target_mult=2.0):
    """Smallest tau (over the top-1 cosine quantiles) at which answered precision >= target_mult x chance; None if never."""
    chance = float(r["prior"].mean()); taus = np.quantile(r["top1"], np.linspace(0, 0.95, 39))
    for t in taus:
        ans = (r["agree"] >= min_agree) & (r["top1"] >= t)
        if ans.sum() >= 10 and r["correct"][ans].mean() >= target_mult * chance:
            return float(t), chance
    return None, chance


def evaluate(r, tau, min_agree):
    ans = (r["agree"] >= min_agree) & (r["top1"] >= tau) if tau is not None else np.zeros(r["n"], bool); chance = float(r["prior"].mean())
    res = {"tau": tau, "min_agree": min_agree, "chance": round(chance, 4), "coverage": round(float(ans.mean()), 4), "n_answered": int(ans.sum()),
           "precision_all_queries": round(float(r["correct"].mean()), 4), "precision_all_ci": boot_groups(r["correct"], r["groups"])}
    if ans.sum() >= 5:
        res["precision_answered"] = round(float(r["correct"][ans].mean()), 4); res["precision_answered_ci"] = boot_groups(r["correct"][ans], r["groups"][ans])
        res["precision_abstained"] = round(float(r["correct"][~ans].mean()), 4) if (~ans).any() else None
    else:
        res["precision_answered"] = None; res["precision_answered_ci"] = None; res["precision_abstained"] = None
    return res


def main():
    t0 = time.time(); zc, ze = np.load(CAL, allow_pickle=True), np.load(EVAL, allow_pickle=True)
    out = {"experiment": "E55_abstention", "held_out_data": True, "calibration": "spent analysis halves (e53_self_test_queries.npz)", "evaluation": "fresh holdouts (e53_fresh_holdout_queries.npz), stored records only", "representations": {}}
    for name in REPS:
        rr = {}
        for s in SETS:
            rc, re_ = records(zc, name, s), records(ze, name, s)
            if rc is None or re_ is None:
                continue
            rs = {"n_calibration": rc["n"], "n_holdout": re_["n"]}
            for min_agree in (0.6, 0.8):
                tau, chance_c = calibrate(rc, min_agree); ev = evaluate(re_, tau, min_agree); ev["calibration_chance"] = round(chance_c, 4)
                ev["calibration_at_tau"] = evaluate(rc, tau, min_agree) if tau is not None else None
                rs[f"agree_{int(min_agree * 5)}of5"] = ev
            rs["holdout_curve_3of5"] = curve(re_, np.quantile(re_["top1"], np.linspace(0, 0.9, 10)), 0.6)
            rr[s] = rs
        out["representations"][name] = rr
        m = rr.get("museum", {}).get("agree_3of5", {})
        print(f"{name:40s} museum tau {m.get('tau')} cov {m.get('coverage')} prec {m.get('precision_answered')} {m.get('precision_answered_ci')} all {m.get('precision_all_queries')} ch {m.get('chance')} | scans " +
              str({k: rr.get("scans", {}).get("agree_3of5", {}).get(k) for k in ("coverage", "precision_answered", "precision_all_queries", "chance")}), flush=True)
    # candidate seed-mean (museum, 3/5): pool the three seeds' holdout decisions
    pooled = {"correct": [], "ans": [], "groups": []}; ok = True
    for t in CANDIDATE:
        rc, re_ = records(zc, f"{t}:pca32_int8", "museum"), records(ze, f"{t}:pca32_int8", "museum")
        if rc is None or re_ is None:
            ok = False; break
        tau, _ = calibrate(rc, 0.6); ans = (re_["agree"] >= 0.6) & (re_["top1"] >= tau) if tau is not None else np.zeros(re_["n"], bool)
        pooled["correct"].append(re_["correct"]); pooled["ans"].append(ans); pooled["groups"].append(re_["groups"])
    summary = {"supported": 0, "seconds": round(time.time() - t0, 1)}
    if ok:
        c = np.concatenate(pooled["correct"]); a = np.concatenate(pooled["ans"]); g = np.concatenate(pooled["groups"]); chance = float(np.mean([records(ze, f"{t}:pca32_int8", "museum")["prior"].mean() for t in CANDIDATE]))
        prec = float(c[a].mean()) if a.any() else None; ci = boot_groups(c[a], g[a]) if a.sum() >= 5 else [None, None]
        summary.update({"museum_coverage": round(float(a.mean()), 4), "museum_precision_answered": round(prec, 4) if prec is not None else None, "museum_precision_answered_ci": ci,
                        "museum_precision_all": round(float(c.mean()), 4), "museum_chance": round(chance, 4)})
        summary["supported"] = int(a.mean() >= 0.2 and prec is not None and prec >= 1.5 * chance and ci[0] is not None and ci[0] > chance and prec > float(c.mean()))
        sc = [out["representations"][f"{t}:pca32_int8"].get("scans", {}).get("agree_3of5", {}) for t in CANDIDATE]
        summary["scans_precision_answered_seed_mean"] = round(float(np.mean([s["precision_answered"] for s in sc if s.get("precision_answered") is not None])), 4) if any(s.get("precision_answered") is not None for s in sc) else None
        summary["scans_coverage_seed_mean"] = round(float(np.mean([s.get("coverage", 0) for s in sc])), 4) if sc and sc[0] else None; summary["scans_chance"] = sc[0].get("chance") if sc and sc[0] else None
    out["summary"] = summary; OUT.write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
