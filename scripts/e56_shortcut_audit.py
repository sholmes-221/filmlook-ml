#!/usr/bin/env python
"""E56: does the candidate's film-stock signal run through photographer identity, scene content or the scanner?

  python scripts/e56_shortcut_audit.py        # -> reports/e56_shortcut_audit.json  (spent analysis half; CPU only)

Data: the commoncatalog_film ANALYSIS half (spent for selection, the right place for diagnostics; the fresh holdout is not
touched). Representations: the frozen candidate seeds (PCA-32 int8), one E23b seed, the untrained ImageNet backbone with
the same PCA (identical settings), DINOv2-Small L4+L8 PCA-64 and CLS++mean PCA-64 (cached teacher features), explicit-28.
Readings per representation:
  variance explained (ridge, standardised embedding as target): from stock-line one-hots, scanner-family one-hots and
  the 28 leading PCs of DINOv2-global content, each under uploader-grouped 5-fold CV; from uploader one-hots under a random
  5-fold split (identity cannot be predicted for unseen uploaders, so this is within-sample identity explained);
  stock-line p@5 with the query's uploader excluded: unrestricted; candidates restricted to the SAME scanner family
  (the scanner cue cannot help); restricted to a DIFFERENT scanner family (the scanner cue works against the stock);
  colour scans only (removes the monochrome-vs-colour cue); plus uploader share and photographer p@5 (subjects disjoint,
  aspect matched) as before.
Pre-registered rule (research/backlog.yaml E56): supported iff for every candidate seed (1) the different-scanner stock-line
p@5 is at least the unrestricted value minus 0.02 (the stock signal does not need the scanner) and (2) the uploader-identity
R^2 (random split) is below the ImageNet-init control's. A representation whose stock retrieval collapses when the scanner
family differs retrieves scanners, not stocks.
"""
import json
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
from e29_audit import grouped_r2  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import FIT  # noqa: E402
from e50_aesthetic_vs_content import onehots  # noqa: E402
from e53_fresh_holdout import hits, load_arrays, scan_labels, share, summarise  # noqa: E402
from freeze_candidate import CANDIDATE, Frozen  # noqa: E402
from style_benchmark import unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

OUT = P.REPORTS / "e56_shortcut_audit.json"


def random_r2(Zx, Y, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    Ys = StandardScaler().fit_transform(Y); pred = np.zeros_like(Ys)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Zx):
        sc = StandardScaler().fit(Zx[tr]); pred[te] = Ridge(alpha=10.0).fit(sc.transform(Zx[tr]), Ys[tr]).predict(sc.transform(Zx[te]))
    return round(float(1 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum()), 4)


def main():
    t0 = time.time()
    rows_all = json.loads(M.CCF.with_suffix(".json").read_text())["rows"]; G = dict(np.load(M.CCF.with_suffix(".npz")))
    ig = np.array([i for i, r in enumerate(rows_all) if r["split"] == "analysis" and r.get("film_stock")]); rows = [rows_all[i] for i in ig]
    L = scan_labels(rows); arrays = load_arrays(rows); Fit = dict(np.load(FIT.with_suffix(".npz")))
    reps = {}
    for t in CANDIDATE + ["style_fs_v2_real_f576", "imagenet_init"]:
        fz = Frozen(t); F = fz.features(arrays); reps[f"{t}:pca32_int8" if t != "imagenet_init" else "imagenet_init:pca64"] = fz.reduce(F, 32, True) if t != "imagenet_init" else fz.reduce(F, 64, False)
    reps["dino_style:pca64"] = PCA64(Fit["dino_style"], 64)(G["dino_style"][ig]); reps["dino_global:pca64"] = PCA64(Fit["dino_global"], 64)(G["dino_global"][ig])
    mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6; reps["explicit28_std"] = (G["explicit"][ig] - mu) / sd
    content = PCA64(G["dino_global"][ig], 28)(G["dino_global"][ig])
    has = L["has"]; same = (L["scanner"][:, None] == L["scanner"][None]) & has[:, None] & has[None]; diff = (L["scanner"][:, None] != L["scanner"][None]) & has[:, None] & has[None]
    colour = L["colour"][:, None] & L["colour"][None]
    X_stock, X_scan, X_up = onehots([L["line"]]), onehots([L["scanner"]]), onehots([L["up"]])
    out = {"experiment": "E56_shortcut_audit", "analysis_data_only": True, "n": len(rows), "n_uploaders": int(len(set(L["up"]))), "n_with_scanner": int(has.sum()), "scanner_families": dict(Counter(L["scanner"][has])),
           "n_stock_lines": int(len(set(L["line"]))), "representations": {}}
    for name, Z in reps.items():
        E = unit(np.nan_to_num(Z.astype(np.float32))); r = {}
        r["r2_from_stock_line_grouped"] = grouped_r2(X_stock, E, L["up"]); r["r2_from_scanner_grouped"] = grouped_r2(X_scan, E, L["up"]); r["r2_from_content28_grouped"] = grouped_r2(content, E, L["up"])
        r["r2_from_uploader_random_split"] = random_r2(X_up, E)
        for key, allowed in (("unrestricted", None), ("same_scanner", same), ("different_scanner", diff), ("colour_only", colour)):
            h, g, pr, _, _ = hits(E, L["line"], L["up"], allowed); r[f"stock_line_p5_{key}"] = summarise(h, g, pr)
        r["uploader_share"] = share(E, L["up"]); hp, gp, prp, _, _ = hits(E, L["up"], np.arange(len(E)), L["photog_ok"]); r["photographer_p5"] = summarise(hp, gp, prp)
        out["representations"][name] = r
        f = lambda k: r[k]["precision@5"]  # noqa: E731
        print(f"{name:38s} R2 stock {r['r2_from_stock_line_grouped']:.3f} scanner {r['r2_from_scanner_grouped']:.3f} content {r['r2_from_content28_grouped']:.3f} uploader(rand) {r['r2_from_uploader_random_split']:.3f} | "
              f"stock p@5 all {f('stock_line_p5_unrestricted'):.3f} same-scan {f('stock_line_p5_same_scanner')} diff-scan {f('stock_line_p5_different_scanner')} colour {f('stock_line_p5_colour_only')} | up share {r['uploader_share']:.3f} photog {r['photographer_p5']['precision@5']}", flush=True)
    R = out["representations"]; cands = [R[f"{t}:pca32_int8"] for t in CANDIDATE]
    ok1 = all((c["stock_line_p5_different_scanner"]["precision@5"] or 0) >= c["stock_line_p5_unrestricted"]["precision@5"] - 0.02 for c in cands)
    ok2 = all(c["r2_from_uploader_random_split"] < R["imagenet_init:pca64"]["r2_from_uploader_random_split"] for c in cands)
    out["summary"] = {"supported": int(ok1 and ok2), "different_scanner_not_worse": ok1, "uploader_identity_below_init": ok2,
                      "candidate_r2_uploader_mean": round(float(np.mean([c["r2_from_uploader_random_split"] for c in cands])), 4), "init_r2_uploader": R["imagenet_init:pca64"]["r2_from_uploader_random_split"],
                      "candidate_r2_stock_mean": round(float(np.mean([c["r2_from_stock_line_grouped"] for c in cands])), 4), "candidate_r2_scanner_mean": round(float(np.mean([c["r2_from_scanner_grouped"] for c in cands])), 4),
                      "candidate_r2_content_mean": round(float(np.mean([c["r2_from_content28_grouped"] for c in cands])), 4),
                      "candidate_stock_p5_unrestricted_mean": round(float(np.mean([c["stock_line_p5_unrestricted"]["precision@5"] for c in cands])), 4),
                      "candidate_stock_p5_different_scanner_mean": round(float(np.mean([c["stock_line_p5_different_scanner"]["precision@5"] or 0 for c in cands])), 4),
                      "candidate_stock_p5_same_scanner_mean": round(float(np.mean([c["stock_line_p5_same_scanner"]["precision@5"] or 0 for c in cands])), 4), "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); OUT.write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
