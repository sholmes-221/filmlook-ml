#!/usr/bin/env python
"""E33: which cheap head on frozen DINOv2-Small L4+L8 statistics carries the most real-photograph signal?

  python scripts/frozen_head_choice.py      # -> reports/frozen_head_choice.json (cached features, CPU, analysis halves only)

E32 showed that the synthetic-family SupCon projection has an effective rank of about 4 and loses process information
relative to its own inputs, and that label-free whitening restores the students' museum-print signal. This experiment
compares label-free heads on the frozen teacher features: raw L4+L8 (1536-d), standardisation, PCA-whitening fitted on
independent unlabeled real originals (bench cache `cc_original` train, CC-BY, training allowed) at 16/32/64/128 dims,
the same fitted on synthetic renders, the SupCon projection with and without whitening, and the explicit statistics.
Metrics (uploader/artist excluded, group bootstrap CIs): commoncatalog_film stock-line p@5 and uploader share;
aic_photographs process p@5 with stereo cards removed and the decade matched, and artist share; synthetic family p@5 on
the filmsynth validation split with the source excluded (cost of the trade); effective rank on commoncatalog_film.
Pre-registered rule (research/backlog.yaml E33): summary.supported = 1 iff `L4L8_pcaw_cc_64` has (a) a museum lower CI above
chance, (b) stock-line p@5 >= the SupCon projection's, (c) uploader-share and (d) artist-share excess over the projection
<= 0.02. No test half, no BlueNeg, no GPU.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C  # noqa: E402
from e29_audit import eff_rank, retrieval  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
DIMS = (16, 32, 64, 128)


class PCAW:
    """Standardise, PCA to k dims, whiten; fitted on one set, applied to any."""
    def __init__(self, k, eps=1e-3):
        self.k, self.eps = k, eps

    def fit(self, Z):
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6; Zs = (Z - self.mu) / self.sd
        U, s, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); self.V = Vt[:self.k].T; self.s = s[:self.k] / np.sqrt(len(Z) - 1); self.c = Zs.mean(0)
        return self

    def __call__(self, Z):
        return (((Z - self.mu) / self.sd - self.c) @ self.V) / (self.s + self.eps)


def share(E, group):
    S = E @ E.T; np.fill_diagonal(S, -np.inf)
    return round(float(np.mean([(group[np.argsort(-S[i])[:5]] == group[i]).mean() for i in range(len(E))])), 4)


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    src = np.array([str(r.get("source")) for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    cc_tr = (st == "cc_original") & (sp == "train"); fs_val = (st == "filmsynth") & (sp == "val") & treated
    A = dict(np.load(AIC.with_suffix(".npz"))); rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    ma = np.array([r["split"] == "analysis" for r in rows_a]); mg = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows_g])
    artist = np.array([r["uploader"] for r in rows_a])[ma]; proc = np.array([process_family(r.get("process")) for r in rows_a])[ma]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows_a])[ma]
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows_a])[ma]
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    up = np.array([r["uploader"] for r in rows_g])[mg]; line = np.array([r.get("film_stock_line") or r.get("film_stock_claim") or "?" for r in rows_g])[mg]
    sets = {"ccf": (G["dino_style"][mg], G["explicit"][mg]), "aic": (A["dino_style"][ma], A["explicit"][ma]), "fs_val": (Fc["dino_style"][fs_val], Fc["explicit"][fs_val])}
    fit_cc, fit_fs = Fc["dino_style"][cc_tr], Fc["dino_style"][tr]
    heads = {"L4L8_raw": lambda d, e: d, "L4L8_std_cc": (lambda mu, sd: (lambda d, e: (d - mu) / sd))(fit_cc.mean(0), fit_cc.std(0) + 1e-6)}
    for k in DIMS:
        heads[f"L4L8_pcaw_cc_{k}"] = (lambda p: (lambda d, e: p(d)))(PCAW(k).fit(fit_cc))
    heads["L4L8_pcaw_fs_64"] = (lambda p: (lambda d, e: p(d)))(PCAW(64).fit(fit_fs))
    hyb_fit = np.hstack([Fc["explicit"][cc_tr], fit_cc]); heads["hybrid_pcaw_cc_64"] = (lambda p: (lambda d, e: p(np.hstack([e, d]))))(PCAW(64).fit(hyb_fit))
    heads["supcon_family_proj"] = lambda d, e: proj(np.hstack([e, d]))
    pw = PCAW(64).fit(proj(hyb_fit)); heads["supcon_proj_pcaw_cc"] = lambda d, e: pw(proj(np.hstack([e, d])))
    heads["explicit28_std_cc"] = (lambda mu, sd: (lambda d, e: (e - mu) / sd))(Fc["explicit"][cc_tr].mean(0), Fc["explicit"][cc_tr].std(0) + 1e-6)
    heads["colour16_std_cc"] = (lambda mu, sd: (lambda d, e: (e[:, IDX_C] - mu) / sd))(Fc["explicit"][cc_tr][:, IDX_C].mean(0), Fc["explicit"][cc_tr][:, IDX_C].std(0) + 1e-6)
    out = {"experiment": "E33_frozen_head_choice", "analysis_data_only": True, "n_ccf": int(mg.sum()), "n_ccf_uploaders": int(len(set(up))), "n_aic": int(ma.sum()), "n_fs_val": int(fs_val.sum()),
           "fit_set": "bench cc_original train (%d real originals)" % int(cc_tr.sum()), "heads": {}}
    print(f"{'head':22s} {'stock-line p@5':>26s} {'up share':>8s} | {'museum p@5 (dec-matched)':>28s} {'art share':>9s} | {'synth fam p@5':>22s} | erank", flush=True)
    for name, h in heads.items():
        Ec = unit(np.nan_to_num(h(*sets["ccf"]).astype(np.float32))); Ea = unit(np.nan_to_num(h(*sets["aic"]).astype(np.float32))); Ev = unit(np.nan_to_num(h(*sets["fs_val"]).astype(np.float32)))
        r = {"stock_line_p5": retrieval(Ec, line, up), "uploader_share": share(Ec, up), "museum_p5": retrieval(Ea, proc, artist, allowed), "artist_share": share(Ea, artist),
             "synth_family_p5": retrieval(Ev, fam[fs_val], src[fs_val]), "eff_rank_ccf": eff_rank(Ec), "dim": int(Ec.shape[1])}
        out["heads"][name] = r; s, mu_, v = r["stock_line_p5"], r["museum_p5"], r["synth_family_p5"]
        print(f"{name:22s} {s['precision@5']:.3f} {str(s['ci95']):>16s} ch {s['chance']:.3f} {r['uploader_share']:8.3f} | {mu_['precision@5']:.3f} {str(mu_['ci95']):>16s} ch {mu_['chance']:.3f} {r['artist_share']:9.3f} | {v['precision@5']:.3f} {str(v['ci95']):>16s} | {r['eff_rank_ccf']}", flush=True)
    b, c = out["heads"]["supcon_family_proj"], out["heads"]["L4L8_pcaw_cc_64"]
    a_ = c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"]; b_ = c["stock_line_p5"]["precision@5"] >= b["stock_line_p5"]["precision@5"]
    c_ = c["uploader_share"] - b["uploader_share"] <= 0.02; d_ = c["artist_share"] - b["artist_share"] <= 0.02
    best = max(out["heads"], key=lambda n: out["heads"][n]["museum_p5"]["precision@5"] + out["heads"][n]["stock_line_p5"]["precision@5"])
    out["summary"] = {"supported": int(a_ and b_ and c_ and d_), "museum_ci_low_above_chance": bool(a_), "stock_line_not_worse": bool(b_), "uploader_leak_ok": bool(c_), "artist_leak_ok": bool(d_),
                      "candidate_stock_line_p5": c["stock_line_p5"]["precision@5"], "baseline_stock_line_p5": b["stock_line_p5"]["precision@5"], "candidate_museum_p5": c["museum_p5"]["precision@5"],
                      "baseline_museum_p5": b["museum_p5"]["precision@5"], "candidate_synth_family_p5": c["synth_family_p5"]["precision@5"], "baseline_synth_family_p5": b["synth_family_p5"]["precision@5"],
                      "best_by_real_sum": best, "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "frozen_head_choice.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
