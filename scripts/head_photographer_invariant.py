#!/usr/bin/env python
"""E35: can photographer identity be removed from the frozen teacher features without losing the real-photograph signal?

  python scripts/head_photographer_invariant.py      # -> reports/head_photographer_invariant.json (cached features, CPU)

E33/E34: label-free heads on DINOv2-Small L4+L8 statistics keep stock-line and museum-process information but cluster by
photographer; the collapsed SupCon projection is photographer-invariant but loses the information; collapse controls only
trade one against the other. This experiment estimates the between-photographer subspace on an independent, training-
allowed set with several images per uploader (commoncatalog_ccby train originals: 601 images, 424 uploaders, 92 with >= 2;
no uploader overlap with commoncatalog_film) by shrinkage LDA in a PCA-256 space, removes its top-k directions
(k = 8/32/64), and applies the label-free whitened PCA-64 head of E33 on what remains. Controls: removing k random
directions and removing the k leading principal components. A second downstream head (SupCon + VICReg, E34) is reported.
Added metric: content-controlled photographer retrieval on commoncatalog_film (same-uploader p@5 among candidates whose
subject tags are disjoint from the query's), separating photographer style from scene content.
Pre-registered rule (research/backlog.yaml E35): primary `lda32_pcaw64`: supported = 1 iff stock-line p@5 lower CI and
museum p@5 lower CI are above chance, uploader share <= projection + 0.02, artist share <= projection + 0.02 and the
scanner-probe accuracy <= projection + 0.05. Analysis halves only; no test half, no BlueNeg, no GPU needed.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from e29_audit import eff_rank, retrieval  # noqa: E402
from eval_commons_film import grouped_probe  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from frozen_head_choice import PCAW, share  # noqa: E402
from head_anticollapse import train_head  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
KS = (8, 32, 64)
PRIMARY = "lda32_pcaw64"


class Remover:
    """Standardise + PCA-256 (fit set), then project out a k-dim subspace: LDA between-uploader directions (shrinkage),
    random directions, or leading principal components."""
    def __init__(self, k, mode="lda", n_pc=256, shrink=0.1, seed=0):
        self.k, self.mode, self.n_pc, self.shrink, self.seed = k, mode, n_pc, shrink, seed

    def fit(self, Z, groups):
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6; Zs = (Z - self.mu) / self.sd; self.c = Zs.mean(0)
        U, s, Vt = np.linalg.svd(Zs - self.c, full_matrices=False); self.V = Vt[:self.n_pc].T; Y = (Zs - self.c) @ self.V
        if self.k == 0:
            self.Q = np.zeros((self.n_pc, 0)); return self
        if self.mode == "pca":
            self.Q = np.eye(self.n_pc)[:, :self.k]
        elif self.mode == "random":
            self.Q = np.linalg.qr(np.random.default_rng(self.seed).standard_normal((self.n_pc, self.k)))[0]
        else:
            from scipy.linalg import eigh
            ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True); multi = cnt[inv] >= 2; Ym, gm = Y[multi], inv[multi]
            mu_all = Ym.mean(0); Sb = np.zeros((self.n_pc, self.n_pc)); Sw = np.zeros((self.n_pc, self.n_pc))
            for g in np.unique(gm):
                Yg = Ym[gm == g]; d = (Yg.mean(0) - mu_all)[:, None]; Sb += len(Yg) * d @ d.T; R = Yg - Yg.mean(0); Sw += R.T @ R
            Sw = (1 - self.shrink) * Sw + self.shrink * np.trace(Sw) / self.n_pc * np.eye(self.n_pc)
            w, V = eigh(Sb, Sw); W = V[:, np.argsort(-w)[:self.k]]; self.Q = np.linalg.qr(W)[0]
            self.n_multi_uploaders, self.n_multi_images = int(len(np.unique(gm))), int(multi.sum())
        return self

    def __call__(self, Z):
        Y = ((Z - self.mu) / self.sd - self.c) @ self.V
        return Y - (Y @ self.Q) @ self.Q.T


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fit = dict(np.load(FIT.with_suffix(".npz"))); meta = json.loads(FIT.with_suffix(".json").read_text()); rows_f = meta["rows"]
    if not meta.get("training_use_allowed"):
        raise SystemExit("fit set must be training-allowed")
    gid = np.array([str(r.get("group_id")) for r in rows_f])
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    src = np.array([str(r.get("source")) for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); fs_val = (st == "filmsynth") & (sp == "val") & treated
    A = dict(np.load(AIC.with_suffix(".npz"))); rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    ma = np.array([r["split"] == "analysis" for r in rows_a]); mg = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows_g])
    artist = np.array([r["uploader"] for r in rows_a])[ma]; proc = np.array([process_family(r.get("process")) for r in rows_a])[ma]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows_a])[ma]
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows_a])[ma]
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    up = np.array([r["uploader"] for r in rows_g])[mg]; line = np.array([r.get("film_stock_line") or r.get("film_stock_claim") or "?" for r in rows_g])[mg]
    scanner = np.array([r.get("scanner_category") or "" for r in rows_g])[mg]; has = scanner != ""
    subj = [set(r.get("subjects") or []) for r in np.array(rows_g, dtype=object)[mg]]; content_disjoint = np.array([[not (a & b) for b in subj] for a in subj])
    sets = {"ccf": (G["dino_style"][mg], G["explicit"][mg]), "aic": (A["dino_style"][ma], A["explicit"][ma]), "fs_val": (Fc["dino_style"][fs_val], Fc["explicit"][fs_val])}
    out = {"experiment": "E35_head_photographer_invariant", "analysis_data_only": True, "fit_set": {"n": len(rows_f), "uploaders": int(len(set(gid)))}, "n_ccf": int(mg.sum()), "n_aic": int(ma.sum()), "heads": {}}
    proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev); heads = {"supcon_family_proj": lambda d, e: proj(np.hstack([e, d]))}
    p0 = PCAW(64).fit(Fit["dino_style"]); heads["pcaw64_no_removal"] = lambda d, e: p0(d)
    for mode in ("lda", "random", "pca"):
        for k in KS:
            rem = Remover(k, mode).fit(Fit["dino_style"], gid); pw = PCAW(64).fit(rem(Fit["dino_style"]))
            heads[f"{mode}{k}_pcaw64"] = (lambda rem, pw: (lambda d, e: pw(rem(d))))(rem, pw)
            if mode == "lda":
                out["fit_set"].update({"multi_uploaders": rem.n_multi_uploaders, "multi_images": rem.n_multi_images})
    rem32 = Remover(32, "lda").fit(Fit["dino_style"], gid)
    hyb_tr = np.hstack([Fc["explicit"][tr], rem32(Fc["dino_style"][tr])]); h_vic = train_head(hyb_tr, fam[tr], dev, tau=0.1, lam_vic=1.0, seed=0)
    heads["lda32_supcon_vic1"] = lambda d, e: h_vic(np.hstack([e, rem32(d)]))
    print(f"{'head':22s} {'stock-line p@5':>24s} {'up':>6s} {'scan':>5s} {'photog(cc)':>10s} | {'museum p@5':>24s} {'art':>6s} | synth | erank", flush=True)
    for name, h in heads.items():
        Ec, Ea, Ev = (unit(np.nan_to_num(h(*sets[k]).astype(np.float32))) for k in ("ccf", "aic", "fs_val"))
        r = {"stock_line_p5": retrieval(Ec, line, up), "uploader_share": share(Ec, up), "scanner_probe": grouped_probe(Ec[has], scanner[has], up[has]),
             "photographer_p5_content_controlled": retrieval(Ec, up, np.arange(len(Ec)), content_disjoint), "museum_p5": retrieval(Ea, proc, artist, allowed), "artist_share": share(Ea, artist),
             "synth_family_p5": retrieval(Ev, fam[fs_val], src[fs_val]), "eff_rank_ccf": eff_rank(Ec)}
        out["heads"][name] = r; s, m_, v, pc = r["stock_line_p5"], r["museum_p5"], r["synth_family_p5"], r["photographer_p5_content_controlled"]
        print(f"{name:22s} {s['precision@5']:.3f} {str(s['ci95']):>15s} ch {s['chance']:.3f} {r['uploader_share']:6.3f} {r['scanner_probe']['accuracy'] or 0:5.3f} {pc['precision@5'] or 0:.3f}/{pc.get('chance', 0) or 0:.3f} | {m_['precision@5']:.3f} {str(m_['ci95']):>15s} ch {m_['chance']:.3f} {r['artist_share']:6.3f} | {v['precision@5']:.3f} | {r['eff_rank_ccf']}", flush=True)
    b, c = out["heads"]["supcon_family_proj"], out["heads"][PRIMARY]
    checks = {"stock_line_ci_low_above_chance": c["stock_line_p5"]["ci95"][0] > c["stock_line_p5"]["chance"], "museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"],
              "uploader_leak_ok": c["uploader_share"] - b["uploader_share"] <= 0.02, "artist_leak_ok": c["artist_share"] - b["artist_share"] <= 0.02,
              "scanner_leak_ok": (c["scanner_probe"]["accuracy"] or 0) - (b["scanner_probe"]["accuracy"] or 0) <= 0.05}
    n0 = out["heads"]["pcaw64_no_removal"]
    out["summary"] = {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY,
                      "primary_stock_line_p5": c["stock_line_p5"]["precision@5"], "no_removal_stock_line_p5": n0["stock_line_p5"]["precision@5"], "baseline_stock_line_p5": b["stock_line_p5"]["precision@5"],
                      "primary_museum_p5": c["museum_p5"]["precision@5"], "no_removal_museum_p5": n0["museum_p5"]["precision@5"], "baseline_museum_p5": b["museum_p5"]["precision@5"],
                      "primary_uploader_share": c["uploader_share"], "no_removal_uploader_share": n0["uploader_share"], "baseline_uploader_share": b["uploader_share"],
                      "primary_artist_share": c["artist_share"], "no_removal_artist_share": n0["artist_share"], "baseline_artist_share": b["artist_share"],
                      "random32_uploader_share": out["heads"]["random32_pcaw64"]["uploader_share"], "pca32_uploader_share": out["heads"]["pca32_pcaw64"]["uploader_share"], "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "head_photographer_invariant.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
