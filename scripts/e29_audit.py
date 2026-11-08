#!/usr/bin/env python
"""E32: what does the hybrid teacher retrieve on museum prints (E29), and why do the real-distilled students lose it?

  python scripts/e29_audit.py        # -> reports/e29_audit.json (aic_photographs analysis half, cached features, CPU)

Three competing explanations are tested on cached features (no GPU, no test half):
  H1 format confound : the "process" signal is stereo-card layout, aspect ratio and decade rather than print rendering.
                       Controls: a 2-d format baseline (log aspect ratio, stereo flag); retrieval with stereo cards removed;
                       retrieval with the candidate bank restricted to the query's decade (chance = within-decade prior).
  H2 collapse        : the students carry the information but their 64-d outputs are anisotropic on this out-of-domain set.
                       Diagnostics: effective rank and mean cosine on AIC vs on commoncatalog_film; p@5 after label-free PCA
                       whitening fitted on the analysis half; geometry agreement with the teacher in and out of domain.
  H3 information loss: the distillation target (SupCon projection) or the cosine loss dropped what the print process needs.
                       Diagnostics: component ablation (colour-tone 16, texture 12, DINOv2 L4+L8, projection); artist-grouped
                       ridge R^2 of colour-tone and of the teacher projection from each student embedding.
Pre-registered rule (research/backlog.yaml E32): summary.teacher_survives = 1 iff the hybrid teacher's process p@5 lower CI
is above chance with stereo cards removed AND with the decade matched. Per student: whitened_recovers = 1 iff the whitened
p@5 lower CI is above chance (supports H2 for that student); otherwise H3 (information loss) is the default reading.
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
from eval_curated_photos import agreement, process_family  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
STUDENTS = ["student_style_fs", "student_style_both", "student_style_fs_v2_sel", "student_style_fs_v2_kdsdxl", "student_style_fs_v2_real", "student_style_l3_real"]
REAL_DISTIL = {"student_style_fs_v2_kdsdxl", "student_style_fs_v2_real", "student_style_l3_real"}


def retrieval(E, label, group, allowed=None):
    """p@5 with the query's group excluded; `allowed[i]` (bool matrix) further restricts the bank per query."""
    S = E @ E.T; prec, prior, qg = [], [], []
    for i in range(len(E)):
        ok = group != group[i]
        if allowed is not None:
            ok &= allowed[i]
        rel = label[ok] == label[i]
        if rel.sum() == 0 or (~rel).sum() == 0 or ok.sum() < 5:
            continue
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:5]
        prec.append((label[order] == label[i]).mean()); prior.append(rel.mean()); qg.append(group[i])
    if not prec:
        return {"precision@5": None, "n": 0}
    return {"precision@5": round(float(np.mean(prec)), 4), "ci95": boot_groups(prec, qg), "chance": round(float(np.mean(prior)), 4), "n": len(prec), "n_groups": int(len(set(qg)))}


def whiten(Z, eps=1e-3):
    Zc = Z - Z.mean(0); U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    return unit((Zc @ Vt.T) / (s / np.sqrt(len(Z) - 1) + eps))


def eff_rank(Z):
    Zc = unit(Z) - unit(Z).mean(0); lam = np.linalg.svd(Zc, compute_uv=False) ** 2
    return round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2)


def mean_cos(Z, seed=0):
    rng = np.random.default_rng(seed); u = unit(Z); i = rng.integers(0, len(u), 20000); j = rng.integers(0, len(u), 20000); k = i != j
    return round(float((u[i] * u[j]).sum(1)[k].mean()), 4)


def grouped_r2(Zx, Y, g):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    Ys = StandardScaler().fit_transform(Y); pred = np.zeros_like(Ys)
    for tr, te in GroupKFold(n_splits=5).split(Zx, Ys, g):
        sc = StandardScaler().fit(Zx[tr]); pred[te] = Ridge(alpha=10.0).fit(sc.transform(Zx[tr]), Ys[tr]).predict(sc.transform(Zx[te]))
    return round(float(1 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum()), 4)


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = dict(np.load(AIC.with_suffix(".npz"))); rows = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    m = np.array([r["split"] == "analysis" for r in rows]); mg = np.array([r["split"] == "analysis" for r in rows_g])
    artist = np.array([r["uploader"] for r in rows])[m]; proc = np.array([process_family(r.get("process")) for r in rows])[m]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows])[m]
    ar = np.array([np.log(r["image_width"] / r["image_height"]) for r in rows])[m]
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows])[m]
    hyb = np.hstack([F["explicit"], F["dino_style"]]); T = proj(hyb)[m]
    hyb_g = np.hstack([G["explicit"], G["dino_style"]]); T_g = proj(hyb_g)[mg]
    cands = {"format2_aspect_stereo": np.stack([ar, stereo.astype(float)], 1), "decade_onehot": (decade[:, None] == np.unique(decade)[None]).astype(float),
             "colour_tone16": F["explicit"][m][:, IDX_C], "texture12": F["explicit"][m][:, IDX_T], "dino_L4L8_1536": F["dino_style"][m], "raw_dino_global": F["dino_global"][m],
             "raw_hybrid": hyb[m], "hybrid_supcon_family": T}
    for s in STUDENTS:
        cands[s + "_64"] = F[s][m]
    same_dec = decade[:, None] == decade[None]; not_stereo = ~stereo
    regimes = {"full": None, "no_stereo": np.broadcast_to(not_stereo[None], same_dec.shape) & not_stereo[:, None], "no_stereo_decade_matched": same_dec & not_stereo[None] & not_stereo[:, None]}
    out = {"experiment": "E32_e29_audit", "analysis_data_only": True, "n": int(m.sum()), "n_artists": int(len(set(artist))), "n_stereo": int(stereo.sum()),
           "processes": dict(Counter(proc)), "decades": dict(Counter(decade)), "candidates": {}, "students": {}}
    print(f"{'candidate':30s} " + " ".join(f"{r:>34s}" for r in regimes), flush=True)
    for name, Z in cands.items():
        E = unit(np.nan_to_num(Z.astype(np.float32))); rec = {}
        for rn, allowed in regimes.items():
            rec[rn] = retrieval(E, proc, artist, allowed)
        own = []; S = E @ E.T; np.fill_diagonal(S, -np.inf)
        for i in range(len(E)):
            own.append((artist[np.argsort(-S[i])[:5]] == artist[i]).mean())
        rec["artist_share_top5"] = round(float(np.mean(own)), 4); out["candidates"][name] = rec
        print(f"{name:30s} " + " ".join(f"{(rec[r]['precision@5'] or 0):.3f} {str(rec[r].get('ci95')):>16s} ch {rec[r].get('chance', 0) or 0:.3f}" for r in regimes), flush=True)
    # H2/H3 diagnostics per student (teacher projection as the reference)
    tone = F["explicit"][m][:, IDX_C]; artist_prior = float(np.mean([(artist == a).mean() for a in artist]))
    diag_ref = {"eff_rank_aic": eff_rank(T), "eff_rank_ccf": eff_rank(T_g), "mean_cos_aic": mean_cos(T), "mean_cos_ccf": mean_cos(T_g), "r2_tone_from_emb": grouped_r2(T, tone, artist)}
    for rn, allowed in regimes.items():
        diag_ref["whitened_" + rn] = retrieval(whiten(T), proc, artist, allowed)
    out["teacher_diagnostics"] = diag_ref
    for s in STUDENTS:
        Zs, Zg = F[s][m], G[s][mg]; d = {"real_distil": s in REAL_DISTIL, "agreement_aic": agreement(Zs, T), "agreement_ccf": agreement(Zg, T_g),
                                        "eff_rank_aic": eff_rank(Zs), "eff_rank_ccf": eff_rank(Zg), "mean_cos_aic": mean_cos(Zs), "mean_cos_ccf": mean_cos(Zg),
                                        "r2_tone_from_emb": grouped_r2(Zs, tone, artist), "r2_teacher_from_emb": grouped_r2(Zs, T, artist)}
        W = whiten(Zs)
        for rn, allowed in regimes.items():
            d["whitened_" + rn] = retrieval(W, proc, artist, allowed)
        wf = d["whitened_full"]; d["whitened_recovers"] = int(bool(wf.get("ci95")) and wf["ci95"][0] > wf["chance"])
        base = out["candidates"][s + "_64"]["full"]; d["raw_ci_low_above_chance"] = int(bool(base.get("ci95")) and base["ci95"][0] > base["chance"])
        out["students"][s] = d
        print(f"{s:28s} agree aic {d['agreement_aic']:.2f} ccf {d['agreement_ccf']:.2f} | erank aic {d['eff_rank_aic']:5.1f} ccf {d['eff_rank_ccf']:5.1f} | cos aic {d['mean_cos_aic']:.2f} ccf {d['mean_cos_ccf']:.2f} | R2 tone {d['r2_tone_from_emb']:.2f} teacher {d['r2_teacher_from_emb']:.2f} | whitened p@5 {wf['precision@5']:.3f} {wf['ci95']} ch {wf['chance']:.3f}", flush=True)
    t = out["candidates"]["hybrid_supcon_family"]; fmt = out["candidates"]["format2_aspect_stereo"]["full"]
    c_ns = t["no_stereo"]; c_dm = t["no_stereo_decade_matched"]
    surv = bool(c_ns.get("ci95")) and c_ns["ci95"][0] > c_ns["chance"] and bool(c_dm.get("ci95")) and c_dm["ci95"][0] > c_dm["chance"]
    out["summary"] = {"teacher_survives": int(surv), "teacher_full": t["full"]["precision@5"], "teacher_no_stereo": c_ns["precision@5"], "teacher_no_stereo_ci": c_ns.get("ci95"), "teacher_no_stereo_chance": c_ns.get("chance"),
                      "teacher_decade_matched": c_dm["precision@5"], "teacher_decade_matched_ci": c_dm.get("ci95"), "teacher_decade_matched_chance": c_dm.get("chance"),
                      "format_baseline_full": fmt["precision@5"], "format_baseline_ci": fmt.get("ci95"),
                      "students_whitened_recover": int(sum(d["whitened_recovers"] for d in out["students"].values())),
                      "real_distil_whitened_recover": int(sum(d["whitened_recovers"] for s, d in out["students"].items() if s in REAL_DISTIL)),
                      "artist_prior": round(artist_prior, 4), "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "e29_audit.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
