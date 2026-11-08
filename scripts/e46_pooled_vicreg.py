#!/usr/bin/env python
"""E46: can the student's pooled layer keep its rank without any teacher, with a VICReg term on the layer itself?

  python scripts/e46_pooled_vicreg.py   # retrains up to five students (~1 GPU-minute each) -> reports/e46_pooled_vicreg.json

E45: the no-teacher student (`--kd 0`) carries the same pooled-layer film signal as the distilled one (museum 0.49 vs
0.50, own-source 0.24) but its PCA-64 missed the rank bound by 0.01 (7.99), on one seed. This run completes the
pre-registered follow-up: the E23b recipe with `--kd 0` and a VICReg variance/covariance term (weight 1) on the 576-d
pooled layer (`--vic-pooled 1`), three seeds, next to the plain no-teacher recipe on the same three seeds and the E43
teacher-distilled reference. Each pooled layer and its PCA-64 (fitted on the student's embeddings of the training-allowed
CC-BY train originals) go through the real-photograph matrix on the analysis halves.
Pre-registered rule (research/backlog.yaml E46): supported = 1 iff for every seed the no-teacher + VICReg student's
PCA-64 has effective rank >= 8, own-source rank-1 <= 0.45, museum p@5 lower CI above chance, within-uploader
different-day AUC lower CI above 0.5 and synthetic family p@5 within 0.10 of the E43 reference PCA-64 seed-mean, and
the VICReg seed-mean museum p@5 is not below the plain no-teacher seed-mean by more than 0.02. No test half, no BlueNeg.
"""
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

PY = sys.executable
BASE = [PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil", "--kd", "0"]
VIC = {"style_fs_v2_real_nvic1": 0, "style_fs_v2_real_nvic1_s1": 1, "style_fs_v2_real_nvic1_s2": 2}
NOKD = {"style_fs_v2_real_nokd": 0, "style_fs_v2_real_nokd_s1": 1, "style_fs_v2_real_nokd_s2": 2}
REF = ["style_fs_v2_real_f576", "style_fs_v2_real_f576_s1", "style_fs_v2_real_f576_s2"]
LAM = 1.0


def main(VIC=VIC, extra_args=(), name="E46_pooled_vicreg", out="e46_pooled_vicreg.json"):
    t0 = time.time()
    for tag, seed in {**VIC, **NOKD}.items():
        if (P.CHECKPOINTS / f"student_{tag}" / "model.onnx").exists():
            continue
        extra = ["--vic-pooled", str(LAM), *extra_args] if tag in VIC else []
        r = subprocess.run([*BASE, "--tag", tag, "--seed", str(seed), *extra], cwd=P.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed for {tag}: {r.stderr[-800:]}")
    import student_real_matrix as M
    from e43_penultimate_pca import PCA64
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); S = {}; train_rep = {}
    for tag in (*REF, *NOKD, *VIC):
        za, zg, zv, zb = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); S[f"{tag}:features_576"] = M.matrix(za, zg, zv, L, zb); print(M.fmt(f"{tag}:features_576", S[f"{tag}:features_576"]), flush=True)
        pca = PCA64(M.embed(tag, fit_paths, True, "features_576")); n = f"{tag}:pca64"; S[n] = M.matrix(pca(za), pca(zg), pca(zv), L, pca(zb)); print(M.fmt(n, S[n]), flush=True)
        rp = json.loads((P.REPORTS / f"student_{tag}.json").read_text()); train_rep[tag] = {"synthetic_test": rp["synthetic_test"], "own_source_rank1_576d_test": rp["content_invariance_test"]["own_source_rank1_576d"], "cost": rp["cost"], "loss_end": rp["train"]["loss_curve"][-1]}
    mean = lambda tags, f: round(float(np.mean([f(S[f"{t}:pca64"]) for t in tags])), 4)  # noqa: E731
    ref_synth = mean(REF, lambda r: r["synth_family_p5"]["precision@5"])
    checks = {}
    for tag in VIC:
        p = S[f"{tag}:pca64"]
        checks[tag] = {"eff_rank_ge_8": p["eff_rank"] >= 8, "own_source_le_0.45": p["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": p["museum_p5"]["ci95"][0] > p["museum_p5"]["chance"],
                       "within_auc_ci_low_above_half": p["within_uploader_auc"]["ci95"][0] > 0.5, "synth_cost_ok": p["synth_family_p5"]["precision@5"] >= ref_synth - 0.10}
    vic_mus, nokd_mus = mean(VIC, lambda r: r["museum_p5"]["precision@5"]), mean(NOKD, lambda r: r["museum_p5"]["precision@5"])
    museum_not_worse = vic_mus >= nokd_mus - 0.02
    ok = all(all(c.values()) for c in checks.values()) and museum_not_worse
    stats = lambda tags: {"museum_mean": mean(tags, lambda r: r["museum_p5"]["precision@5"]), "museum_ci_low_min": round(float(min(S[f"{t}:pca64"]["museum_p5"]["ci95"][0] for t in tags)), 4),  # noqa: E731
                          "stock_line_mean": mean(tags, lambda r: r["stock_line_p5"]["precision@5"]), "within_auc_mean": mean(tags, lambda r: r["within_uploader_auc"]["auc"]), "colour_auc_mean": mean(tags, lambda r: r["colour_only_auc"]["auc"]),
                          "synth_mean": mean(tags, lambda r: r["synth_family_p5"]["precision@5"]), "own_source_mean": mean(tags, lambda r: r["own_source_rank1"]), "eff_rank_mean": mean(tags, lambda r: r["eff_rank"]),
                          "eff_rank_min": round(float(min(S[f"{t}:pca64"]["eff_rank"] for t in tags)), 2), "artist_share_mean": mean(tags, lambda r: r["artist_share"]), "uploader_share_mean": mean(tags, lambda r: r["uploader_share"]),
                          "photographer_p5_mean": mean(tags, lambda r: r["photographer_p5"]["precision@5"] or 0.0), "raw576_eff_rank_mean": round(float(np.mean([S[f"{t}:features_576"]["eff_rank"] for t in tags])), 2)}
    rep = {"experiment": name, "analysis_data_only": True, "vic_pooled": LAM, "extra_args": list(extra_args), "matrix": S, "checks": checks, "training": train_rep,
           "summary": {"supported": int(ok), "seeds_passing": int(sum(all(c.values()) for c in checks.values())), "museum_not_worse_than_nokd": bool(museum_not_worse),
                       **{f"vic_{k}": v for k, v in stats(list(VIC)).items()}, **{f"nokd_{k}": v for k, v in stats(list(NOKD)).items()}, **{f"ref_{k}": v for k, v in stats(REF).items()}, "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / out).write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
