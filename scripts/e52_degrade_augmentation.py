#!/usr/bin/env python
"""E52: does re-encoding the training crops close the JPEG / rescale gap of E49 without dulling the edit sensitivity?

  python scripts/e52_degrade_augmentation.py   # retrains three students (~1 GPU-minute each) -> reports/e52_degrade_augmentation.json

E49: the pooled PCA-64 keeps its nearest neighbour under crops and flips (0.96-1.00) but loses it under JPEG q50 and
0.5x rescale 5-11 % of the time (one seed 11.5 %, which falsified the 0.9 bound). Uploads arrive re-encoded. Here the
E23b recipe (teacher-distilled, the E43 reference) is retrained with `--aug-degrade 0.5`: half of every batch's crops
are re-encoded, JPEG q40-70 or 0.5x rescale and back, before the network sees them. Three seeds. Readings: the E49
protocol on the commoncatalog_ccby validation originals and their manifest edits (no reference rows recomputed), and
the real-photograph matrix on the analysis halves for the PCA-64.
Pre-registered rule (research/backlog.yaml E52): supported = 1 iff for every seed own-image rank-1 >= 0.9 under
centre crop, flip, JPEG q50 and rescale; edit-type p@5 >= the E49 reference seed-mean - 0.03 (0.436); edited own-source
rank-1 <= 0.5; and on the matrix effective rank >= 8, own-source rank-1 <= 0.45, museum p@5 lower CI above chance and
within-uploader different-day AUC lower CI above 0.5. No test half, no BlueNeg.
"""
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import e49_robustness as E49  # noqa: E402
import student_real_matrix as M  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import fit_embed  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

PY = sys.executable
DEG = {"style_fs_v2_real_f576_deg": 0, "style_fs_v2_real_f576_deg_s1": 1, "style_fs_v2_real_f576_deg_s2": 2}
REF_EDIT_P5 = 0.4662          # E49 reference seed-mean (reports/e49_robustness.json)


def main():
    t0 = time.time()
    for tag, seed in DEG.items():
        if (P.CHECKPOINTS / f"student_{tag}" / "model.onnx").exists():
            continue
        r = subprocess.run([PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil", "--aug-degrade", "0.5", "--tag", tag, "--seed", str(seed)], cwd=P.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed for {tag}: {r.stderr[-800:]}")
    E49.main(REF=list(DEG), name="E52_degrade_augmentation_robustness", out_name="e52_robustness.json", refs=False)
    rob = json.loads((P.REPORTS / "e52_robustness.json").read_text())
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); S = {}; checks = {}
    for tag in DEG:
        za, zg, zv, zb = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); p = PCA64(fit_embed(tag, fit_paths))
        S[f"{tag}:pca64"] = M.matrix(p(za), p(zg), p(zv), L, p(zb)); print(M.fmt(f"{tag}:pca64", S[f"{tag}:pca64"]), flush=True)
        r = rob["representations"][f"{tag}:pca64"]; m = S[f"{tag}:pca64"]
        checks[tag] = {"invariant_rank1_ge_0.9": min(r["invariance"][k]["own_rank1"] for k in E49.INVARIANT) >= 0.9, "edit_p5_kept": r["edit_type_p5"]["precision@5"] >= REF_EDIT_P5 - 0.03, "edited_own_source_le_0.5": r["content"]["edited_own_source_rank1"] <= 0.5,
                       "eff_rank_ge_8": m["eff_rank"] >= 8, "own_source_le_0.45": m["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": m["museum_p5"]["ci95"][0] > m["museum_p5"]["chance"], "within_auc_ci_low_above_half": m["within_uploader_auc"]["ci95"][0] > 0.5}
    ok = all(all(c.values()) for c in checks.values()); mean = lambda f: round(float(np.mean([f(t) for t in DEG])), 4)  # noqa: E731
    inv = lambda t, k: rob["representations"][f"{t}:pca64"]["invariance"][k]["own_rank1"]  # noqa: E731
    rep = {"experiment": "E52_degrade_augmentation", "analysis_and_validation_data_only": True, "matrix": S, "robustness_report": "reports/e52_robustness.json", "checks": checks,
           "summary": {"supported": int(ok), "seeds_passing": int(sum(all(c.values()) for c in checks.values())), "min_invariant_rank1": round(float(min(inv(t, k) for t in DEG for k in E49.INVARIANT)), 4),
                       **{f"{k}_rank1_mean": mean(lambda t: inv(t, k)) for k in E49.PERTURB}, "edit_p5_mean": mean(lambda t: rob["representations"][f"{t}:pca64"]["edit_type_p5"]["precision@5"]), "ref_edit_p5": REF_EDIT_P5,
                       "edited_own_source_mean": mean(lambda t: rob["representations"][f"{t}:pca64"]["content"]["edited_own_source_rank1"]), "film_emulation_disp_rel_mean": mean(lambda t: rob["representations"][f"{t}:pca64"]["displacement"]["film_emulation"]["relative_to_crop80"]),
                       "museum_mean": mean(lambda t: S[f"{t}:pca64"]["museum_p5"]["precision@5"]), "within_auc_mean": mean(lambda t: S[f"{t}:pca64"]["within_uploader_auc"]["auc"]), "own_source_mean": mean(lambda t: S[f"{t}:pca64"]["own_source_rank1"]),
                       "eff_rank_min": round(float(min(S[f"{t}:pca64"]["eff_rank"] for t in DEG)), 2), "synth_mean": mean(lambda t: S[f"{t}:pca64"]["synth_family_p5"]["precision@5"]), "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e52_degrade_augmentation.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
