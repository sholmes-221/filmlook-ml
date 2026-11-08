#!/usr/bin/env python
"""E45: does the student's pooled-layer signal (E42/E43) come from the teacher at all?

  python scripts/e45_pooled_without_teacher.py   # retrains two students (~1 GPU-minute each) -> reports/e45_pooled_without_teacher.json

Variants of the E23b recipe, both exporting the 576-d pooled layer:
  style_fs_v2_real_nokd    `--kd 0`: synthetic supervision only (treated / family / six parameters), no distillation target
  style_fs_v2_real_vic3f   `--kd-target supcon_vic3`: the anti-collapse target of E38 (its pooled layer was never read)
Each pooled layer and its PCA-64 (fitted on the student's embeddings of the CC-BY train originals) go through the real-
photograph matrix on the analysis halves, next to the E43 seed-0 student.
Pre-registered rule (research/backlog.yaml E45): supported = 1 iff the no-teacher student's PCA-64 has effective rank
>= 8, own-source rank-1 <= 0.45, museum p@5 lower CI above chance, within-uploader different-day AUC lower CI above 0.5
and synthetic family p@5 within 0.10 of the E43 seed-0 PCA-64 (0.629). No test half, no BlueNeg.
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

PY = sys.executable
BASE = [PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil"]
RUNS = {"style_fs_v2_real_nokd": ["--kd", "0"], "style_fs_v2_real_vic3f": ["--kd-target", "supcon_vic3"]}
REF = "style_fs_v2_real_f576"; PRIMARY = "style_fs_v2_real_nokd"


def main():
    t0 = time.time()
    for tag, extra in RUNS.items():
        if (P.CHECKPOINTS / f"student_{tag}" / "model.onnx").exists():
            continue
        r = subprocess.run([*BASE, "--tag", tag, *extra], cwd=P.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed for {tag}: {r.stderr[-800:]}")
    import student_real_matrix as M
    from e43_penultimate_pca import PCA64
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); S = {}
    for tag in (REF, *RUNS):
        za, zg, zv, zb = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); S[f"{tag}:features_576"] = M.matrix(za, zg, zv, L, zb); print(M.fmt(f"{tag}:features_576", S[f"{tag}:features_576"]), flush=True)
        pca = PCA64(M.embed(tag, fit_paths, True, "features_576")); n = f"{tag}:pca64"; S[n] = M.matrix(pca(za), pca(zg), pca(zv), L, pca(zb)); print(M.fmt(n, S[n]), flush=True)
        if tag != REF:
            za, zg, zv, zb = M.student_features(tag, ra, rg, rv, rb, L["ia"], L["ig"]); S[f"{tag}:head"] = M.matrix(za, zg, zv, L, zb); print(M.fmt(f"{tag}:head", S[f"{tag}:head"]), flush=True)
    c, ref = S[f"{PRIMARY}:pca64"], S[f"{REF}:pca64"]
    checks = {"eff_rank_ge_8": c["eff_rank"] >= 8, "own_source_le_0.45": c["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"],
              "within_auc_ci_low_above_half": c["within_uploader_auc"]["ci95"][0] > 0.5, "synth_cost_ok": c["synth_family_p5"]["precision@5"] >= ref["synth_family_p5"]["precision@5"] - 0.10}
    rep = {"experiment": "E45_pooled_without_teacher", "analysis_data_only": True, "matrix": S,
           "summary": {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY,
                       **{f"{who}_{k}": S[n][k]["precision@5"] for who, n in (("nokd", f"{PRIMARY}:pca64"), ("vic3f", "style_fs_v2_real_vic3f:pca64"), ("ref", f"{REF}:pca64")) for k in ("museum_p5", "stock_line_p5", "synth_family_p5")},
                       **{f"{who}_own_source": S[n]["own_source_rank1"] for who, n in (("nokd", f"{PRIMARY}:pca64"), ("vic3f", "style_fs_v2_real_vic3f:pca64"), ("ref", f"{REF}:pca64"))},
                       **{f"{who}_eff_rank": S[n]["eff_rank"] for who, n in (("nokd", f"{PRIMARY}:pca64"), ("vic3f", "style_fs_v2_real_vic3f:pca64"), ("ref", f"{REF}:pca64"))},
                       "nokd_within_auc": c["within_uploader_auc"]["auc"], "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e45_pooled_without_teacher.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
