#!/usr/bin/env python
"""E42: does the end-to-end student's 576-d pooled representation hold a higher-rank, content-invariant film signal
than the 64-d distilled head?

  python scripts/e42_student_penultimate.py    # retrains the baseline recipe once with the 576-d export (~1 GPU-minute),
                                               # -> reports/e42_student_penultimate.json

E40/E41: on frozen DINOv2-Small statistics every head that raises rank lets scene content in. The MobileNetV3 students
are trained end to end on content-free synthetic supervision (treated / family / six parameters) plus distillation, and
their 64-d head is content-invariant (own-source rank-1 0.10-0.18) but collapsed; the trainer's 576-d own-source
rank-1 (0.31-0.42) says the penultimate layer keeps more than the head. The baseline recipe (E23b, seed 0) is retrained
with the pooled features exported, and both outputs go through the real-photograph matrix.
Pre-registered rule (research/backlog.yaml E42): supported = 1 iff `features_576` has effective rank >= 8, own-source
rank-1 <= 0.45, museum p@5 and within-uploader different-day AUC not below the 64-d head's, and synthetic family p@5
within 0.10 of the 64-d head's. No test half, no BlueNeg.
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from filmlook_ml import paths as P  # noqa: E402

PY = sys.executable; TAG = "style_fs_v2_real_f576"


def main():
    t0 = time.time()
    if not (P.CHECKPOINTS / f"student_{TAG}" / "model.onnx").exists():
        r = subprocess.run([PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil", "--tag", TAG], cwd=P.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed: {r.stderr[-800:]}")
    import student_real_matrix as M
    sys.argv = ["student_real_matrix", "--tags", f"{TAG}:features_576", "--reference", TAG, "--out", "e42_student_penultimate_matrix.json"]
    S = M.main()["students"]; h, f = S[TAG], S[f"{TAG}:features_576"]
    checks = {"eff_rank_ge_8": f["eff_rank"] >= 8, "own_source_le_0.45": f["own_source_rank1"] <= 0.45, "museum_not_worse": f["museum_p5"]["precision@5"] >= h["museum_p5"]["precision@5"],
              "within_auc_not_worse": f["within_uploader_auc"]["auc"] >= h["within_uploader_auc"]["auc"], "synth_cost_ok": f["synth_family_p5"]["precision@5"] >= h["synth_family_p5"]["precision@5"] - 0.10}
    rep = {"experiment": "E42_student_penultimate", "analysis_data_only": True, "matrix": S,
           "summary": {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "f576_eff_rank": f["eff_rank"], "head_eff_rank": h["eff_rank"], "f576_own_source_rank1": f["own_source_rank1"], "head_own_source_rank1": h["own_source_rank1"],
                       "f576_museum_p5": f["museum_p5"]["precision@5"], "head_museum_p5": h["museum_p5"]["precision@5"], "f576_within_auc": f["within_uploader_auc"]["auc"], "head_within_auc": h["within_uploader_auc"]["auc"],
                       "f576_stock_line_p5": f["stock_line_p5"]["precision@5"], "head_stock_line_p5": h["stock_line_p5"]["precision@5"], "f576_synth_family_p5": f["synth_family_p5"]["precision@5"], "head_synth_family_p5": h["synth_family_p5"]["precision@5"],
                       "f576_artist_share": f["artist_share"], "head_artist_share": h["artist_share"], "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e42_student_penultimate.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
