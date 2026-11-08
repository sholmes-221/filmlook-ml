#!/usr/bin/env python
"""E38: does a non-collapsed distillation target give the lightweight student the teacher's real-photograph signal?

  python scripts/e38_student_targets.py        # trains three students (E23b recipe, ~1 GPU-minute each), then
                                               # -> reports/e38_student_targets.json (real-photograph matrix, analysis halves)

Students (MobileNetV3-Small, synthetic supervision + real distillation inputs, `scripts/train_style_student.py`):
  style_fs_v2_real_s1    the E23b recipe with seed 1 (seed control for the baseline student_style_fs_v2_real)
  style_fs_v2_real_vic3  distilled from the SupCon + VICReg (lambda 3) head of E34 (effective rank ~10)
  style_fs_v2_real_pcaw  distilled from the label-free whitened PCA-64 of the L4+L8 statistics (E33; rank ~40)
Pre-registered rule (research/backlog.yaml E38), primary = style_fs_v2_real_vic3 against the baseline student: supported = 1
iff museum p@5 lower CI above chance, within-uploader different-day AUC lower CI above 0.5, stock-line p@5 >= baseline -
0.01, scanner-probe accuracy <= baseline + 0.05, synthetic family p@5 >= baseline - 0.10, effective rank >= 8.
The seed control bounds what a seed change alone does to the real metrics. No test half, no BlueNeg.
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from filmlook_ml import paths as P  # noqa: E402

PY = sys.executable
BASE = [PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil"]
RUNS = [("style_fs_v2_real_s1", ["--seed", "1"]), ("style_fs_v2_real_vic3", ["--kd-target", "supcon_vic3"]), ("style_fs_v2_real_pcaw", ["--kd-target", "pcaw64"])]
PRIMARY, BASELINE = "style_fs_v2_real_vic3", "style_fs_v2_real"


def main():
    t0 = time.time(); trained = {}
    for tag, extra in RUNS:
        ck = P.CHECKPOINTS / f"student_{tag}" / "model.onnx"
        if ck.exists():
            trained[tag] = "existing"; continue
        r = subprocess.run([*BASE, "--tag", tag, *extra], cwd=P.ROOT, capture_output=True, text=True)
        trained[tag] = "ok" if r.returncode == 0 else f"failed rc {r.returncode}: {r.stderr[-800:]}"
        print(tag, trained[tag][:200], flush=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed for {tag}")
    import student_real_matrix as M
    sys.argv = ["student_real_matrix", "--tags", ",".join(t for t, _ in RUNS), "--reference", BASELINE, "--out", "e38_student_targets_matrix.json"]
    out = M.main(); S = out["students"]; b, c = S[BASELINE], S[PRIMARY]
    checks = {"museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"], "within_auc_ci_low_above_half": c["within_uploader_auc"]["ci95"][0] > 0.5,
              "stock_line_not_worse": c["stock_line_p5"]["precision@5"] >= b["stock_line_p5"]["precision@5"] - 0.01,
              "scanner_leak_ok": (c["scanner_probe"]["accuracy"] or 0) <= (b["scanner_probe"]["accuracy"] or 0) + 0.05,
              "synth_cost_ok": c["synth_family_p5"]["precision@5"] >= b["synth_family_p5"]["precision@5"] - 0.10, "eff_rank_ok": c["eff_rank"] >= 8}
    rep = {"experiment": "E38_student_targets", "analysis_data_only": True, "trained": trained, "matrix": S,
           "summary": {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY, "baseline": BASELINE,
                       **{f"{who}_{k}": S[t][k]["precision@5"] for who, t in (("primary", PRIMARY), ("baseline", BASELINE), ("seed1", "style_fs_v2_real_s1"), ("pcaw", "style_fs_v2_real_pcaw")) for k in ("museum_p5", "stock_line_p5", "synth_family_p5")},
                       **{f"{who}_within_auc": S[t]["within_uploader_auc"]["auc"] for who, t in (("primary", PRIMARY), ("baseline", BASELINE), ("seed1", "style_fs_v2_real_s1"), ("pcaw", "style_fs_v2_real_pcaw"))},
                       **{f"{who}_eff_rank": S[t]["eff_rank"] for who, t in (("primary", PRIMARY), ("baseline", BASELINE), ("pcaw", "style_fs_v2_real_pcaw"))}, "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e38_student_targets.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
