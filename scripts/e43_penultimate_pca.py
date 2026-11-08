#!/usr/bin/env python
"""E43: is the student's penultimate signal (E42) seed-robust, and does a label-free 64-d PCA of it keep it?

  python scripts/e43_penultimate_pca.py    # retrains the baseline recipe with seeds 1 and 2 (576-d export, ~1 GPU-minute each),
                                           # -> reports/e43_penultimate_pca.json

For each of three seeds of the baseline recipe (E23b; seed 0 from E42) the real-photograph matrix is computed for the
64-d distilled head, the 576-d pooled features, and a PCA-64 of the pooled features fitted on the student's own
embeddings of the training-allowed commoncatalog_ccby train originals (601 photographs; no evaluation data in the fit).
Pre-registered rule (research/backlog.yaml E43): supported = 1 iff for every seed the PCA-64 reduction has effective rank
>= 8, own-source rank-1 <= 0.45, museum p@5 lower CI above chance, within-uploader different-day AUC lower CI above 0.5,
and synthetic family p@5 within 0.10 of that seed's 64-d head. No test half, no BlueNeg.
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
SEEDS = {"style_fs_v2_real_f576": 0, "style_fs_v2_real_f576_s1": 1, "style_fs_v2_real_f576_s2": 2}


def main():
    t0 = time.time()
    for tag, seed in SEEDS.items():
        if (P.CHECKPOINTS / f"student_{tag}" / "model.onnx").exists():
            continue
        r = subprocess.run([PY, "scripts/train_style_student.py", "--synth", "film_synth_v2", "--train-generators", "filmsynth", "--no-blueneg-eval", "--real-distil", "--tag", tag, "--seed", str(seed)], cwd=P.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"training failed for {tag}: {r.stderr[-800:]}")
    import student_real_matrix as M
    from frozen_head_choice import PCAW
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); S = {}
    for tag in SEEDS:
        for output in ("style_embedding", "features_576"):
            za, zg, zv, zb = M.student_features(f"{tag}:{output}" if output != "style_embedding" else tag, ra, rg, rv, rb, L["ia"], L["ig"])
            name = f"{tag}:{output}"; S[name] = M.matrix(za, zg, zv, L, zb); print(M.fmt(name, S[name]), flush=True)
            if output == "features_576":
                fit = M.embed(tag, fit_paths, True, output); pca = PCA64(fit)
                name = f"{tag}:features_576_pca64_ccby"; S[name] = M.matrix(pca(za), pca(zg), pca(zv), L, pca(zb)); print(M.fmt(name, S[name]), flush=True)
    checks = {}
    for tag in SEEDS:
        h, p = S[f"{tag}:style_embedding"], S[f"{tag}:features_576_pca64_ccby"]
        checks[tag] = {"eff_rank_ge_8": p["eff_rank"] >= 8, "own_source_le_0.45": p["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": p["museum_p5"]["ci95"][0] > p["museum_p5"]["chance"],
                       "within_auc_ci_low_above_half": p["within_uploader_auc"]["ci95"][0] > 0.5, "synth_cost_ok": p["synth_family_p5"]["precision@5"] >= h["synth_family_p5"]["precision@5"] - 0.10}
    ok = all(all(c.values()) for c in checks.values()); pca_names = [f"{t}:features_576_pca64_ccby" for t in SEEDS]
    mean = lambda f: round(float(np.mean([f(S[n]) for n in pca_names])), 4)  # noqa: E731
    rep = {"experiment": "E43_penultimate_pca", "analysis_data_only": True, "fit_set": f"commoncatalog_ccby train originals ({len(fit_paths)})", "matrix": S, "checks": checks,
           "summary": {"supported": int(ok), "seeds_passing": int(sum(all(c.values()) for c in checks.values())), "pca_museum_mean": mean(lambda r: r["museum_p5"]["precision@5"]), "pca_within_auc_mean": mean(lambda r: r["within_uploader_auc"]["auc"]),
                       "pca_stock_line_mean": mean(lambda r: r["stock_line_p5"]["precision@5"]), "pca_synth_mean": mean(lambda r: r["synth_family_p5"]["precision@5"]), "pca_own_source_mean": mean(lambda r: r["own_source_rank1"]),
                       "pca_eff_rank_mean": mean(lambda r: r["eff_rank"]), "pca_artist_share_mean": mean(lambda r: r["artist_share"]), "head_museum_mean": round(float(np.mean([S[f"{t}:style_embedding"]["museum_p5"]["precision@5"] for t in SEEDS])), 4),
                       "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e43_penultimate_pca.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


class PCA64:
    """Plain PCA to 64 dims (centred, not whitened) fitted on one set."""
    def __init__(self, Z, k=64):
        self.mu = Z.mean(0); U, s, Vt = np.linalg.svd(Z - self.mu, full_matrices=False); self.V = Vt[:k].T

    def __call__(self, Z):
        return (Z - self.mu) @ self.V


if __name__ == "__main__":
    main()
