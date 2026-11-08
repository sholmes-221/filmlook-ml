#!/usr/bin/env python
"""E47: which label-free reduction of the student's pooled layer should a Mini Tool ship?

  python scripts/e47_pooled_reductions.py   # CPU only (cached pooled features of the three E43 seeds) -> reports/e47_pooled_reductions.json

Reductions of the 576-d pooled features, every one fitted on the student's own embeddings of the training-allowed
commoncatalog_ccby train originals (601 photographs) and applied unchanged to the evaluation sets:
  raw576                 L2-normalised pooled features (no reduction)
  pca{8,16,32,64,128}    centred PCA to k dimensions
  pcaw64                 PCA-64 followed by whitening (unit variance per component)
  rproj64                Gaussian random projection to 64 dimensions (centred; seed 0)
  pca64_int8             PCA-64 quantised to int8 with one scale per component (what a compact export would store)
  pca32_int8             the same at 32 dimensions
Each goes through the real-photograph matrix on the analysis halves for three seeds.
Pre-registered rule (research/backlog.yaml E47): supported = 1 iff for every seed PCA-32 has effective rank >= 8,
own-source rank-1 <= 0.45, museum p@5 lower CI above chance and within-uploader different-day AUC lower CI above 0.5,
and the PCA-32 seed-mean museum p@5 and within-uploader AUC are each within 0.02 of the PCA-64 seed-means. No test
half, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import fit_embed  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

SEEDS = ["style_fs_v2_real_f576", "style_fs_v2_real_f576_s1", "style_fs_v2_real_f576_s2"]
PRIMARY = "pca32"


class Whiten:
    def __init__(self, pca, Zfit):
        self.pca = pca; self.sd = pca(Zfit).std(0) + 1e-6

    def __call__(self, Z):
        return self.pca(Z) / self.sd


class Int8:
    """Per-component symmetric int8 quantisation with the scale fitted on the fit set (max |value| -> 127)."""
    def __init__(self, red, Zfit):
        self.red = red; self.scale = np.abs(red(Zfit)).max(0) / 127.0 + 1e-9

    def __call__(self, Z):
        return np.clip(np.round(self.red(Z) / self.scale), -127, 127).astype(np.int8).astype(np.float32) * self.scale


class RProj:
    def __init__(self, Zfit, k=64, seed=0):
        rng = np.random.default_rng(seed); self.mu = Zfit.mean(0); self.R = rng.standard_normal((Zfit.shape[1], k)) / np.sqrt(k)

    def __call__(self, Z):
        return (Z - self.mu) @ self.R


def reductions(Zfit):
    red = {"raw576": lambda Z: Z}
    for k in (8, 16, 32, 64, 128):
        red[f"pca{k}"] = PCA64(Zfit, k)
    red["pcaw64"] = Whiten(PCA64(Zfit, 64), Zfit); red["rproj64"] = RProj(Zfit, 64)
    red["pca64_int8"] = Int8(PCA64(Zfit, 64), Zfit); red["pca32_int8"] = Int8(PCA64(Zfit, 32), Zfit)
    return red


def main():
    t0 = time.time()
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); S = {}
    for tag in SEEDS:
        za, zg, zv, zb = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); Zfit = fit_embed(tag, fit_paths)
        for name, f in reductions(Zfit).items():
            n = f"{tag}:{name}"; S[n] = M.matrix(f(za), f(zg), f(zv), L, f(zb)); print(M.fmt(n, S[n]), flush=True)
    names = ["raw576", "pca8", "pca16", "pca32", "pca64", "pca128", "pcaw64", "rproj64", "pca64_int8", "pca32_int8"]
    mean = lambda name, f: round(float(np.mean([f(S[f"{t}:{name}"]) for t in SEEDS])), 4)  # noqa: E731
    table = {name: {"museum_mean": mean(name, lambda r: r["museum_p5"]["precision@5"]), "museum_ci_low_min": round(float(min(S[f"{t}:{name}"]["museum_p5"]["ci95"][0] for t in SEEDS)), 4), "stock_line_mean": mean(name, lambda r: r["stock_line_p5"]["precision@5"]),
                    "within_auc_mean": mean(name, lambda r: r["within_uploader_auc"]["auc"]), "within_ci_low_min": round(float(min(S[f"{t}:{name}"]["within_uploader_auc"]["ci95"][0] for t in SEEDS)), 4), "colour_auc_mean": mean(name, lambda r: r["colour_only_auc"]["auc"]),
                    "synth_mean": mean(name, lambda r: r["synth_family_p5"]["precision@5"]), "own_source_mean": mean(name, lambda r: r["own_source_rank1"]), "own_source_max": round(float(max(S[f"{t}:{name}"]["own_source_rank1"] for t in SEEDS)), 4),
                    "eff_rank_min": round(float(min(S[f"{t}:{name}"]["eff_rank"] for t in SEEDS)), 2), "artist_share_mean": mean(name, lambda r: r["artist_share"]), "uploader_share_mean": mean(name, lambda r: r["uploader_share"]),
                    "photographer_p5_mean": mean(name, lambda r: r["photographer_p5"]["precision@5"] or 0.0)} for name in names}
    checks = {}
    for tag in SEEDS:
        p = S[f"{tag}:{PRIMARY}"]
        checks[tag] = {"eff_rank_ge_8": p["eff_rank"] >= 8, "own_source_le_0.45": p["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": p["museum_p5"]["ci95"][0] > p["museum_p5"]["chance"], "within_auc_ci_low_above_half": p["within_uploader_auc"]["ci95"][0] > 0.5}
    close = {"museum_within_0.02": abs(table[PRIMARY]["museum_mean"] - table["pca64"]["museum_mean"]) <= 0.02, "within_auc_within_0.02": abs(table[PRIMARY]["within_auc_mean"] - table["pca64"]["within_auc_mean"]) <= 0.02}
    ok = all(all(c.values()) for c in checks.values()) and all(close.values())
    chance = {"museum": S[f"{SEEDS[0]}:pca64"]["museum_p5"]["chance"], "stock_line": S[f"{SEEDS[0]}:pca64"]["stock_line_p5"]["chance"], "own_source": round(1 / len(rb), 4)}
    rep = {"experiment": "E47_pooled_reductions", "analysis_data_only": True, "fit_set": f"commoncatalog_ccby train originals ({len(fit_paths)})", "chance": chance, "seed_table": table, "matrix": S, "checks": checks,
           "summary": {"supported": int(ok), "primary": PRIMARY, "seeds_passing": int(sum(all(c.values()) for c in checks.values())), **{k: bool(v) for k, v in close.items()},
                       **{f"{n}_{k}": table[n][k] for n in ("raw576", "pca8", "pca16", "pca32", "pca64", "pca128", "pcaw64", "rproj64", "pca64_int8", "pca32_int8") for k in ("museum_mean", "within_auc_mean", "own_source_mean", "eff_rank_min")},
                       "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e47_pooled_reductions.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
