#!/usr/bin/env python
"""E48: is the training necessary at all? Frozen baselines next to the trained student's pooled layer, with their cost.

  python scripts/e48_frozen_vs_trained.py   # CPU only (cached features; ImageNet MobileNetV3 embeddings computed once) -> reports/e48_frozen_vs_trained.json

Rows (all label-free, PCA-64 fitted on the training-allowed commoncatalog_ccby train originals, 601 photographs):
  imagenet_init:pca64      the ImageNet-pretrained MobileNetV3-Small pooled 576-d layer, no training at all (same export, same preprocessing)
  dino_global:pca64        frozen DINOv2-Small CLS ++ mean token (768-d), the generic embedding
  dino_style:pca64         frozen DINOv2-Small L4+L8 patch mean ++ std (1536-d), the teacher's style statistics
  explicit28_std           the 28 deterministic measured statistics, standardised on the fit set
  student_f576 x 3 seeds   the E43 teacher-distilled students' pooled layer (PCA-64), the current research candidate
Raw (un-reduced, L2-normalised) versions of the frozen rows are reported too. Cost: fp32 size, parameter count and
single-thread CPU latency at 224 px measured here for MobileNetV3-Small (ONNX) and DINOv2-Small (PyTorch).
Pre-registered rule (research/backlog.yaml E48): the training is necessary; supported = 1 iff the ImageNet-initialised
pooled PCA-64 fails at least one of {effective rank >= 8, own-source rank-1 <= 0.45, museum p@5 lower CI above chance,
within-uploader different-day AUC lower CI above 0.5} or its museum p@5 is more than 0.03 below the trained students'
seed-mean. The DINOv2 rows are reference only (they decide nothing here). Analysis halves only; no test half, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from style_benchmark import load_cache  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
REF = ["style_fs_v2_real_f576", "style_fs_v2_real_f576_s1", "style_fs_v2_real_f576_s2"]
INIT = "imagenet_init"


def export_imagenet_init():
    """checkpoints/student_imagenet_init: the trainer's architecture at its ImageNet initialisation (heads random, unused)."""
    out = P.CHECKPOINTS / f"student_{INIT}"
    if (out / "model.onnx").exists():
        return
    import torch
    from torch import nn
    from train_style_student import StyleStudent
    torch.manual_seed(0); net = StyleStudent(9).eval()

    class W(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): f, t, fam, p, kd = s.n(x); return t, fam, kd, f
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(W(net), (torch.zeros(1, 3, 224, 224),), str(out / "model.onnx"), input_names=["pixel_values"], output_names=["treated_logit", "family_logits", "style_embedding", "features_576"],
                      dynamic_axes={"pixel_values": {0: "batch"}}, opset_version=17, dynamo=False)
    (out / "meta.json").write_text(json.dumps({"model_version": "mobilenet_v3_small IMAGENET1K_V1, untrained heads (E48 control)", "outputs": ["treated_logit", "family_logits", "style_embedding", "features_576"], "input_size": 224,
                                               "label_sources": "none (ImageNet-pretrained backbone, no FilmLook training)"}, indent=1))


def fit_embed(tag, fit_paths):
    cache = P.EMBEDDINGS / f"student_fit_{tag}_features_576.npy"
    if cache.exists():
        return np.load(cache)
    Z = M.embed(tag, fit_paths, True, "features_576"); np.save(cache, Z); return Z


def cpu_latency():
    import torch
    import onnxruntime as ort
    torch.set_num_threads(1); so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(P.CHECKPOINTS / f"student_{REF[0]}" / "model.onnx"), so, providers=["CPUExecutionProvider"]); x = np.zeros((1, 3, 224, 224), np.float32); sess.run(None, {"pixel_values": x})
    ts = []
    for _ in range(30):
        t = time.perf_counter(); sess.run(None, {"pixel_values": x}); ts.append(time.perf_counter() - t)
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").eval(); px = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        m(pixel_values=px); td = []
        for _ in range(10):
            t = time.perf_counter(); m(pixel_values=px, output_hidden_states=True); td.append(time.perf_counter() - t)
    return {"mobilenet_v3_small": {"onnx_mb": round((P.CHECKPOINTS / f"student_{REF[0]}" / "model.onnx").stat().st_size / 1e6, 2), "cpu_1thread_ms_p50": round(1000 * float(np.median(ts)), 1), "params_m": 1.6},
            "dinov2_small": {"fp32_mb": round(sum(p.numel() for p in m.parameters()) * 4 / 1e6, 1), "cpu_1thread_ms_p50": round(1000 * float(np.median(td)), 1), "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "note": "PyTorch eager, hidden states on; plus the 28 explicit statistics for the hybrid teacher"},
            "cpu": "i9-14900KF dev workstation"}


def main():
    t0 = time.time(); export_imagenet_init()
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); A = dict(np.load(M.AIC.with_suffix(".npz"))); G = dict(np.load(M.CCF.with_suffix(".npz"))); Fit = dict(np.load(FIT.with_suffix(".npz")))
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    q = (st == "filmsynth") & (sp == "val") & treated; bank = (st == "source") & (sp == "val")
    assert q.sum() == len(rv) and bank.sum() == len(rb)
    S = {}

    def run(name, za, zg, zv, zb):
        S[name] = M.matrix(za, zg, zv, L, zb); print(M.fmt(name, S[name]), flush=True)
    for key in ("dino_global", "dino_style"):
        za, zg, zv, zb = A[key][L["ia"]], G[key][L["ig"]], Fc[key][q], Fc[key][bank]
        run(f"{key}:raw", za, zg, zv, zb); p = PCA64(Fit[key]); run(f"{key}:pca64", p(za), p(zg), p(zv), p(zb))
    mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6
    run("explicit28_std", (A["explicit"][L["ia"]] - mu) / sd, (G["explicit"][L["ig"]] - mu) / sd, (Fc["explicit"][q] - mu) / sd, (Fc["explicit"][bank] - mu) / sd)
    for tag in (INIT, *REF):
        za, zg, zv, zb = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); p = PCA64(fit_embed(tag, fit_paths))
        if tag == INIT:
            run(f"{tag}:raw576", za, zg, zv, zb)
        run(f"{tag}:pca64", p(za), p(zg), p(zv), p(zb))
    c = S[f"{INIT}:pca64"]; ref_mus = float(np.mean([S[f"{t}:pca64"]["museum_p5"]["precision@5"] for t in REF]))
    init_rule = {"eff_rank_ge_8": c["eff_rank"] >= 8, "own_source_le_0.45": c["own_source_rank1"] <= 0.45, "museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"], "within_auc_ci_low_above_half": c["within_uploader_auc"]["ci95"][0] > 0.5}
    init_passes = all(init_rule.values()); gap = round(ref_mus - c["museum_p5"]["precision@5"], 4)
    supported = (not init_passes) or gap > 0.03
    cost = cpu_latency()
    rep = {"experiment": "E48_frozen_vs_trained", "analysis_data_only": True, "fit_set": f"commoncatalog_ccby train originals ({len(fit_paths)})", "matrix": S, "cost": cost, "imagenet_init_rule": {k: bool(v) for k, v in init_rule.items()},
           "summary": {"supported": int(supported), "imagenet_init_passes_candidate_rule": bool(init_passes), "museum_gap_trained_minus_init": gap, "init_museum_p5": c["museum_p5"]["precision@5"], "trained_museum_mean": round(ref_mus, 4),
                       "init_within_auc": c["within_uploader_auc"]["auc"], "trained_within_auc_mean": round(float(np.mean([S[f"{t}:pca64"]["within_uploader_auc"]["auc"] for t in REF])), 4), "init_own_source": c["own_source_rank1"], "init_eff_rank": c["eff_rank"],
                       "dino_style_pca64_museum": S["dino_style:pca64"]["museum_p5"]["precision@5"], "dino_style_pca64_own_source": S["dino_style:pca64"]["own_source_rank1"], "dino_global_pca64_museum": S["dino_global:pca64"]["museum_p5"]["precision@5"],
                       "dino_global_pca64_own_source": S["dino_global:pca64"]["own_source_rank1"], "explicit28_museum": S["explicit28_std"]["museum_p5"]["precision@5"], "mobilenet_ms": cost["mobilenet_v3_small"]["cpu_1thread_ms_p50"], "dinov2_ms": cost["dinov2_small"]["cpu_1thread_ms_p50"],
                       "seconds": round(time.time() - t0, 1)}}
    (P.REPORTS / "e48_frozen_vs_trained.json").write_text(json.dumps(rep, indent=1, default=str)); print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
