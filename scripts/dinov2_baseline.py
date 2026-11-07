#!/usr/bin/env python
"""Reproducible DINOv2-small baseline: Stage A retrieval + Stage B frozen-embedding heads.

  python scripts/dinov2_baseline.py                 # uses cached embeddings; writes reports/dinov2_baseline.json
  python scripts/dinov2_baseline.py --latency       # also times DINOv2-small forward passes on CPU and GPU

Data: eligible originals (registry-approved, not quarantined), with the existing uploader-grouped splits.
Seeds are fixed. No Claude labels are used. Artifacts: checkpoints/dinov2_baseline_v0/ (experimental, not
promoted).
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

from filmlook_ml import dinov2_baseline as B
from filmlook_ml import paths as P
from filmlook_ml.annotation.workflow import discover
from filmlook_ml.compliance.preflight import require_training_preflight
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import filter_manifest, load_quarantine
from filmlook_ml.datasets.registry import Registry


def latency(n=32):
    import torch
    from filmlook_infer.pixels import dinov2_pixels
    from filmlook_infer.imageio import load_image
    from filmlook_ml.models.filmlook_model import DinoEmbedder
    man = [r for r in read_jsonl(P.MANIFESTS / "manifest.jsonl") if r["edit_type"] == "original"][:n]
    px = np.stack([dinov2_pixels(load_image(r["path"], max_side=1280), 224) for r in man])
    model = DinoEmbedder().eval()
    out = {"params": int(sum(p.numel() for p in model.parameters())),
           "fp32_mb": round(sum(p.numel() for p in model.parameters()) * 4 / 1e6, 1)}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        x = torch.from_numpy(px[:1])
        with torch.inference_mode():
            model(x)
            ts = []
            for i in range(n):
                t0 = time.perf_counter()
                model(torch.from_numpy(px[i:i + 1]))
                ts.append(time.perf_counter() - t0)
        out[f"cpu_{threads}thread_ms_p50"] = round(1000 * float(np.median(ts)), 1)
    if torch.cuda.is_available():
        m = model.cuda()
        with torch.inference_mode():
            m(torch.from_numpy(px[:1]).cuda())
            torch.cuda.synchronize()
            ts = []
            for i in range(n):
                t0 = time.perf_counter()
                m(torch.from_numpy(px[i:i + 1]).cuda())
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            out["gpu_batch1_ms_p50"] = round(1000 * float(np.median(ts)), 2)
            xb = torch.from_numpy(px).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(xb)
            torch.cuda.synchronize()
            out[f"gpu_batch{n}_ms_per_image"] = round(1000 * (time.perf_counter() - t0) / n, 2)
        out["gpu"] = torch.cuda.get_device_name(0) + " (shared with another job)"
    out["cpu"] = "development workstation (i9-14900KF); NOT the Oracle A1 deployment target"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    require_training_preflight()
    np.random.seed(0)
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, skipped = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    rows, emb, feats, E, F = B.load(eligible, P.EMBEDDINGS / B.EMB_FILE, P.EMBEDDINGS / B.FEAT_FILE)
    split = np.array([r["split"] for r in rows])
    groups = np.array([r["group_id"] for r in rows], dtype=object)
    Z, mu = B.normalise(emb, split == "train")
    rep = {"baseline_version": B.BASELINE_VERSION, "n_images": len(rows), "skipped": dict(skipped),
           "splits": {s: int((split == s).sum()) for s in ("train", "val", "test")},
           "data_hash": hashlib.sha256("".join(sorted(r["sha256"] for r in rows)).encode()).hexdigest()[:16]}

    # Stage A: retrieval
    Fz = (feats - feats[split == "train"].mean(0)) / (feats[split == "train"].std(0) + 1e-8)
    Fz = Fz / (np.linalg.norm(Fz, axis=1, keepdims=True) + 1e-8)
    cap = lambda r: r.get("medium") or "pd12m"  # noqa: E731
    rep["stage_a"] = {
        "capture_type_retrieval": {"dinov2": B.retrieval_eval(Z, rows, cap, split == "train", split == "test"),
                                   "measured_stats": B.retrieval_eval(Fz, rows, cap, split == "train", split == "test")},
        "same_uploader_retrieval": {"dinov2": B.same_group_retrieval(Z, rows), "measured_stats": B.same_group_retrieval(Fz, rows)},
        "edit_invariance": B.edit_invariance(Z, rows, E, mu, filter_manifest(man), split == "test"),
    }

    # Stage B: heads
    attrs = [B.attribute_labels(F[r["sha256"]]) for r in rows]
    targets = {"capture_type": np.array([r.get("medium") if r.get("medium") in B.CAPTURE_CLASSES else None
                                         for r in rows], dtype=object)}
    for k in attrs[0]:
        targets[k] = np.array([x[k] for x in attrs], dtype=object)
    rep["stage_b"], heads = {}, {}
    for name, y in targets.items():
        res, fitted, classes = B.run_head(name, y, {"dinov2_linear": Z, "measured_stats_linear": feats}, split, groups, Z)
        rep["stage_b"][name] = res
        sc, clf = fitted["dinov2_linear"]
        heads[name] = (sc, clf, classes)
    rep["stage_b_summary"] = {n: {k: r[k]["test"]["macro_f1"] for k in ("dinov2_linear", "measured_stats_linear",
                                                                         "knn10_dinov2", "majority")}
                              for n, r in rep["stage_b"].items()}
    if a.latency:
        rep["latency"] = latency()
    ck = P.CHECKPOINTS / "dinov2_baseline_v0"
    B.save_heads(ck, heads, mu, {"baseline_version": B.BASELINE_VERSION, "backbone": "facebook/dinov2-small (frozen)",
                                 "embedding": "CLS ++ mean patch, 224 px, max_side 1280, train-mean centred, L2",
                                 "taxonomy_version": "taxonomy-db-v1 (not used by heads)",
                                 "label_definitions": {"capture_type": "EXIF-derived medium (metadata)",
                                                       "others": "fixed thresholds on measured statistics "
                                                                 "(filmlook_infer.style.THRESHOLDS)"},
                                 "data_hash": rep["data_hash"], "status": "experimental baseline, not promoted"})
    rep["artifact"] = str(ck.relative_to(P.ROOT))
    (P.REPORTS / "dinov2_baseline.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_images", "splits", "stage_a", "stage_b_summary", *(("latency",) if a.latency else ()))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
