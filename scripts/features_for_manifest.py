#!/usr/bin/env python
"""Frozen-teacher feature pass (explicit 28 + DINOv2-Small global + L4/L8 statistics) for a manifest subset.

  python scripts/features_for_manifest.py --dataset commoncatalog_ccby --split train --edit-type original --tag ccby_train_originals
      -> data/embeddings/ccby_train_originals_features_v1.{npz,json}

Keeps the manifest rows (uploader `group_id`, medium, subjects) next to the arrays so head experiments can use the
photographer structure without touching any evaluation set. Training-eligibility is read from configs/datasets.yaml
through the registry gate; the pass refuses datasets that are not training-allowed unless --allow-eval-only is given
(the features then carry `training_use_allowed: false`).
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--edit-type", default="original")
    ap.add_argument("--tag", required=True); ap.add_argument("--manifest", default=str(P.DATA / "manifests" / "manifest.jsonl")); ap.add_argument("--allow-eval-only", action="store_true")
    ap.add_argument("--students", default="", help="comma-separated student checkpoint tags whose 64-d style embeddings are added (e.g. student_style_fs_v2_real)")
    a = ap.parse_args(); t0 = time.time()
    import torch
    from PIL import Image
    allowed = Registry.load().dataset_allowed(a.dataset)
    if not allowed and not a.allow_eval_only:
        raise SystemExit(f"{a.dataset} is not training-allowed in configs/datasets.yaml; pass --allow-eval-only for an evaluation-only feature cache")
    rows = [r for r in read_jsonl(a.manifest) if r.get("dataset_id") == a.dataset and (not a.split or r.get("split") == a.split) and (not a.edit_type or r.get("edit_type") == a.edit_type)]
    if not rows:
        raise SystemExit("no rows selected")
    with Pool(16) as pool:
        arrs = pool.map(V3._load, [(r["path"], r["image_id"], True, True) for r in rows], chunksize=8)
    ims = [Image.fromarray(x) for x in arrs]
    explicit = np.stack([X.explicit_of(im) for im in ims]).astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dg, dsty = X.Dino(dev)(ims)
    feats = {"explicit": explicit, "dino_global": dg.astype(np.float32), "dino_style": dsty.astype(np.float32)}
    tags = [t for t in a.students.split(",") if t]
    if tags:
        import onnxruntime as ort
        from filmlook_infer.compact import preprocess
        for t in tags:                                                        # 64-d style embeddings of the lightweight students (CPU ONNX)
            ck = P.CHECKPOINTS / t; sess = ort.InferenceSession(str(ck / "model.onnx"), providers=["CPUExecutionProvider"])
            oi = json.loads((ck / "meta.json").read_text())["outputs"].index("style_embedding"); acc = []
            for i in range(0, len(ims), 64):
                acc.append(sess.run(None, {"pixel_values": np.concatenate([preprocess(im) for im in ims[i:i + 64]])})[oi])
            feats[t] = np.concatenate(acc).astype(np.float32)
    out = P.EMBEDDINGS / f"{a.tag}_features_v1"; P.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), **feats)
    keep = ["image_id", "path", "dataset_id", "split", "edit_type", "group_id", "medium", "subjects", "camera", "license", "source_url", "monochrome_tag", "width", "height"]
    out.with_suffix(".json").write_text(json.dumps({"version": f"{a.tag}-features-v1", "n": len(rows), "dataset": a.dataset, "split": a.split, "edit_type": a.edit_type,
                                                    "training_use_allowed": allowed, "rows": [{k: r.get(k) for k in keep} for r in rows], "seconds": round(time.time() - t0, 1)}))
    print(json.dumps({"n": len(rows), "uploaders": len(set(r.get("group_id") for r in rows)), "seconds": round(time.time() - t0, 1), "out": str(out)}))


if __name__ == "__main__":
    main()
