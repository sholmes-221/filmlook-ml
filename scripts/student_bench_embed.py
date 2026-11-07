#!/usr/bin/env python
"""Embed the benchmark v1 image list with a student ONNX checkpoint (CPU) so it can be a benchmark candidate.

  python scripts/student_bench_embed.py --tag style_fs_v2_real   # -> data/embeddings/student_<tag>_bench.npz

Reads the cached image list (data/embeddings/bench_features_v1.json), re-standardises each image exactly as the
cache did (same seeded JPEG, same nuisance copy), runs checkpoints/student_<tag>/model.onnx and stores
style_embedding (64-d) and the 576-d pooled features are not exported, so only the 64-d head is used.
Use with: scripts/style_benchmark.py --extra research.candidates_student:build  (env STUDENT_TAGS=tag1,tag2)
"""
import argparse
import json
import sys
from multiprocessing import Pool

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import build_bench_features as BB  # noqa: E402

from filmlook_infer.compact import preprocess  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    rows = json.loads((P.EMBEDDINGS / "bench_features_v1.json").read_text())["rows"]
    with Pool(16) as pool:
        arrs = pool.map(BB._load, [(r["path"], r["id"], r["real"], r["nuis"]) for r in rows], chunksize=8)
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(P.CHECKPOINTS / f"student_{a.tag}" / "model.onnx"), so, providers=["CPUExecutionProvider"])
    meta = json.loads((P.CHECKPOINTS / f"student_{a.tag}" / "meta.json").read_text()); oi = meta["outputs"].index("style_embedding")
    out = []
    for i in range(0, len(arrs), 64):
        out.append(sess.run(None, {"pixel_values": np.concatenate([preprocess(Image.fromarray(x)) for x in arrs[i:i + 64]])})[oi])
    Z = np.concatenate(out).astype(np.float32)
    np.savez(P.EMBEDDINGS / f"student_{a.tag}_bench.npz", student64=Z, ids=np.array([r["id"] for r in rows]))
    print(json.dumps({"tag": a.tag, "n": len(Z), "dim": int(Z.shape[1])}))


if __name__ == "__main__":
    main()
