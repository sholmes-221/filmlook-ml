#!/usr/bin/env python
"""Build the benchmark feature cache once (GPU): explicit, DINOv2 global, DINOv2 L4+L8 statistics, student embeddings.

  python scripts/build_bench_features.py    # -> data/embeddings/bench_features_v1.{npz,json}

Images: filmsynth v2 variants (+ a nuisance copy of each: JPEG q55 + 0.5x rescale), SDXL and Z-Image treated
and controls, raw sources, CommonCatalog originals (all splits, with subjects/medium), BlueNeg previews.
Every image is standardised as in synth_experiment_v3 before features. Nothing here looks at labels.
"""
import io
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402

from filmlook_infer.compact import preprocess  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

OUT = P.EMBEDDINGS / "bench_features_v1"


def nuisance(im: Image.Image) -> Image.Image:
    b = io.BytesIO(); im.save(b, "JPEG", quality=55); im2 = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    return im2.resize((max(32, im2.width // 2), max(32, im2.height // 2)), Image.BILINEAR).resize(im2.size, Image.BILINEAR)


def _load(args):
    path, key, real, nuis = args
    arr = V3._load((path, key, True, real))
    im = Image.fromarray(arr)
    if nuis:
        im = nuisance(im)
    return np.asarray(im, dtype=np.uint8)


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    fs = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    zim = read_jsonl(P.DATA / "synthetic" / "zimage_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx} & {a["source_image_id"] for a in zim})
    sp = {s: by_id[s]["split"] for s in srcs}
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    rows = []
    for a in fs:
        if a["source_image_id"] in sp:
            base = {"set": "filmsynth", "source": a["source_image_id"], "split": sp[a["source_image_id"]], "family": a["augmentation_family"],
                    "treated": a["augmentation_family"] != "identity_v2", "params": {k: a["parameters"][k] for k in ("temperature_shift", "saturation_adjustment", "contrast_adjustment", "grain_strength", "shadow_lift", "highlight_compression", "monochrome")}}
            rows.append({"id": a["variant_id"], "path": str(P.ROOT / a["path"]), "real": False, "nuis": False, **base})
            rows.append({"id": a["variant_id"] + "__nuis", "path": str(P.ROOT / a["path"]), "real": False, "nuis": True, **base, "set": "filmsynth_nuis"})
    for name, arr in (("sdxl", sdx), ("zimage", zim)):
        for a in arr:
            if a["source_image_id"] in sp:
                rows.append({"id": a["variant_id"], "path": str(P.ROOT / a["path"]), "real": False, "nuis": False, "set": name, "source": a["source_image_id"],
                             "split": sp[a["source_image_id"]], "family": a["treatment"], "treated": bool(a.get("treated", True))})
    for s in srcs:
        rows.append({"id": s, "path": by_id[s]["path"], "real": True, "nuis": False, "set": "source", "source": s, "split": sp[s], "family": "raw", "treated": False})
    for r in eligible:
        if r["image_id"] not in sp and r["dataset_id"] == "commoncatalog_ccby":
            rows.append({"id": r["image_id"], "path": r["path"], "real": True, "nuis": False, "set": "cc_original", "source": r["image_id"], "split": r["split"],
                         "family": "real", "treated": False, "medium": r.get("medium"), "subjects": r.get("subjects") or [], "group": r["group_id"]})
    for r in bn:
        rows.append({"id": r["image_id"], "path": r["path"], "real": True, "nuis": False, "set": "blueneg", "source": r["image_id"], "split": "test",
                     "family": r["film_stock_verified"], "treated": False, "partition": r["partition"], "roll": r["roll_id"]})
    print(f"{len(rows)} images", file=sys.stderr)
    with Pool(16) as pool:
        arrs = pool.map(_load, [(r["path"], r["id"], r["real"], r["nuis"]) for r in rows], chunksize=8)
    ims = [Image.fromarray(a) for a in arrs]
    explicit = np.stack([X.explicit_of(im) for im in ims]).astype(np.float32)
    dino = X.Dino(dev)
    dg, ds = dino(ims)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(P.CHECKPOINTS / "student_style_fs_v2" / "model.onnx"), providers=["CPUExecutionProvider"])
    meta = json.loads((P.CHECKPOINTS / "student_style_fs_v2" / "meta.json").read_text())
    oi = meta["outputs"].index("style_embedding")
    st64, st_t = [], []
    for i in range(0, len(ims), 64):
        xs = np.concatenate([preprocess(im) for im in ims[i:i + 64]])
        outs = sess.run(None, {"pixel_values": xs}); st64.append(outs[oi]); st_t.append(outs[meta["outputs"].index("treated_logit")])
    st64 = np.concatenate(st64).astype(np.float32); st_t = np.concatenate(st_t).astype(np.float32)
    np.savez(OUT.with_suffix(".npz"), explicit=explicit, dino_global=dg.astype(np.float32), dino_style=ds.astype(np.float32), student64=st64, student_treated=st_t)
    OUT.with_suffix(".json").write_text(json.dumps({"version": "bench-features-v1", "n": len(rows), "rows": rows, "seconds": round(time.time() - t0, 1),
                                                    "student": meta["model_version"], "notes": "standardised 384 px short side, seeded JPEG; nuisance = JPEG q55 + 0.5x rescale"}))
    print(json.dumps({"n": len(rows), "by_set": {s: sum(r["set"] == s for r in rows) for s in sorted({r["set"] for r in rows})}, "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
