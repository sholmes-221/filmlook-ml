#!/usr/bin/env python
"""Export an ensemble checkpoint -> ONNX fp32 (+ INT8 for DINOv2 variants) and check parity."""
import argparse
import json
import random

from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.export.onnx_export import export_onnx, parity, quantize_int8, write_export_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(P.CHECKPOINTS / "best"))
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--int8", choices=["auto", "yes", "no"], default="auto",
                    help="auto: quantise only graphs with a DINOv2 encoder (tiny MLPs gain nothing)")
    a = ap.parse_args()
    meta = json.loads((P.ROOT / a.ckpt / "meta.json").read_text()) if not a.ckpt.startswith("/") else \
        json.loads(open(f"{a.ckpt}/meta.json").read())
    fp32 = export_onnx(a.ckpt)
    man = [r for r in read_jsonl(P.MANIFESTS / "manifest.jsonl") if r.get("split") == "test"]
    random.Random(0).shuffle(man)
    imgs = [r["path"] for r in man[: a.n_images]]
    info = {"kind": meta["kind"], "fp32_mb": fp32.stat().st_size / 1e6,
            "parity_fp32_random": parity(a.ckpt, fp32), "parity_fp32_images": parity(a.ckpt, fp32, len(imgs), images=imgs),
            "n_images": len(imgs)}
    if a.int8 == "yes" or (a.int8 == "auto" and meta["kind"] != "texture"):
        int8 = quantize_int8(fp32)
        info["int8_mb"] = int8.stat().st_size / 1e6
        info["parity_int8_images"] = parity(a.ckpt, int8, len(imgs), images=imgs)
    write_export_report(a.ckpt, info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
