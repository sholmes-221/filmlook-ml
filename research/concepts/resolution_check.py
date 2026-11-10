"""Does the 320 px training thumbnail stand in for the website's prepared JPEG? (deployed ONNX encoder, CPU)

  python -m research.concepts.resolution_check [--n 120]  -> reports/concepts/resolution_check.json

For N test images (fixed by hash), the CVDF copy (about 1024 px) is fetched again into memory only, passed through the
website's preparation (long edge <= 2048, JPEG q90) and embedded with models/photo_gate/dinov2_small_cls_patch.onnx; the
local thumbnail is embedded the same way. Reports the cosine between the two vectors and, for the artefact's heads, how
often the fire / no-fire decision agrees, plus single-thread CPU latency of the heads themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

from research.concepts import predict as P
from research.concepts.common import CVDF, DATA, DIST, REPORTS, USER_AGENT
from research.concepts.features import WEB_ONNX
from research.concepts.features import pixels as _pixels  # noqa: F401  (same preprocessing function)


def onnx_session():
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(WEB_ONNX, so, providers=["CPUExecutionProvider"])


def embed(sess, im: Image.Image) -> np.ndarray:
    import sys
    from research.concepts.common import ROOT
    sys.path.insert(0, str(ROOT / "src"))
    from filmlook_infer.pixels import dinov2_pixels
    return sess.run(None, {"pixel_values": dinov2_pixels(im.convert("RGB"))[None]})[0][0]


def prepared(raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw)); im.load(); im = im.convert("RGB")
    if max(im.size) > 2048:
        im.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    out = Image.open(io.BytesIO(buf.getvalue())); out.load()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); a = ap.parse_args()
    art = P.load(sorted(DIST.glob("concept-heads-*.json"))[-1])
    sel = pd.read_parquet(DATA / "selection.parquet")
    man = pd.read_json(DATA / "manifest.jsonl", lines=True).merge(sel[["image_id", "split", "oi_split"]], on="image_id")
    te = man[man.split == "test"].copy()
    te["h"] = te.image_id.map(lambda i: hashlib.sha256(("res" + i).encode()).hexdigest())
    te = te.sort_values("h").head(a.n)
    sess = onnx_session()
    s = requests.Session(); s.headers["User-Agent"] = USER_AGENT
    cos, agree, total, t_embed = [], 0, 0, []
    for r in te.itertuples():
        resp = s.get(CVDF.format(split=r.oi_split, id=r.image_id), timeout=60)
        if resp.status_code != 200:
            continue
        t = time.perf_counter(); full = embed(sess, prepared(resp.content)); t_embed.append(time.perf_counter() - t)
        thumb = embed(sess, Image.open(DATA / r.path))
        cos.append(float(full @ thumb / np.linalg.norm(full) / np.linalg.norm(thumb)))
        pf, pt = P.predict(art, full), P.predict(art, thumb)
        for o in art["outputs"]:
            c = art["classes"][o["classes"][0]]
            agree += (pf[c] >= o["threshold"]) == (pt[c] >= o["threshold"]); total += 1
    v = np.zeros(768, np.float32); v[:] = 0.036
    ts = []
    for _ in range(2000):
        t = time.perf_counter(); P.signals(art, v); ts.append(time.perf_counter() - t)
    t0 = time.perf_counter(); P.load(sorted(DIST.glob("concept-heads-*.json"))[-1]); load_ms = (time.perf_counter() - t0) * 1000
    res = {"n_images": len(cos), "cosine_min": min(cos), "cosine_mean": float(np.mean(cos)), "cosine_p05": float(np.percentile(cos, 5)),
           "decision_agreement": agree / total, "decisions_compared": total,
           "encoder_ms_1thread_p50_workstation": float(np.median(t_embed) * 1000),
           "heads_ms_p50": float(np.median(ts) * 1000), "heads_ms_p99": float(np.percentile(ts, 99) * 1000), "artefact_load_ms": load_ms}
    (REPORTS / "resolution_check.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
