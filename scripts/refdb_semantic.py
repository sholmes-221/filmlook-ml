#!/usr/bin/env python
"""SigLIP image embeddings, zero-shot tag scores and pixel measurements for every embedded reference work.

  python scripts/refdb_semantic.py [--batch 32]

Writes data/refdb/siglip_v1.npz: work_id, X (768-d, L2-normalised SigLIP image embeddings), tag_names, tag_scores
(sigmoid zero-shot scores), measured (pixel facts as a JSON string per row). The SigLIP vectors are a *content*
representation: the evaluation uses them as the semantic signal and as the subject-bias baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.refdb import manifest  # noqa: E402
from filmlook_ml.visualdna.semantic import Tagger, measure  # noqa: E402

OUT = P.DATA / "refdb" / "siglip_v1.npz"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    with manifest.connect() as c:
        ids = [r[0] for r in c.execute("SELECT DISTINCT work_id FROM embeddings WHERE model = 'dinov2s_cls_224' ORDER BY work_id")]
    t = Tagger()
    X, S, M = [], [], []
    for i in range(0, len(ids), a.batch):
        imgs = [Image.open(manifest.image_path(w)).convert("RGB") for w in ids[i:i + a.batch]]
        e = t.image_embeddings(imgs)
        X.append(e); S.append(t.logits(e, t.text)); M += [json.dumps(measure(im)) for im in imgs]
        print(f"{i + len(imgs)}/{len(ids)}", flush=True)
    L = np.concatenate(S).astype(np.float32)
    np.savez_compressed(OUT, work_id=np.array(ids), X=np.concatenate(X).astype(np.float32), tag_names=np.array(t.tags),
                        tag_logits=L, measured=np.array(M))
    # Per-tag logit statistics over the reference corpus: the detection baseline (semantic.TAG_STATS).
    stats = {tag: [float(L[:, j].mean()), float(L[:, j].std())] for j, tag in enumerate(t.tags)}
    (P.DATA / "refdb" / "siglip_tag_stats.json").write_text(json.dumps(stats, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
