"""NumPy-only reference for the concept-heads artefact (no torch, no sklearn): what the website must reproduce.

    art = load("dist/concepts/concept-heads-<date>-<hash>.json")
    predict(art, embedding)  -> {"umbrella": 0.93, ...}  (score of every shipped head)
    signals(art, embedding)  -> {"umbrella", ...}        (the _WORDS keys whose head reached its threshold)

`embedding` is the photo gate's vector, `dinov2s-cls-patch-768`: [L2norm(CLS), L2norm(mean patch)] of DINOv2-small on
`dinov2_pixels(prepared JPEG)`, 768 float32, used as is (not renormalised as a whole). The artefact follows the website's
`filmlook/local_vision.py` heads schema (kind "concepts", activation "sigmoid", one class per output).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMBEDDING = "dinov2s-cls-patch-768"


def load(path) -> dict:
    art = json.loads(Path(path).read_text())
    if art.get("embedding") != EMBEDDING or art.get("activation") != "sigmoid":
        raise ValueError("not a dinov2s-cls-patch-768 sigmoid heads artefact")
    art["_W"] = np.asarray(art["weights"], dtype=np.float32)
    art["_b"] = np.asarray(art["bias"], dtype=np.float32)
    if art["_W"].shape != (len(art["classes"]), 768):
        raise ValueError("weights do not match classes x 768")
    return art


def predict(art: dict, embedding) -> dict[str, float]:
    v = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if v.shape != (768,):
        raise ValueError("embedding must have 768 values")
    p = 1.0 / (1.0 + np.exp(-(art["_W"] @ v + art["_b"])))
    return {c: float(p[i]) for i, c in enumerate(art["classes"])}


def signals(art: dict, embedding) -> set[str]:
    p = predict(art, embedding)
    return {o["signal"] for o in art["outputs"] if p[art["classes"][o["classes"][0]]] >= o["threshold"]}
