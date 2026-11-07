#!/usr/bin/env python
"""Golden fixtures for Python <-> JavaScript parity (features, engine output, learned-model inference).

Images: CC0 PD12M photos + synthetic scenes and their edits only (no CC-BY photo is redistributed).
The learned models in the fixtures are trained on TOY labels purely to exercise the code paths; they are
never shipped and say nothing about accuracy.
"""
import json
import random

import numpy as np
from PIL import Image

from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.style import learned as L
from filmlook_ml.style.engine import analyze_features
from filmlook_ml.style.features import feature_vector, resize_for_style, style_features
from filmlook_ml.style.profiles import PROFILES

OUT = P.ROOT / "js" / "filmlook-style" / "test" / "fixtures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    rng = random.Random(0)
    pool = [r for r in man if r["dataset_id"] in ("pd12m_pilot", "synthetic_fixture")]
    rng.shuffle(pool)
    picks, seen = [], set()
    for r in pool:  # diverse edit types
        if r["edit_type"] in seen and len(picks) < 9:
            continue
        seen.add(r["edit_type"])
        picks.append(r)
        if len(picks) >= 14:
            break
    model_data = json.loads((P.ROOT / "js" / "filmlook-style" / "model-data.js").read_text()
                            .split("module.exports = ", 1)[1].rstrip().rstrip(";"))
    q = model_data["quantiles"]
    cases = []
    for k, r in enumerate(picks):
        a = resize_for_style(np.asarray(Image.open(r["path"]).convert("RGB")))
        rgba = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], 2)
        name = f"case{k:02d}"
        (OUT / f"{name}.rgba").write_bytes(rgba.tobytes())
        f = style_features(a)
        cases.append({"name": name, "w": int(a.shape[1]), "h": int(a.shape[0]), "edit": r["edit_type"],
                      "features": f, "en": analyze_features(f, q, "en"), "zh": analyze_features(f, q, "zh")})
    # learned-model code-path fixtures (toy labels from feature thresholds; NOT a real model)
    F = json.loads((P.EMBEDDINGS / "style_features.json").read_text())
    rows = [F[r["sha256"]] for r in man[:1500]]
    X = np.stack([feature_vector(f) for f in rows])
    ids = list(PROFILES)
    y = [ids[int(f["sat_mean"] * 7) % 3] if f["y_std"] > 0.2 else "none" for f in rows]
    learned = {}
    for kind in L.KINDS:
        sc, m = L.fit(kind, X, y, seed=0)
        ex = L.export(kind, sc, m)
        learned[kind] = {"model": ex, "expected": [L.predict_exported(feature_vector(c["features"]), ex).tolist()
                                                   for c in cases]}
    # box2x parity: an odd-sized random RGBA image -> expected 2x2 box average
    from filmlook_ml.style.features import box2x
    rnd = np.random.default_rng(1).integers(0, 256, (37, 53, 4), dtype=np.uint8)
    (OUT / "box_in.rgba").write_bytes(rnd.tobytes())
    box = {"w": 53, "h": 37, "expected": box2x(rnd[..., :3]).tolist()}
    (OUT / "golden.json").write_text(json.dumps({"cases": cases, "learned": learned, "box2x": box}))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
