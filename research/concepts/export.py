"""Write the deployable artefact with the shipped heads only, plus its attribution file.

  python -m research.concepts.export -> dist/concepts/concept-heads-<date>-<hash>.json, TRAINING_DATA_ATTRIBUTION.jsonl

The JSON follows the website's local_vision heads schema; `hash` is the first 8 hex of sha256 over the weights, bias
and thresholds. Weights are the train-split fit whose threshold was chosen on val and measured on test (what was
tested is what ships; no refit on more data).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np

from research.concepts.common import DATA, DIST, REPORTS, read_jsonl
from research.concepts.predict import EMBEDDING

SPEC = ("facebook/dinov2-small (revision ed25f3a31f01, frozen), via models/photo_gate/dinov2_small_cls_patch.onnx; "
        "input = dinov2_pixels(prepared JPEG): RGB, shorter side resized to 256 (bicubic), centre crop 224, /255, ImageNet "
        "mean (0.485, 0.456, 0.406) std (0.229, 0.224, 0.225), NCHW float32; output = [L2norm(CLS token), L2norm(mean of the "
        "256 patch tokens)] = 768 float32, used as is (the same vector the photo gate scores)")


def main() -> None:
    res = json.loads((REPORTS / "heads_results.json").read_text())
    f = np.load(DATA / "heads_model.npz")
    idx = {h: i for i, h in enumerate(f["heads"])}
    shipped = [h for h, r in res["heads"].items() if r["decision"] == "shipped"]
    W = [f["W"][idx[h]].round(7).tolist() for h in shipped]
    b = [round(float(f["b"][idx[h]]), 7) for h in shipped]
    outputs = []
    for i, h in enumerate(shipped):
        r = res["heads"][h]
        t = r["test"]
        outputs.append({"signal": r["key"], "concept": h, "classes": [i], "threshold": round(r["threshold"], 6),
                        "test": {k: t[k] for k in ("n_pos", "n_neg", "n_generic_neg", "precision", "precision_wilson95", "recall",
                                                   "fpr_generic", "precision_at_generic_prevalence", "ap")}})
    h8 = hashlib.sha256(json.dumps([shipped, W, b, [o["threshold"] for o in outputs]]).encode()).hexdigest()[:8]
    date = dt.date.today().strftime("%Y%m%d")
    version = f"concept-heads-{date}-{h8}"
    art = {"version": version, "kind": "concepts", "embedding": EMBEDDING, "embedding_spec": SPEC, "activation": "sigmoid",
           "classes": shipped, "weights": W, "bias": b, "outputs": outputs,
           "signal_source": "each output's `signal` is the exact key of filmlook/collection/context.py _WORDS",
           "training": {"labels": "Open Images V7 human-verified image-level labels (CC BY 4.0, Google LLC)",
                        "images": "Open Images V7 Flickr photographs listed as CC BY 2.0 (author, source page per image in TRAINING_DATA_ATTRIBUTION.jsonl)",
                        "split": "Flickr-author-disjoint 60/20/20, near-duplicates removed across splits",
                        "classifier": "logistic regression per concept (class_weight balanced, C chosen on val AP)",
                        "threshold_rule": "lowest val score keeping val precision >= 0.95 at every higher threshold (>= 8 fires)",
                        "ship_rule": res["rule"], "experiment": "E60_badge_concept_heads"},
           "unsupported": {h: r.get("reason", "") for h, r in res["heads"].items() if r["decision"] != "shipped"}}
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"{version}.json"
    out.write_text(json.dumps(art, indent=1))
    sel = __import__("pandas").read_parquet(DATA / "selection.parquet")
    train_ids = set(sel[sel.split == "train"].image_id)
    attr = [{k: m[k] for k in ("image_id", "license", "license_url", "author", "author_url", "source_page")}
            for m in read_jsonl(DATA / "manifest.jsonl") if m["image_id"] in train_ids]
    with open(DIST / "TRAINING_DATA_ATTRIBUTION.jsonl", "w") as fh:
        for a in attr:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(out, hashlib.sha256(out.read_bytes()).hexdigest(), f"{len(shipped)} heads; attribution rows {len(attr)}")


if __name__ == "__main__":
    main()
