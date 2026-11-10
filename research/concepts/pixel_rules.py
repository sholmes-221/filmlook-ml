"""Deterministic pixel rules for the colour / time-of-day concepts, evaluated exactly like the heads.

  python -m research.concepts.pixel_rules     -> reports/concepts/pixel_rules.json

Each rule is one statistic of the thumbnail (resized to 128 px on the long side, HSV in [0, 1]) and one threshold
chosen on validation with the same precision-first rule; test read once:
  night        fraction of pixels with value < 0.15
  golden_hour  fraction with hue in [15, 50] degrees, saturation > 0.45, value > 0.35 (warm, saturated light)
  blue_hour    fraction with hue in [200, 250] degrees, saturation > 0.30, value in [0.10, 0.60] (deep blue, dim)
  pink         fraction with hue in [300, 350] degrees, or hue in [345, 360] with saturation in [0.15, 0.60], value > 0.55
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from research.concepts.common import DATA, REPORTS


def stats(iid: str) -> np.ndarray:
    im = Image.open(DATA / "images" / iid[:2] / f"{iid}.jpg").convert("RGB")
    im.thumbnail((128, 128))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    return np.array([
        (v < 0.15).mean(),
        ((h >= 15) & (h <= 50) & (s > 0.45) & (v > 0.35)).mean(),
        ((h >= 200) & (h <= 250) & (s > 0.30) & (v > 0.10) & (v < 0.60)).mean(),
        (((h >= 300) & (h <= 350)) | ((h >= 345) & (s > 0.15) & (s < 0.6) & (v > 0.55))).mean(),
    ])


RULES = ["night", "golden_hour", "blue_hour", "pink"]


def main() -> None:
    from research.concepts.train import CONCEPTS, SHIP_LB, SHIP_N, SHIP_P, SHIP_R, dedup, load, metrics, threshold
    rows, X = load()
    rows = rows[dedup(rows, X)].reset_index(drop=True)
    rows = rows[rows.split.isin(["val", "test"])].reset_index(drop=True)
    with ThreadPoolExecutor(16) as ex:
        S = np.stack(list(ex.map(stats, rows.image_id)))
    split, generic = rows.split.to_numpy(), rows.generic.to_numpy()
    res = {}
    for j, head in enumerate(RULES):
        y = rows[head].to_numpy().astype(int)
        va, te = (split == "val") & (y >= 0), (split == "test") & (y >= 0)
        t = threshold(S[va, j], y[va])
        r = {"statistic": __doc__.split("\n")[6 + j].strip()}
        if t is None:
            r.update(decision="unsupported", reason="no val threshold reaches precision 0.95 with >= 8 fires")
        else:
            tm = metrics(S[te, j], y[te], generic[te], t)
            ok = tm["n_pos"] >= SHIP_N and tm["fires"] > 0 and tm["precision"] >= SHIP_P and tm["precision_wilson95"][0] >= SHIP_LB and tm["recall"] >= SHIP_R
            r.update(threshold=t, test=tm, decision="would_ship" if ok else "unsupported")
        from sklearn.metrics import average_precision_score
        r["test_ap"] = float(average_precision_score(y[te] == 1, S[te, j]))
        res[head] = r
        print(head, r.get("decision"), f"AP={r['test_ap']:.3f}", {k: r.get("test", {}).get(k) for k in ("precision", "recall", "n_pos")})
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "pixel_rules.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
