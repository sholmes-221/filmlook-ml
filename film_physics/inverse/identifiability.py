"""Can a stock be recovered from one image once scanning and editing are unknown? An oracle-scene upper bound.

For each base scene and each stock s, simulate I_s (simulator/forward.py), apply a random nuisance edit (the unknown scan
and edit) and JPEG q85. Then, knowing the scene exactly (the oracle case, far easier than real life), fit the same
nuisance class from every candidate stock's clean render I_t to the query and rank stocks by residual. If the true stock
is not recovered here, it cannot be recovered from an arbitrary JPEG whose scene is unknown.

Nuisance classes, all fitted by least squares in log-linear space:
  wb        per-channel offset (white balance and exposure)            3 parameters
  levels    per-channel offset and slope (scan levels, contrast)       6 parameters
  colour    levels plus a 3x3 cross-channel matrix (saturation, hue)  12 parameters
Run: python film_physics/inverse/identifiability.py <manifest.json> <image dir> <out.json>
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
from forward import Stock, linear_to_srgb, simulate, srgb_to_linear  # noqa: E402

STOCKS = ["kodak_portra_160", "kodak_portra_400", "kodak_gold_200", "kodak_ektar_100", "kodak_ultramax_400",
          "fujifilm_c200", "fujifilm_xtra_400", "fujifilm_pro_400h", "fujifilm_provia_100f", "fujifilm_velvia_100"]
CLASSES = ("wb", "levels", "colour")


def fit(src: np.ndarray, dst: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    """Fit nuisance(src) ≈ dst in log-linear space; return the fitted image (sRGB) and RMSE in sRGB."""
    ls = np.log(np.clip(srgb_to_linear(src.reshape(-1, 3)), 1e-4, 1))
    ld = np.log(np.clip(srgb_to_linear(dst.reshape(-1, 3)), 1e-4, 1))
    ok = (ld > np.log(2e-3)).all(1) & (ld < np.log(0.98)).all(1)
    out = np.empty_like(ls)
    for c in range(3):
        if kind == "wb":  # log dst = log src + offset
            offset = float(np.mean(ld[ok, c] - ls[ok, c]))
            out[:, c] = ls[:, c] + offset
            continue
        if kind == "levels":
            X = np.stack([np.ones(ls.shape[0]), ls[:, c]], axis=1)
        else:
            X = np.concatenate([np.ones((ls.shape[0], 1)), ls], axis=1)
        coef, *_ = np.linalg.lstsq(X[ok], ld[ok, c], rcond=None)
        out[:, c] = X @ coef
    img = linear_to_srgb(np.exp(out)).reshape(src.shape)
    return img, float(np.sqrt(((img - dst) ** 2).mean()))


def random_nuisance(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    l = np.log(np.clip(srgb_to_linear(img.reshape(-1, 3)), 1e-4, 1))
    off = rng.normal(0, 0.25, 3)
    if kind == "wb":
        l = l + off
    else:
        slope = rng.uniform(0.8, 1.25, 3)
        l = l * slope + off
        if kind == "colour":
            m = np.eye(3) + rng.normal(0, 0.08, (3, 3))
            l = l @ m.T
    return linear_to_srgb(np.exp(l)).reshape(img.shape)


def jpeg(img: np.ndarray, q: int = 85) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())), dtype=np.float64) / 255


def main(manifest: Path, imgdir: Path, out: Path, n_scenes: int = 24) -> dict:
    items = [i for i in json.loads(manifest.read_text()) if i["label"] == "digital"]
    rng = np.random.default_rng(20261109)
    stocks = {s: Stock.load(s) for s in STOCKS}
    scenes = []
    for it in items:
        f = it.get("file") or hashlib.sha1(it["title"].encode()).hexdigest()[:16] + ".jpg"
        p = imgdir / f
        if p.exists():
            a = np.asarray(Image.open(p).convert("RGB").resize((128, 96)), dtype=np.float64) / 255
            if a.std() > 0.08:  # skip near-flat frames
                scenes.append(a)
        if len(scenes) >= n_scenes:
            break
    result = {"scenes": len(scenes), "stocks": STOCKS, "classes": {}}
    for kind in CLASSES:
        correct, ranks, margins, confusion = 0, [], [], np.zeros((len(STOCKS), len(STOCKS)), int)
        noise_floor = []
        for a in scenes:
            renders = {s: simulate(a, stocks[s]) for s in STOCKS}
            for i, s in enumerate(STOCKS):
                query = jpeg(random_nuisance(renders[s], kind, rng))
                noise_floor.append(float(np.sqrt(((jpeg(renders[s]) - renders[s]) ** 2).mean())))
                res = [fit(renders[t], query, kind)[1] for t in STOCKS]
                order = np.argsort(res)
                confusion[i, order[0]] += 1
                correct += int(order[0] == i)
                ranks.append(int(np.where(order == i)[0][0]) + 1)
                best_other = min(r for j, r in enumerate(res) if j != i)
                margins.append(best_other - res[i])
        n = len(scenes) * len(STOCKS)
        result["classes"][kind] = {
            "top1": round(correct / n, 3), "top3": round(float(np.mean(np.array(ranks) <= 3)), 3), "chance_top1": round(1 / len(STOCKS), 3),
            "median_margin_rmse": round(float(np.median(margins)), 5), "jpeg_noise_rmse": round(float(np.median(noise_floor)), 5),
            "confusion": confusion.tolist(),
        }
        print(kind, {k: v for k, v in result["classes"][kind].items() if k != "confusion"}, flush=True)
    out.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
