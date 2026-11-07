#!/usr/bin/env python
"""Spatial composition probe: do spatial representations recover framing where pooled features could not?

  python scripts/spatial_probe.py     # -> reports/spatial_probe.json

Controlled crops of real photos (never used as synthetic sources): zoom 1-2x, signed horizontal and vertical
offsets in [-1, 1] (the crop window slides toward one side), 4 crops per source, sources split train/test by the
existing uploader-grouped split. Ridge regression (alpha chosen by RidgeCV) predicts the crop parameters from each
representation on held-out sources; R^2 <= 0 means nothing recovered.
Representations:
  dino_global      CLS ++ mean patch (768)                       [pooled: failed before]
  dino_style       layer 4/8 patch mean ++ std (1536)            [pooled]
  dino_grid2       final-layer patch tokens average-pooled to a 2x2 grid (4 x 384)
  dino_grid4       4x4 grid (16 x 384)
  dino_pyramid     1x1 ++ 2x2 ++ 4x4 (21 x 384)
  mnv3_grid2       ImageNet MobileNetV3-Small feature map (7x7x576) pooled to 2x2 (2304) ++ global (576)
Mirror sensitivity: cosine distance between an image and its horizontal mirror, relative to the median distance
to other images (0 = flip-invariant, 1 = as different as another photo). A composition-aware representation must
not be flip-invariant. These are synthetic-perturbation proxies, not human composition judgements.
"""
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402

from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_infer.pixels import dinov2_pixels  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

N_SOURCES, CROPS = 240, 4


class SpatialDino:
    def __init__(self, dev):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, ims):
        out = {k: [] for k in ("dino_global", "dino_style", "dino_grid2", "dino_grid4", "dino_pyramid")}
        for i in range(0, len(ims), 64):
            px = torch.from_numpy(np.stack([dinov2_pixels(im, 224) for im in ims[i:i + 64]])).to(self.dev)
            o = self.m(pixel_values=px, output_hidden_states=True)
            h = o.last_hidden_state
            out["dino_global"].append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
            st = [torch.cat([hs[:, 1:].mean(1), hs[:, 1:].std(1)], 1) for hs in (o.hidden_states[4], o.hidden_states[8])]
            out["dino_style"].append(torch.cat(st, 1).float().cpu().numpy())
            g = h[:, 1:].reshape(h.shape[0], 16, 16, -1).permute(0, 3, 1, 2)          # B, C, 16, 16
            g2 = torch.nn.functional.adaptive_avg_pool2d(g, 2).flatten(1)
            g4 = torch.nn.functional.adaptive_avg_pool2d(g, 4).flatten(1)
            g1 = g.mean((2, 3))
            out["dino_grid2"].append(g2.float().cpu().numpy()); out["dino_grid4"].append(g4.float().cpu().numpy())
            out["dino_pyramid"].append(torch.cat([g1, g2, g4], 1).float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


class SpatialMnv3:
    def __init__(self, dev):
        import torchvision
        self.f = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").features.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, ims):
        from filmlook_infer.compact import preprocess
        out = []
        for i in range(0, len(ims), 64):
            x = torch.from_numpy(np.concatenate([preprocess(im) for im in ims[i:i + 64]])).to(self.dev)
            fm = self.f(x)
            out.append(torch.cat([torch.nn.functional.adaptive_avg_pool2d(fm, 2).flatten(1), fm.mean((2, 3))], 1).float().cpu().numpy())
        return np.concatenate(out)


def r2(p, t):
    return round(float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()), 4)


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    syn = read_jsonl(P.DATA / "synthetic" / "film_synth_v1" / "annotations.jsonl")
    used = {a["source_image_id"] for a in syn}
    srcs = [r for r in eligible if r["image_id"] not in used and r["split"] in ("train", "test")][:N_SOURCES]
    rng = np.random.default_rng(0)
    ims, tz, tx, ty, split, gid, base = [], [], [], [], [], [], []
    for r in srcs:
        im = X.to_short(load_image(r["path"], max_side=1280), 512)
        base.append(X.to_short(im))
        for _ in range(CROPS):
            z, ox, oy = float(rng.uniform(1.0, 2.0)), float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
            w, h = im.width / z, im.height / z
            cx, cy = im.width / 2 + ox * (im.width - w) / 2, im.height / 2 + oy * (im.height - h) / 2
            box = tuple(int(round(v)) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            ims.append(X.to_short(im.crop(box))); tz.append(z); tx.append(ox); ty.append(oy)
            split.append(r["split"]); gid.append(r["image_id"])
    split = np.array(split); T = {"zoom": np.array(tz), "x_offset": np.array(tx), "y_offset": np.array(ty),
                                  "abs_x_offset": np.abs(tx), "abs_y_offset": np.abs(ty)}
    sd, sm = SpatialDino(dev), SpatialMnv3(dev)
    R = sd(ims); R["mnv3_grid2"] = sm(ims)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    tr, te = split == "train", split == "test"
    res = {}
    for name, F in R.items():
        sc = StandardScaler().fit(F[tr])
        res[name] = {"dim": int(F.shape[1])}
        for tname, t in T.items():
            m = RidgeCV(alphas=[1, 10, 100, 1000, 10000]).fit(sc.transform(F[tr]), t[tr])
            res[name][tname + "_test_r2"] = r2(m.predict(sc.transform(F[te])), t[te])
    # mirror sensitivity on base images (test sources)
    bidx = [i for i, r in enumerate(srcs) if r["split"] == "test"]
    bims = [base[i] for i in bidx]; mims = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in bims]
    Rb, Rm = sd(bims), sd(mims); Rb["mnv3_grid2"], Rm["mnv3_grid2"] = sm(bims), sm(mims)
    mirror = {}
    for name in R:
        A = Rb[name] / np.linalg.norm(Rb[name], axis=1, keepdims=True); B = Rm[name] / np.linalg.norm(Rm[name], axis=1, keepdims=True)
        d_flip = 1 - (A * B).sum(1)
        S = A @ A.T; np.fill_diagonal(S, np.nan)
        d_other = 1 - np.nanmedian(S, axis=1)
        mirror[name] = round(float(np.median(d_flip / (d_other + 1e-8))), 4)
    rep = {"n_sources": len(srcs), "crops_per_source": CROPS, "test_sources": int(sum(r["split"] == "test" for r in srcs)),
           "crop_regression": res, "mirror_sensitivity_ratio": mirror,
           "reading": "R^2 > 0 on held-out sources = framing recoverable; mirror ratio near 0 = flip-invariant (blind to left/right placement)",
           "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "spatial_probe.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
