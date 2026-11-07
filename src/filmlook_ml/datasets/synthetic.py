"""Procedural 'photographs' used as license-free base images for tests and pilots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def make_scene(seed: int, size: tuple[int, int] = (768, 512)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = h * rng.uniform(0.35, 0.65)
    top, bot = rng.uniform(0.3, 0.9, 3), rng.uniform(0.2, 0.7, 3)
    sky = np.stack([top[i] + (bot[i] - top[i]) * (yy / horizon) for i in range(3)], -1)
    # bright sun / window -> real highlight gradients
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.05, 0.4) * h
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sky += (np.exp(-r2 / (2 * (rng.uniform(20, 60)) ** 2)) * rng.uniform(0.5, 1.0))[..., None]
    tex = ndimage.gaussian_filter(rng.standard_normal((h, w)), rng.uniform(1.5, 4))
    ground_col = rng.uniform(0.1, 0.6, 3)
    ground = ground_col + 0.15 * tex[..., None] + 0.1 * ((yy - horizon) / h)[..., None]
    a = np.where((yy > horizon)[..., None], ground, sky)
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(3, 8)):
        x0, y0 = rng.uniform(0, w), rng.uniform(horizon - 100, h)
        x1, y1 = x0 + rng.uniform(30, 200), y0 + rng.uniform(30, 200)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=col)
    a = np.asarray(img, dtype=np.float32) / 255
    a = ndimage.gaussian_filter(a, (0.8, 0.8, 0))                 # lens-like softness
    a *= (1 - 0.35 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * w / 4)))[..., None]  # vignette
    a += rng.normal(0, 0.004, a.shape)                            # faint sensor noise
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def write_synthetic_bases(out_dir: Path, n: int, seed: int = 0) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in range(n):
        p = out_dir / f"synthetic_{seed}_{i:04d}.png"
        if not p.exists():
            make_scene(seed * 100000 + i).save(p)
        recs.append({"path": str(p), "dataset_id": "synthetic_fixture", "license": "synthetic",
                     "source_url": "generated", "source_id": p.stem, "attribution": ""})
    return recs
