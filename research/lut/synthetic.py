"""Synthetic preset-labelled dataset: apply every parsed preset to licensed digital photographs already on disk.

    python -m research.lut.synthetic --out data/lut/synth [--train-variants 2] [--no-grain]

Base photographs: `data/manifests/raw_commoncatalog.jsonl` rows with medium digital_camera / smartphone (CC-BY-2.0,
`commoncatalog_ccby`, training_allowed: true in configs/datasets.yaml); near-monochrome frames (mean max-min < 6/255)
are dropped. Filtered uploads (smartphone_filtered / digital_filtered) are kept apart as "unknown edit" negatives
(`--negatives`), never as training scenes. Film scans are excluded from the base pool.
Labels are the preset name (deterministic, provenance `synthetic`), plus `none` for the untouched photo. No model
output is used as a label anywhere.

Splits are scene-disjoint by uploader (group_id hash): train 60 %, val 20 %, test 20 %.
Nuisance (so that the classifier cannot key on the exact base photo): before the preset, exposure U(-0.5, 0.5) EV in
linear light and white-balance gains U(0.94, 1.06) on R and B; after the preset, grain emulation from the preset's
GrainAmount/GrainSize (Gaussian luminance noise, std 0.10*amount/100 at 512 px, blurred by size/100*1.2 px), a random
crop of 80-100 % area and JPEG quality U(75, 95). Val/test get one variant per (scene, preset) with the same nuisance
distribution but a fixed seed; the base photo is resized to a 512 px short side first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

from research.lut.forward_model import Preset, apply_lattice, identity_preset, load_presets, srgb_decode, srgb_encode


def split_of(group_id: str) -> str:
    h = int(hashlib.sha1(group_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 80 else "test")


def base_rows(manifest="data/manifests/raw_commoncatalog.jsonl", media=("digital_camera", "smartphone")) -> list[dict]:
    rows = [json.loads(l) for l in open(manifest)]
    out = []
    for r in rows:
        if r.get("medium") not in media:
            continue
        im = Image.open(r["path"]).convert("RGB")
        im.thumbnail((128, 128))
        a = np.asarray(im, dtype=np.float64)
        if (a.max(-1) - a.min(-1)).mean() < 6:
            continue
        out.append(r)
    return out


def load_base(path, short=512) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = short / min(im.size)
    im = im.resize((max(short, round(im.width * s)), max(short, round(im.height * s))), Image.LANCZOS)
    return np.asarray(im)


def pre_nuisance(img_u8: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    ev = float(rng.uniform(-0.5, 0.5))
    gr, gb = float(rng.uniform(0.94, 1.06)), float(rng.uniform(0.94, 1.06))
    lin = srgb_decode(img_u8 / 255) * (2 ** ev) * np.array([gr, 1.0, gb])
    out = np.clip(srgb_encode(np.clip(lin, 0, 1)) * 255, 0, 255).round().astype(np.uint8)
    return out, {"ev": round(ev, 3), "wb_r": round(gr, 3), "wb_b": round(gb, 3)}


def add_grain(img_u8: np.ndarray, grain: dict, rng: np.random.Generator) -> np.ndarray:
    amount = float(grain.get("GrainAmount", 0) or 0)
    if amount <= 0:
        return img_u8
    size = float(grain.get("GrainSize", 25) or 25)
    h, w = img_u8.shape[:2]
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    sigma = max(size / 100 * 1.2, 0.01)
    if sigma > 0.3:
        noise = gaussian_filter(noise, sigma)
        noise /= max(noise.std(), 1e-6)
    x = img_u8.astype(np.float32) / 255
    lum = x.mean(-1, keepdims=True)
    weight = 1 - 0.7 * lum  # less visible in highlights
    x = x + (0.10 * amount / 100) * noise[..., None] * weight
    return np.clip(x * 255, 0, 255).round().astype(np.uint8)


def post_nuisance(img_u8: np.ndarray, rng: np.random.Generator) -> tuple[bytes, dict]:
    h, w = img_u8.shape[:2]
    area = float(rng.uniform(0.8, 1.0))
    ch, cw = int(h * np.sqrt(area)), int(w * np.sqrt(area))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    q = int(rng.integers(75, 96))
    buf = io.BytesIO()
    Image.fromarray(img_u8[y0:y0 + ch, x0:x0 + cw]).save(buf, "JPEG", quality=q, subsampling=0)
    return buf.getvalue(), {"crop_area": round(area, 3), "jpeg_q": q}


def render(base_u8: np.ndarray, preset: Preset, lut: np.ndarray, rng: np.random.Generator, grain: bool) -> tuple[bytes, dict]:
    x, pre = pre_nuisance(base_u8, rng)
    y = apply_lattice(lut, x)
    if grain:
        y = add_grain(y, preset.grain, rng)
    data, post = post_nuisance(y, rng)
    return data, {**pre, **post}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/lut/synth")
    ap.add_argument("--presets", default="data/lut/presets.json")
    ap.add_argument("--train-variants", type=int, default=2)
    ap.add_argument("--no-grain", action="store_true")
    ap.add_argument("--negatives", action="store_true", help="also render the filtered uploads untouched as unknown-edit negatives")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    presets = load_presets(a.presets)
    presets["none"] = identity_preset()
    luts = {k: p.lattice(33) for k, p in presets.items()}
    rows = base_rows()
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} base scenes; splits:", {s: sum(split_of(r['group_id']) == s for r in rows) for s in ("train", "val", "test")})
    manifest = []
    n_bytes = 0
    for i, r in enumerate(rows):
        sid = r["source_id"]
        split = split_of(r["group_id"])
        base = load_base(r["path"])
        nvar = a.train_variants if split == "train" else 1
        for name, p in presets.items():
            for v in range(nvar):
                seed = int(hashlib.sha1(f"{sid}:{name}:{v}".encode()).hexdigest(), 16) % (2 ** 32)
                rng = np.random.default_rng(seed)
                data, nuis = render(base, p, luts[name], rng, grain=not a.no_grain)
                fn = f"{sid}__{name.replace(chr(39), '')}__{v}.jpg"
                d = out / split
                d.mkdir(exist_ok=True)
                (d / fn).write_bytes(data)
                n_bytes += len(data)
                manifest.append({"path": str((d / fn).resolve()), "scene_id": sid, "group_id": r["group_id"], "preset": name,
                                 "split": split, "variant": v, "nuisance": nuis, "base_dataset": r["dataset_id"],
                                 "base_license": r["license"], "label_source": "synthetic", "labeling_method": "preset_render"})
        if i % 25 == 0:
            print(f"  {i + 1}/{len(rows)} scenes, {n_bytes / 1e6:.0f} MB")
    if a.negatives:
        neg = base_rows(media=("smartphone_filtered", "digital_filtered"))
        d = out / "negatives"
        d.mkdir(exist_ok=True)
        for r in neg:
            base = load_base(r["path"])
            rng = np.random.default_rng(int(hashlib.sha1(r["source_id"].encode()).hexdigest(), 16) % (2 ** 32))
            data, nuis = post_nuisance(base, rng)
            fn = f"{r['source_id']}__filtered__0.jpg"
            (d / fn).write_bytes(data)
            n_bytes += len(data)
            manifest.append({"path": str((d / fn).resolve()), "scene_id": r["source_id"], "group_id": r["group_id"], "preset": "unknown_filtered",
                             "split": "negatives", "variant": 0, "nuisance": nuis, "base_dataset": r["dataset_id"], "base_license": r["license"],
                             "label_source": "synthetic", "labeling_method": "untouched_filtered_upload"})
    with open(out / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(manifest)} images, {n_bytes / 1e6:.0f} MB, manifest {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
