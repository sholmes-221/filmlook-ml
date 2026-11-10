"""Reproducible synthetic set from the LUT library: validated colour film-look LUTs applied to tier-1 licensed photos.

    python -m research.lut.synthetic_library --out data/lut/synthetic_library --luts-per-image 8

Sources: research/lut_library/source_photos.json tier 1 (untouched digital originals; monochrome frames dropped),
downscaled so the long side is <= 1024 px. Splits are source-disjoint by `group_id` (uploader / institution) with the
same hash as research/lut/synthetic.py (sha1 % 100: <60 train, <80 val, else test), so scenes never cross splits and
the earlier 13-preset scenes keep their split. LUTs: every ingested colour film-look LUT of research/lut/db_library
(re-read from the library files at 33^3), B&W LUTs excluded. Per image, `--luts-per-image` LUTs are drawn stratified by
claimed stock (least-used stocks first, random LUT within the stock, seeded by the image id) plus the untouched
`none` render. Nuisance: exposure U(-0.5, 0.5) EV and white balance U(0.94, 1.06) before the LUT; 80-100 % crop and
JPEG q75-95 after. Labels per row: lut_id, claimed stock, film type, variant tokens, push/pull, LUT licence, photo
licence, output path, split, category. No grain simulation (the library LUTs carry no grain parameters).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from research.lut.library_db import COLOUR_TYPES, ingest, load_manifest
from research.lut.synthetic import post_nuisance, pre_nuisance, split_of


def tier1_photos(path="research/lut_library/source_photos.json") -> list[dict]:
    sp = json.loads(Path(path).read_text())
    rows = [p for p in sp["images"] if p.get("tier") == 1 and not p.get("monochrome_tag") and p.get("category_primary") != "monochrome"]
    rows = [p for p in rows if Path(p["path"]).exists()]
    for p in rows:
        p["split"] = split_of(p["group_id"])
    return rows


def load_photo(path, long_side=1024) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.LANCZOS)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/lut/synthetic_library")
    ap.add_argument("--luts-per-image", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    styles, luts, _ = ingest(load_manifest()["luts"])
    keep = [i for i, s in enumerate(styles) if s["film_type"] in COLOUR_TYPES]
    styles, luts = [styles[i] for i in keep], [luts[i] for i in keep]
    stocks = sorted({s["film_stock"] for s in styles})
    by_stock = {st: [i for i, s in enumerate(styles) if s["film_stock"] == st] for st in stocks}
    photos = tier1_photos()
    if a.limit:
        photos = photos[:a.limit]
    print(f"{len(styles)} colour LUTs over {len(stocks)} stocks; {len(photos)} tier-1 photos; splits",
          {s: sum(p['split'] == s for p in photos) for s in ('train', 'val', 'test')})
    usage = {st: 0 for st in stocks}
    manifest, n_bytes = [], 0
    for n, p in enumerate(photos):
        base = load_photo(p["path"])
        seed = int(hashlib.sha1(p["id"].encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        # least-used stocks first (ties broken randomly), then a random LUT within each chosen stock
        order = sorted(stocks, key=lambda st: (usage[st], rng.random()))
        chosen = [(st, int(rng.choice(by_stock[st]))) for st in order[:a.luts_per_image]]
        for st, i in chosen:
            usage[st] += 1
        renders = [(styles[i]["id"], st, i) for st, i in chosen] + [("none", "none", None)]
        d = out / p["split"]
        d.mkdir(exist_ok=True)
        for lut_id, st, i in renders:
            r2 = np.random.default_rng(int(hashlib.sha1(f"{p['id']}:{lut_id}".encode()).hexdigest(), 16) % (2 ** 32))
            x, pre = pre_nuisance(base, r2)
            y = luts[i].apply_u8(x) if i is not None else x
            data, post = post_nuisance(y, r2)
            fn = f"{p['id']}__{lut_id[:60]}.jpg"
            (d / fn).write_bytes(data)
            n_bytes += len(data)
            s = styles[i] if i is not None else None
            manifest.append({"path": str((d / fn).resolve()), "scene_id": p["id"], "group_id": p["group_id"], "category": p.get("category_primary"),
                             "photo_license": p["license"], "photo_source": p["source"], "split": p["split"], "lut_id": lut_id,
                             "library_id": s["library_id"] if s else None, "stock": st, "film_type": s["film_type"] if s else "none",
                             "variant_tokens": s["variant_tokens"] if s else [], "push_pull_stops": s["push_pull_stops"] if s else 0,
                             "lut_license": s["license"] if s else None, "lut_source": s["source"] if s else None, "nuisance": {**pre, **post},
                             "label_source": "synthetic", "labeling_method": "lut_render"})
        if n % 50 == 0:
            print(f"  {n + 1}/{len(photos)} photos, {n_bytes / 1e6:.0f} MB")
    with open(out / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    summary = {"n_images": len(manifest), "n_photos": len(photos), "n_luts": len(styles), "n_stocks": len(stocks), "bytes": n_bytes,
               "per_split": {s: sum(m["split"] == s for m in manifest) for s in ("train", "val", "test")},
               "scenes_per_split": {s: len({m["scene_id"] for m in manifest if m["split"] == s}) for s in ("train", "val", "test")},
               "lut_usage": usage, "min_renders_per_stock": min(usage.values()), "max_renders_per_stock": max(usage.values())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "lut_usage"}, indent=1))


if __name__ == "__main__":
    main()
