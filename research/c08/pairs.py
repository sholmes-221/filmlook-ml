"""C08c/d: fully crossed, controlled LUT renders of the same licensed originals, stored as features only.

    python -m research.c08.pairs build            # -> data/c08/pairs/{meta.jsonl, dino.npy, colour.npy, thumbs.npy, luts.json, sources.json}

Design (owner briefs of 2026-11-11):
- Sources: tier-1 licensed colour originals (research/lut_library/source_photos.json). Split by uploader/institution
  BEFORE rendering (60/20/20 by a hash of the group; platforms iNaturalist and Wikimedia Commons per photograph), then a 64-bit dHash guard moves any val/test source
  within 10 bits of a train source into train, so near-duplicates never straddle the boundary.
- LUTs: the 13 owner presets (the live engine's classes), every GlassMeter colour LUT, up to two base RawTherapee
  variants for each stock that another source also implements (the unseen-implementation test), the CC0 camera looks,
  and `none`. Every LUT is applied to every source.
- Per (source, LUT): one clean render (full strength, no nuisance) and variation renders (2 for train sources, 1 for
  val/test): strength 25/50/75/100 %, exposure -1..+1 EV and white balance before the LUT (scene side), then mild
  contrast/saturation, optional grain, resize and JPEG after it. Every parameter is in meta.jsonl.
- Features per render: DINOv2-small (the production weights) CLS and mean-patch tokens at blocks 4, 8 and 12 with the
  production preprocessing; the 312 colour values of research.lut.features; a 64x64 thumbnail for a small learned
  colour model. Nothing rendered is written to disk.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from research.c08.common import OUT, all_luts
from research.lut.features import colour_features_from_array
from research.lut.forward_model import srgb_decode, srgb_encode
from research.lut.synthetic_library import load_photo, tier1_photos

PAIRS = OUT / "pairs"
LAYERS = (4, 8, 12)
PLATFORM_GROUPS = {"pd12m:inaturalist", "pd12m:Wikimedia Commons"}
_STATE: dict = {}


def dhash(img_u8: np.ndarray) -> int:
    g = np.asarray(Image.fromarray(img_u8).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def select_luts() -> list[dict]:
    items = all_luts()
    presets = [x for x in items if x["source"] == "owner_presets"]
    colour = [x for x in items if x["source"] != "owner_presets" and x.get("role") == "film_look" and x.get("film_type") not in (None, "bw_negative", "bw_chromogenic", "bw")
              and not str(x.get("film_type", "")).startswith("bw")]
    glass = [x for x in colour if x["source"] == "glassmeter_film_looks"]
    cam = [x for x in colour if x["source"] == "sguyader_filmsim"]
    multi = {x["film_stock"] for x in glass}
    rt = defaultdict(list)
    for x in colour:
        if x["source"] == "rawtherapee_film_simulation" and x["film_stock"] in multi and not x.get("push_pull"):
            rt[x["film_stock"]].append(x)
    chosen_rt = []
    for stock, xs in sorted(rt.items()):
        xs.sort(key=lambda x: (len(x.get("variant_tokens") or []), x["id"]))
        chosen_rt += xs[:2]
    out = presets + glass + chosen_rt + cam
    ident = dict(presets[0]); ident.update({"id": "none", "name": "none", "source": "identity", "film_stock": None, "lut": None, "license": "n/a"})
    return out + [ident]


def _vary(rng: np.random.Generator) -> dict:
    return {"strength": float(rng.choice([0.25, 0.5, 0.75, 1.0], p=[0.15, 0.2, 0.25, 0.4])),
            "ev": float(rng.choice([-1, -0.5, 0, 0.5, 1], p=[0.1, 0.2, 0.4, 0.2, 0.1])),
            "warmth": float(rng.uniform(-0.06, 0.06)), "tint": float(rng.uniform(-0.03, 0.03)),
            "contrast": float(rng.choice([1.0, rng.uniform(0.9, 1.1)])), "saturation": float(rng.choice([1.0, rng.uniform(0.9, 1.1)])),
            "grain": float(rng.choice([0.0, rng.uniform(0.008, 0.025)], p=[0.7, 0.3])), "scale": float(rng.uniform(0.5, 1.0)), "jpeg_q": int(rng.integers(70, 96))}


def render(src: np.ndarray, lut, v: dict | None, rng: np.random.Generator) -> np.ndarray:
    x = src
    if v:
        lin = srgb_decode(x / 255.0) * (2 ** v["ev"]) * np.array([1 + v["warmth"], 1 + v["tint"], 1 - v["warmth"]])
        x = np.clip(srgb_encode(np.clip(lin, 0, 1)) * 255, 0, 255).round().astype(np.uint8)
    if lut is not None:
        y = lut.apply_u8(x)
        if v and v["strength"] < 1:
            y = np.clip((1 - v["strength"]) * x.astype(np.float32) + v["strength"] * y.astype(np.float32), 0, 255).round().astype(np.uint8)
        x = y
    if v:
        f = x.astype(np.float32) / 255
        if v["contrast"] != 1.0:
            f = 0.5 + (f - 0.5) * v["contrast"]
        if v["saturation"] != 1.0:
            g = f.mean(-1, keepdims=True); f = g + (f - g) * v["saturation"]
        if v["grain"] > 0:
            f = f + rng.normal(0, v["grain"], f.shape[:2])[..., None]
        x = np.clip(f * 255, 0, 255).round().astype(np.uint8)
        im = Image.fromarray(x)
        im = im.resize((max(64, round(im.width * v["scale"])), max(64, round(im.height * v["scale"]))), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=v["jpeg_q"]); x = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return x


def dino_crop(x: np.ndarray, size: int = 224) -> np.ndarray:
    """filmlook photo_gate.dinov2_pixels geometry (short side 256 bicubic, centre crop 224), kept as uint8."""
    im = Image.fromarray(x)
    s = round(size * 256 / 224) / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.BICUBIC)
    l, t = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def _init(luts):
    _STATE["luts"] = luts


def work(job: dict):
    luts = _STATE["luts"]
    src = load_photo(job["path"], 512)
    out_meta, crops, colours, thumbs = [], [], [], []
    for li, L in enumerate(luts):
        nvar = 2 if job["split"] == "train" else 1
        for k in range(1 + nvar):
            seed = int(hashlib.sha1(f"{job['source_id']}:{L['id']}:{k}".encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            v = None if k == 0 else _vary(rng)
            x = render(src, L["lut"], v, rng)
            crops.append(dino_crop(x))
            colours.append(colour_features_from_array(x).astype(np.float32))
            thumbs.append(np.asarray(Image.fromarray(x).resize((64, 64), Image.BOX)))
            out_meta.append({"source_id": job["source_id"], "split": job["split"], "category": job["category"], "group_id": job["group_id"],
                             "lut": L["id"], "lut_source": L["source"], "stock": L.get("film_stock"), "preset": L["name"] if L["source"] == "owner_presets" else None,
                             "variant": k, "clean": v is None, "params": v})
    return out_meta, np.stack(crops), np.stack(colours), np.stack(thumbs)


class DinoLayers:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()  # fp32: fp16 overflows DINOv2 activations (cosine 0.95 vs production)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, crops_u8: np.ndarray, batch: int = 256) -> np.ndarray:
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(crops_u8), batch):
                x = torch.from_numpy(crops_u8[i:i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255
                x = (x - self.mean) / self.std
                o = self.model(pixel_values=x, output_hidden_states=True)
                feats = []
                for layer in LAYERS:
                    h = o.last_hidden_state if layer == 12 else o.hidden_states[layer]
                    feats += [h[:, 0], h[:, 1:].mean(1)]
                outs.append(torch.cat(feats, 1).float().cpu().numpy().astype(np.float16))
        return np.concatenate(outs)


def sources() -> list[dict]:
    rows = tier1_photos()
    hashes = {}
    for r in rows:
        hashes[r["source_id"] if "source_id" in r else r["id"]] = dhash(load_photo(r["path"], 256))
    jobs = []
    for r in rows:
        sid = r.get("source_id") or r["id"]
        # Platforms are not uploaders: grouping all of iNaturalist (220 photos, many observers) as one group put every
        # wildlife photo in one split. They are grouped per photograph; the dHash guard below catches near-duplicates.
        key = sid if r["group_id"] in PLATFORM_GROUPS else r["group_id"]
        h = int(hashlib.sha1(f"c08:{key}".encode()).hexdigest(), 16) % 100
        jobs.append({"source_id": sid, "path": r["path"], "split": "train" if h < 60 else "val" if h < 80 else "test", "group_id": r["group_id"],
                     "split_key": key, "category": r.get("category_primary")})
    train_h = [(j["source_id"], hashes[j["source_id"]]) for j in jobs if j["split"] == "train"]
    moved = 0
    for j in jobs:
        if j["split"] != "train":
            h = hashes[j["source_id"]]
            if any(bin(h ^ th).count("1") <= 10 for _, th in train_h):
                j["split"], j["moved_from"] = "train", j["split"]; moved += 1
    print("sources", len(jobs), Counter(j["split"] for j in jobs), "moved to train by dHash", moved, flush=True)
    return jobs


def build(a) -> None:
    PAIRS.mkdir(parents=True, exist_ok=True)
    luts = select_luts()
    (PAIRS / "luts.json").write_text(json.dumps([{k: v for k, v in L.items() if k not in ("lut", "path")} for L in luts], indent=1))
    print("LUTs", len(luts), Counter(L["source"] for L in luts), flush=True)
    jobs = sources()
    if a.limit:
        jobs = jobs[:a.limit]
    (PAIRS / "sources.json").write_text(json.dumps(jobs, indent=1))
    dino = DinoLayers()
    metas, D, C, T = [], [], [], []
    with Pool(a.workers, initializer=_init, initargs=(luts,)) as pool:
        for n, (m, crops, col, th) in enumerate(pool.imap(work, jobs, chunksize=1)):
            metas += m; D.append(dino(crops)); C.append(col); T.append(th)
            if n % 25 == 0:
                print(f"{n + 1}/{len(jobs)} sources, {len(metas)} renders", flush=True)
    np.save(PAIRS / "dino.npy", np.concatenate(D)); np.save(PAIRS / "colour.npy", np.concatenate(C)); np.save(PAIRS / "thumbs.npy", np.concatenate(T))
    with open(PAIRS / "meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print("done", len(metas), "renders", flush=True)


def _extra_job(row: dict):
    x = load_photo(row["path"], 512)
    return row, dino_crop(x), colour_features_from_array(x).astype(np.float32), np.asarray(Image.fromarray(x).resize((64, 64), Image.BOX))


def extra(a) -> None:
    """Featurise the evaluation groups that are not rendered here, with exactly the same features:
    `synth_test` (13-preset renders of the 81 Common Catalog test scenes: an image pool never used for training),
    `edited` (110 filtered uploads: digitally edited photographs with unknown looks), `real` (1,146 real film scans with
    the uploader's stock claim)."""
    groups = {
        "synth_test": [json.loads(l) for l in open("data/lut/synth/manifest.jsonl") if '"split": "test"' in l],
        "edited": [{"path": str(p), "group": "edited"} for p in sorted(Path("data/lut/synth/negatives").glob("*.jpg"))],
        "real": [json.loads(l) for l in open("data/lut/real/manifest.jsonl")],
    }
    dino = DinoLayers()
    for name, rows in groups.items():
        rows = [r for r in rows if Path(r["path"]).exists()]
        metas, crops, C, T = [], [], [], []
        with Pool(a.workers) as pool:
            for r, cr, col, th in pool.imap(_extra_job, rows, chunksize=4):
                metas.append(r); crops.append(cr); C.append(col); T.append(th)
        d = dino(np.stack(crops))
        out = PAIRS / f"extra_{name}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dino.npy", d); np.save(out / "colour.npy", np.stack(C)); np.save(out / "thumbs.npy", np.stack(T))
        with open(out / "meta.jsonl", "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        print(name, len(metas), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--workers", type=int, default=24); b.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("extra"); e.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"build": build, "extra": extra}[a.cmd](a)


if __name__ == "__main__":
    main()
