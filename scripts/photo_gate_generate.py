#!/usr/bin/env python
"""Generated images for the photo gate with Z-Image-Turbo (Apache-2.0), locally on the GPU.

  python scripts/photo_gate_generate.py [--n 300]

- ``anime``/``digital_art`` (label 0): anime illustrations, digital paintings, flat vector art, 3D renders, pixel art.
  Two prompt families: ``a`` for training, ``b`` (different styles and subjects) for the held-out test.
- ``ai_photoreal`` (ambiguous, evaluation only): photorealistic generated scenes. They are reported separately and
  never used to train; the gate makes no claim to tell generated from real photographs.
Each file has a record in data/raw/photo_gate/manifest_generated.jsonl (prompt, seed, model, revision, licence).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402

BASE = "Tongyi-MAI/Z-Image-Turbo"
CACHE = P.DATA / "hf_cache"
OUT = P.DATA / "raw" / "photo_gate" / "generated"
SUBJECTS = ["a girl with a camera on a city street", "a cat on a windowsill", "a mountain lake at sunrise", "a crowded night market", "a boy riding a bicycle",
            "a cafe interior with plants", "a rainy street with umbrellas", "a lighthouse by the sea", "two friends at a train station", "a bowl of fruit on a table",
            "an old man reading a newspaper", "a field of sunflowers", "a robot in a workshop", "a dog running on a beach", "a temple in the snow"]
STYLES = {
    "a": ["anime illustration, cel shading, clean line art", "digital painting, painterly brush strokes", "flat vector illustration, simple shapes",
          "children's book watercolor illustration", "manga style black and white ink drawing"],
    "b": ["3D render, pixar style, soft lighting", "pixel art, 16-bit", "chibi anime sticker, white outline", "oil painting in impressionist style",
          "comic book art with halftone dots", "low poly 3D illustration"],
}
PHOTOREAL = "a candid photograph, 35mm film, natural light, realistic"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    from diffusers import ZImagePipeline
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(BASE, cache_dir=str(CACHE), local_files_only=True,
                                 allow_patterns=["*.json", "*.txt", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*"])
    pipe = ZImagePipeline.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # as in gen_zimage_film.py: the full pipeline does not fit beside other GPU work
    rev = Path(base_dir).name
    rng = random.Random(0)
    man_path = OUT / "manifest_generated.jsonl"; OUT.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["path"] for l in man_path.read_text().splitlines()} if man_path.exists() else set()
    plan = [("a", "illustration", a.n), ("b", "illustration", a.n // 3), ("p", "ai_photoreal", a.n // 3)]
    with open(man_path, "a") as man:
        for fam, kind, n in plan:
            d = OUT / f"{kind}_{fam}"; d.mkdir(exist_ok=True)
            for i in range(n):
                path = d / f"{kind}_{fam}_{i:04d}.jpg"
                rel = str(path.relative_to(P.DATA))
                subj = rng.choice(SUBJECTS); style = rng.choice(STYLES[fam]) if fam != "p" else PHOTOREAL
                seed = rng.randint(0, 2**31 - 1); w, h = rng.choice([(640, 832), (832, 640), (768, 768)])
                if rel in done:
                    continue
                g = torch.Generator("cuda").manual_seed(seed)
                img = pipe(prompt=f"{subj}, {style}", num_inference_steps=a.steps, guidance_scale=0.0, height=h, width=w, generator=g).images[0]
                img.convert("RGB").save(path, quality=90)
                man.write(json.dumps({"path": rel, "kind": kind, "family": fam, "label": 0 if kind == "illustration" else None,
                                      "split": {"a": "train", "b": "test", "p": "ambiguous"}[fam], "generator": f"{BASE}@{rev}", "licence": "Apache-2.0 model; outputs generated here",
                                      "prompt": f"{subj}, {style}", "seed": seed, "steps": a.steps}) + "\n")
                man.flush()
            print(kind, fam, "done", flush=True)


if __name__ == "__main__":
    main()
