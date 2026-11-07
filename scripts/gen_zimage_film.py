#!/usr/bin/env python
"""Third generator: Z-Image-Turbo image-to-image film treatments (generator family "zimage_i2i").

  python scripts/gen_zimage_film.py --sources 100 --treatments filmgrain_lora,base_film_prompt,base_slide_prompt,vae_identity

Licences (recorded per output): Tongyi-MAI/Z-Image-Turbo Apache-2.0; artificialguybr/FILMGRAIN-REDMOND-ZIMAGE
Apache-2.0 (training data undisclosed). Outputs are treatments, never film-stock examples (film_stock_claim null).
Uses the same source photos and splits as filmsynth and SDXL (reads the filmsynth annotation file for the list).

Memory: bf16 with model CPU offload (transformer 12.3 GB + Qwen3 text encoder about 8 GB do not fit together on
a shared 24 GB card; on a 12 GB RTX 4070 the transformer alone is borderline: use sequential offload or a quantised
transformer there). Turbo is distilled: guidance_scale 0, about 8 DiT forwards; img2img strength 0.35 runs ~3 steps.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import torch
from PIL import Image

from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.annotation.workflow import discover
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.filmsynth import variant_seed

OUT = P.DATA / "synthetic" / "zimage_i2i_v1"
CACHE = P.DATA / "hf_cache"
BASE = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "artificialguybr/FILMGRAIN-REDMOND-ZIMAGE"
TREATMENTS = {
    "filmgrain_lora": {"lora": True, "scale": 0.8, "prompt": "FILMGRAIN, FilmGrainAF, analog film photograph, visible film grain, natural colours"},
    "base_film_prompt": {"lora": False, "scale": 0.0, "prompt": "analog 35mm film photograph, colour negative film look, soft highlights, natural grain"},
    "base_slide_prompt": {"lora": False, "scale": 0.0, "prompt": "slide film photograph, saturated colour reversal film, high contrast, fine grain"},
    "vae_identity": {"lora": False, "scale": 0.0, "prompt": None},
}


def _fit(img: Image.Image, long_side: int) -> Image.Image:
    s = long_side / max(img.size)
    w, h = (max(64, int(round(img.width * s / 16)) * 16), max(64, int(round(img.height * s / 16)) * 16))
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=100)
    ap.add_argument("--treatments", default="filmgrain_lora,base_film_prompt,base_slide_prompt,vae_identity")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-sources-as", default=str(P.DATA / "synthetic" / "film_synth_v1" / "annotations.jsonl"))
    a = ap.parse_args()
    from diffusers import ZImageImg2ImgPipeline
    from huggingface_hub import snapshot_download

    dev = torch.device("cuda")
    print(f"free VRAM {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", file=sys.stderr)
    base_dir = snapshot_download(BASE, cache_dir=str(CACHE), local_files_only=True,
                                 allow_patterns=["*.json", "*.txt", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*"])
    pipe = ZImageImg2ImgPipeline.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    revisions = {BASE: os.path.basename(base_dir)}
    treatments = a.treatments.split(",")
    lora_file = None
    if any(TREATMENTS[t]["lora"] for t in treatments):
        d = snapshot_download(LORA_REPO, cache_dir=str(CACHE), local_files_only=True, allow_patterns=["*.safetensors", "README.md"])
        lora_file = next(f for f in sorted(os.listdir(d)) if f.endswith(".safetensors"))
        pipe.load_lora_weights(d, weight_name=lora_file, adapter_name="filmgrain")
        revisions[LORA_REPO] = os.path.basename(d)

    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    prev = read_jsonl(a.same_sources_as)
    src_ids = sorted({x["source_image_id"] for x in prev})[: a.sources]
    rows = [by_id[i] for i in src_ids if i in by_id]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    anns, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for r in rows:
        img = _fit(load_image(r["path"], max_side=1280), a.long_side)
        for t in treatments:
            cfg = TREATMENTS[t]
            vid = f"{r['family_id']}__zimage_{t}"
            path = OUT / "images" / f"{vid}.jpg"
            if path.exists():
                continue
            seed = variant_seed(r["image_id"], f"zimage_{t}", a.seed, "zimage-i2i-v1")
            if t == "vae_identity":
                with torch.no_grad():
                    vae = pipe.vae
                    x = pipe.image_processor.preprocess(img).to(vae.device, vae.dtype)
                    lat = vae.encode(x).latent_dist.mode()
                    dec = vae.decode(lat).sample
                out = pipe.image_processor.postprocess(dec.float(), output_type="pil")[0]
            else:
                if lora_file:
                    pipe.set_adapters(["filmgrain"], adapter_weights=[cfg["scale"] if cfg["lora"] else 0.0])
                g = torch.Generator(device="cpu").manual_seed(seed)
                out = pipe(prompt=cfg["prompt"], image=img, strength=a.strength, num_inference_steps=a.steps,
                           guidance_scale=0.0, height=img.height, width=img.width, generator=g).images[0]
            out.save(path, quality=92, subsampling=0)
            data = path.read_bytes()
            anns.append({"source_image_id": r["image_id"], "variant_id": vid, "generator": "zimage_i2i", "generator_version": "zimage-i2i-v1",
                         "treatment": t, "treated": t != "vae_identity",
                         "models": {BASE: {"revision": revisions[BASE], "license": "Apache-2.0"},
                                    **({LORA_REPO: {"revision": revisions[LORA_REPO], "file": lora_file, "license": "Apache-2.0", "scale": cfg["scale"]}} if cfg["lora"] else {})},
                         "prompt": cfg["prompt"], "negative_prompt": None, "seed": seed,
                         "strength": a.strength if t != "vae_identity" else 0.0, "steps": a.steps if t != "vae_identity" else 0,
                         "guidance_scale": 0.0 if t != "vae_identity" else None, "input_size": list(img.size),
                         "label_source": "diffusion_image_to_image", "is_real_film_scan": False, "film_stock_claim": None,
                         "split": r["split"], "group_id": r["group_id"], "source_license": r.get("license"), "source_dataset": r["dataset_id"],
                         "source_attribution": r.get("attribution"), "path": str(path.relative_to(P.ROOT)), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"{len(anns)} images, {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", file=sys.stderr)
    with open(OUT / "annotations.jsonl", "a") as fh:
        for x in anns:
            fh.write(json.dumps(x) + "\n")
    summ = {"version": "zimage-i2i-v1", "sources": len(rows), "new_variants": len(anns), "treatments": dict(Counter(x["treatment"] for x in anns)),
            "strength": a.strength, "steps": a.steps, "seconds": round(time.time() - t0, 1),
            "seconds_per_image": round((time.time() - t0) / max(1, len(anns)), 2), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "offload": "model_cpu_offload", "gpu": torch.cuda.get_device_name(0) + " (dev workstation; owner's RTX 4070 has 12 GB)"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
