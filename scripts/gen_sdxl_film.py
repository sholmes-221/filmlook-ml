#!/usr/bin/env python
"""Controlled image-to-image film treatments with SDXL base 1.0 + film LoRAs (generator family "sdxl_i2i").

  python scripts/gen_sdxl_film.py --sources 100 --treatments portra_lora,filmgrain_lora,base_film_prompt --strength 0.35

Second, independent generator for the leave-one-generator-out study (the first is filmlook_ml.filmsynth). Every
output records source id, generator, model ids and revisions, licences, prompt, seed, strength and the LoRA scale.
Outputs are NOT examples of any film stock; the LoRA names are treatment ids only.

Licences (recorded per output):
- stabilityai/stable-diffusion-xl-base-1.0: CreativeML Open RAIL++-M (commercial use allowed subject to the
  use-based restrictions; the licensor claims no rights in outputs).
- Fihade/Kodak-Portra400-xl-LoRA: creativeml-openrail-m (training data undisclosed).
- artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl: bespoke Civitai licence, commercial use of IMAGES
  allowed, derivatives of the weights not allowed (training data undisclosed).

Memory: fp16, VAE slicing, 768 px long side; about 7-8 GB VRAM, so it fits an RTX 4070 (12 GB). Content preservation
is checked afterwards (scripts/synth_experiment_v2.py), not assumed.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image

from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.annotation.workflow import discover
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.filmsynth import variant_seed

OUT = P.DATA / "synthetic" / os.environ.get("SDXL_OUT", "sdxl_i2i_v1")
CACHE = P.DATA / "hf_cache"
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
LORAS = {
    "portra_lora": {"repo": "Fihade/Kodak-Portra400-xl-LoRA", "file": None, "scale": 0.8,
                    "prompt": "Portra 400, film photograph, natural colours, soft contrast, fine grain",
                    "license": "creativeml-openrail-m"},
    "filmgrain_lora": {"repo": "artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl", "file": None, "scale": 0.8,
                       "prompt": "Film Grain, FilmGrainAF, analog film photograph, visible grain",
                       "license": "bespoke-lora-trained-license (Civitai: commercial use of images allowed, no derivatives)"},
    "base_film_prompt": {"repo": None, "file": None, "scale": 0.0,
                         "prompt": "analog 35mm film photograph, colour negative film look, natural grain, soft highlights",
                         "license": None},
    "base_slide_prompt": {"repo": None, "file": None, "scale": 0.0,
                          "prompt": "slide film photograph, saturated colour reversal film, high contrast, fine grain",
                          "license": None},
    "vae_identity": {"repo": None, "file": None, "scale": 0.0, "prompt": None, "license": None,
                     "note": "nuisance-parity control: SDXL VAE encode/decode only, no diffusion"},
    "base_bw_prompt": {"repo": None, "file": None, "scale": 0.0,
                       "prompt": "black and white 35mm film photograph, visible grain, rich tonality",
                       "license": None},
}
NEGATIVE = "painting, illustration, cartoon, 3d render, text, watermark, deformed"


def _fit(img: Image.Image, long_side: int) -> Image.Image:
    s = long_side / max(img.size)
    w, h = (max(64, int(round(img.width * s / 64)) * 64), max(64, int(round(img.height * s / 64)) * 64))
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=100)
    ap.add_argument("--treatments", default="portra_lora,filmgrain_lora,base_film_prompt")
    ap.add_argument("--strength", type=float, default=0.35, help="img2img denoising strength (content preservation)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-sources-as", default=str(P.DATA / "synthetic" / "film_synth_v1" / "annotations.jsonl"),
                    help="reuse the filmsynth source selection so both generators share sources and splits")
    a = ap.parse_args()
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from huggingface_hub import snapshot_download

    dev = torch.device("cuda")
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"free VRAM {free:.1f} GB", file=sys.stderr)
    if free < 9:
        raise SystemExit("need about 9 GB free VRAM for SDXL fp16 img2img")
    sdxl_patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "text_encoder/config.json",
                     "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json",
                     "text_encoder_2/model.fp16.safetensors", "unet/config.json",
                     "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
                     "vae/diffusion_pytorch_model.fp16.safetensors"]
    base_dir = snapshot_download(BASE, cache_dir=str(CACHE), local_files_only=True, allow_patterns=sdxl_patterns)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_dir, torch_dtype=torch.float16, variant="fp16",
                                                            use_safetensors=True).to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    revisions = {BASE: os.path.basename(base_dir)}
    treatments = a.treatments.split(",")
    for t in treatments:
        cfg = LORAS[t]
        if cfg["repo"]:
            d = snapshot_download(cfg["repo"], cache_dir=str(CACHE), local_files_only=True,
                                  allow_patterns=["*.safetensors", "README.md"])
            revisions[cfg["repo"]] = os.path.basename(d)
            wfile = next(f for f in sorted(os.listdir(d)) if f.endswith(".safetensors"))
            pipe.load_lora_weights(d, weight_name=wfile, adapter_name=t)
            cfg["file"] = wfile

    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    prev = read_jsonl(a.same_sources_as) if os.path.exists(a.same_sources_as) else []
    src_ids = sorted({x["source_image_id"] for x in prev})[: a.sources] if prev else [r["image_id"] for r in eligible[: a.sources]]
    rows = [by_id[i] for i in src_ids if i in by_id]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    anns, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for r in rows:
        img = _fit(load_image(r["path"], max_side=1280), a.long_side)
        for t in treatments:
            cfg = LORAS[t]
            vid = f"{r['family_id']}__sdxl_{t}"
            path = OUT / "images" / f"{vid}.jpg"
            seed = variant_seed(r["image_id"], f"sdxl_{t}", a.seed, os.environ.get("SDXL_OUT", "sdxl-i2i-v1"))
            if path.exists():
                continue
            if cfg["repo"]:
                pipe.set_adapters([t], adapter_weights=[cfg["scale"]])
            else:
                pipe.set_adapters([x for x in treatments if LORAS[x]["repo"]], adapter_weights=[0.0] * sum(bool(LORAS[x]["repo"]) for x in treatments)) \
                    if any(LORAS[x]["repo"] for x in treatments) else None
            if t == "vae_identity":
                with torch.no_grad():
                    xin = pipe.image_processor.preprocess(img).to(dev, dtype=torch.float32)
                    vae = pipe.vae
                    was = vae.dtype
                    vae.to(torch.float32)                     # SDXL's VAE is unstable in fp16; the pipeline upcasts too
                    lat = vae.encode(xin).latent_dist.mode()
                    dec = vae.decode(lat).sample
                    vae.to(was)
                out = pipe.image_processor.postprocess(dec.to(torch.float32), output_type="pil")[0]
            else:
                g = torch.Generator(device=dev).manual_seed(seed)
                out = pipe(prompt=cfg["prompt"], negative_prompt=NEGATIVE, image=img, strength=a.strength,
                           num_inference_steps=a.steps, guidance_scale=5.0, generator=g).images[0]
            out.save(path, quality=92, subsampling=0)
            data = path.read_bytes()
            anns.append({"source_image_id": r["image_id"], "variant_id": vid, "generator": "sdxl_i2i",
                         "generator_version": os.environ.get("SDXL_OUT", "sdxl-i2i-v1"), "treatment": t,
                         "models": {BASE: {"revision": revisions[BASE], "license": "CreativeML Open RAIL++-M"},
                                    **({cfg["repo"]: {"revision": revisions[cfg["repo"]], "file": cfg["file"],
                                                      "license": cfg["license"], "scale": cfg["scale"]}} if cfg["repo"] else {})},
                         "prompt": cfg["prompt"], "negative_prompt": NEGATIVE if t != "vae_identity" else None, "seed": seed,
                         "strength": a.strength if t != "vae_identity" else 0.0, "steps": a.steps if t != "vae_identity" else 0,
                         "guidance_scale": 5.0 if t != "vae_identity" else None, "input_size": list(img.size),
                         "treated": t != "vae_identity",
                         "label_source": "diffusion_image_to_image", "is_real_film_scan": False,
                         "film_stock_claim": None, "split": r["split"], "group_id": r["group_id"],
                         "source_license": r.get("license"), "source_dataset": r["dataset_id"],
                         "source_attribution": r.get("attribution"), "path": str(path.relative_to(P.ROOT)),
                         "sha256": hashlib.sha256(data).hexdigest()})
        print(f"{len(anns)} images, {time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", file=sys.stderr)
    with open(OUT / "annotations.jsonl", "a") as fh:
        for x in anns:
            fh.write(json.dumps(x) + "\n")
    summ = {"version": "sdxl-i2i-v1", "sources": len(rows), "new_variants": len(anns),
            "treatments": dict(Counter(x["treatment"] for x in anns)), "strength": a.strength, "steps": a.steps,
            "seconds": round(time.time() - t0, 1), "seconds_per_image": round((time.time() - t0) / max(1, len(anns)), 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gpu": torch.cuda.get_device_name(0) + " (dev workstation; the owner's RTX 4070 has 12 GB)"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
