# Hugging Face film-generation models: research for FilmLook-ML

Compiled 2026-11-07 from Hugging Face API metadata, model cards and licence texts (fetched read-only). Licence
readings are an engineering record, not legal advice. "Commercial training suitability" means: may FilmLook, a
commercial product, generate images with the model and use them to train its own specialized classifier?

Hardware context: the owner trains on an **RTX 4070 (12 GB VRAM)**. Numbers measured here come from the development
workstation (RTX 4090, 24 GB, shared) and are marked as such.

## Starting point: Shakker-Labs/FilmPortrait

- **Architecture:** a LoRA (613 MB `filmfotos.safetensors`) for **FLUX.1-dev** (12B rectified-flow transformer, T5-XXL +
  CLIP text encoders). Trigger words `filmfotos, film grain, reversal film photography`. Text-to-image; the card gives
  no image-to-image workflow (FLUX.1-dev img2img exists in diffusers, so it is possible in principle).
- **Model card:** describes a subdued, low-saturation palette "reminiscent of classic Japanese cinema", with "a subtle
  bias towards Asian features" in portraits. Three example prompts; a comparison against base FLUX.1-dev.
- **Licence:** `other`, `flux-1-dev-non-commercial-license`, linking to the FLUX.1-dev licence. The licence file on
  Hugging Face is gated (login required), so the text was checked through public mirrors and summaries only:
  model use is limited to non-commercial purposes; Black Forest Labs claims no ownership of outputs and outputs are
  not Derivatives; outputs may not be used to train a model competitive with FLUX.1-dev. Whether generating training
  images for a commercial product counts as non-commercial *use of the model* is disputed in community readings.
  **Treated as not permitted for FilmLook without a commercial licence from Black Forest Labs.**
- **Training data provenance:** not disclosed on the card. No dataset is published.
- **Inference requirements:** FLUX.1-dev bf16 needs about 24 GB for the transformer alone; on a 12 GB 4070 it runs only
  with NF4/FP8 quantization plus CPU offload (slow, roughly 30-90 s per 1024 px image from community reports; not
  measured here). The base weights are gated behind acceptance of the licence.
- **Decision:** not used. Recorded only.

## Comparison table

| Model | Base model | Licence | Data provenance | VRAM (12 GB 4070) | Image-to-image | Commercial training suitability | Recommendation |
|---|---|---|---|---|---|---|---|
| [Shakker-Labs/FilmPortrait](https://huggingface.co/Shakker-Labs/FilmPortrait) | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (gated) | FLUX.1-dev non-commercial | undisclosed | needs NF4/FP8 + offload | possible via FLUX img2img, not documented | **no** (non-commercial model licence) | reject |
| [ilkerzgi/krea-2-*-film-grain-lora](https://huggingface.co/ilkerzgi/krea-2-grainy-nineties-film-lora) (6 variants) | Krea 2 (not stated on card) | `other` | undisclosed | unknown | no | unclear licence → **no** | reject |
| [Muapi/kodak-portra-400-analog-film-stocks-footage-f1d-xl](https://huggingface.co/Muapi/kodak-portra-400-analog-film-stocks-footage-f1d-xl) and ~20 other Muapi "f1d-xl" uploads | FLUX.1-dev **and** SDXL variants in one repo (per the name) | metadata says `openrail++` but the FLUX.1-dev half inherits a non-commercial base | re-uploads of third-party Civitai LoRAs; undisclosed | SDXL half fits; FLUX half does not | SDXL half: yes (img2img) | **ambiguous** (licence metadata unverified, aggregator account) | reject for now |
| [artificialguybr/FILMGRAIN-REDMOND-QWENIMAGE](https://huggingface.co/artificialguybr/FILMGRAIN-REDMOND-QWENIMAGE) | [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image) 20B, Apache-2.0 | Apache-2.0 | "Film Grain style images", undisclosed | 20B: needs heavy quantization + offload | via [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) (Apache-2.0, 58 GB) | licences yes; provenance undisclosed | candidate for a later generator-diversity round, not the pilot |
| [artificialguybr/FILMGRAIN-REDMOND-ZIMAGE](https://huggingface.co/artificialguybr/FILMGRAIN-REDMOND-ZIMAGE) | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 6B, Apache-2.0, 8 steps | Apache-2.0 | undisclosed | 6B DiT bf16 ≈ 12 GB + Qwen3 text encoder: offload needed on 4070 | Z-Image-Edit "to be released"; Turbo is text-to-image | licences yes; provenance undisclosed | **second candidate** (fast, permissive); 33 GB download |
| [artificialguybr/FILMGRAIN-REDMOND-FLUXKLEIN9B](https://huggingface.co/artificialguybr/FILMGRAIN-REDMOND-FLUXKLEIN9B) | [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | LoRA Apache-2.0, base `flux-non-commercial-license` | undisclosed | — | yes (base is image-to-image) | **no** (base non-commercial) | reject |
| [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | — | Apache-2.0 | BFL, undisclosed | 12B: NF4 + offload on 4070; 4 steps | img2img pipeline exists in diffusers | yes | possible third generator; 34 GB download; no film LoRA with clear licence found for it |
| [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | — | CreativeML Open RAIL++-M: commercial use allowed, use-based restrictions (Attachment A), "Licensor claims no rights in the Output" | Stability AI, undisclosed | **fits: fp16 ≈ 7-8 GB at 768 px** | yes, mature `StableDiffusionXLImg2ImgPipeline` | **yes** | **pilot generator** |
| [Fihade/Kodak-Portra400-xl-LoRA](https://huggingface.co/Fihade/Kodak-Portra400-xl-LoRA) | SDXL base 1.0 | creativeml-openrail-m | undisclosed ("Kodak porta400 film") | +0.2 GB | with SDXL img2img | licence yes; provenance undisclosed; the name is a treatment id, not a stock label | **pilot treatment** |
| [artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl](https://huggingface.co/artificialguybr/filmgrain-redmond-filmgrain-lora-for-sdxl) | SDXL base 1.0 | bespoke Civitai licence: commercial use of **images** allowed, no derivatives of the weights | undisclosed | +0.2 GB | with SDXL img2img | images yes; weights may not be redistributed or merged | **pilot treatment** |
| [markredito/FilmGrain-LoRA-stablediffusion](https://huggingface.co/markredito/FilmGrain-LoRA-stablediffusion) | SD 1.5 (per card) | artistic-2.0 | undisclosed | fits | yes | licence uncommon; SD 1.5 base quality lower | reserve |
| [dx8152/Qwen-Image-Edit-2509-White_film_to_rendering](https://huggingface.co/dx8152/Qwen-Image-Edit-2509-White_film_to_rendering) | Qwen-Image-Edit | none stated | undisclosed | 20B | yes | no licence → **no** | reject |

Searches run (HF API, sorted by downloads): "film photography", "analog film", "kodak", "fujifilm", "film grain",
"portra", "analog photo", "image-to-image film", "film simulation", "analog img2img". No model with *documented*
training-data provenance was found; every film LoRA says at most "trained on film-style images".

## Decisions

1. **Pilot generator: SDXL base 1.0 image-to-image** with three treatments (Portra-named LoRA, film-grain LoRA, and a
   prompt-only film treatment), strength about 0.35 so composition and subject are preserved. Fits the 4070 in fp16.
2. **Second diffusion generator (later, if the pilot shows transfer):** Z-Image-Turbo (Apache-2.0, 8 steps) with
   the Apache-2.0 film-grain LoRA, run with CPU offload. It diversifies generator fingerprints for the
   leave-one-generator-out test.
3. **Not used:** anything on FLUX.1-dev / FLUX.2-klein (non-commercial), anything with no licence, and Muapi
   re-uploads until their licence can be traced to the original author.
4. Generated images are never labelled with a film stock. The LoRA name "Portra400" is kept only as a treatment id with
   `film_stock_claim: null` in the provenance record.
5. Generator provenance recorded per image: base model id and snapshot revision, LoRA id, file and scale, licence,
   prompt, negative prompt, seed, strength, steps, guidance, input size (`scripts/gen_sdxl_film.py`).

## Limits of this research

- Licence texts were read from the Hugging Face repositories except FLUX.1-dev (gated), which was read through mirrors.
- "Training data provenance: undisclosed" is the finding for every film LoRA. The provenance gap is in the LoRA
  weights, not in FilmLook's use of generated images; it is recorded, not resolved.
- VRAM figures for the 4070 are estimates from parameter counts; only SDXL was measured (on the 4090, see
  `data/synthetic/sdxl_i2i_v1/summary.json`).
