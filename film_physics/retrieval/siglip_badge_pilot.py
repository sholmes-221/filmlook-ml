"""Pilot (2026-11-09): zero-shot SigLIP (google/siglip-base-patch16-224, Apache-2.0, cached locally) for badge scene
concepts on CC0 / public-domain photographs. Usage: python siglip_badge_pilot.py <dir with light/img and e2e/images>.
A pilot on 20 photos, not a validation; see docs/research/siglip-badge-concepts-pilot.md."""
import glob, os, sys, json
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
S = sys.argv[1]
m = AutoModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True).eval()
p = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", local_files_only=True)
concepts = {"umbrella": "a photo of a person holding an umbrella", "moon": "a photo of the moon in the sky", "duck": "a photo of ducks",
            "bird": "a photo of a bird", "flower": "a photo of flowers", "street": "a photo of a city street", "night": "a photo taken at night",
            "candle": "a photo of a candle", "stage": "a photo of a concert stage", "stapler": "a photo of a stapler"}
files = sorted(glob.glob(f"{S}/light/img/*.jpg")) + [f"{S}/e2e/images/{n}" for n in ("umbrella.jpg",)]
truth = {"umbrella_man": {"umbrella", "street"}, "moon_full": {"moon", "night"}, "moon_branches": {"moon", "night"}, "hl_duck_glitter": {"duck", "bird"},
         "stapler": {"stapler"}, "candle": {"candle"}, "beam_guitarist": {"stage"}, "stage_silhouettes": {"stage"}, "neg_highkey_bird": {"bird", "flower"},
         "soft_village1": set(), "soft_village2": set(), "umbrella": set(), "hard_steeple": {"street"}}
texts = list(concepts.values())
rows = {}
with torch.no_grad():
    t = p(text=texts, padding="max_length", return_tensors="pt")
    tf = m.get_text_features(**t)
    tf = getattr(tf, "pooler_output", tf)
    for f in files:
        n = os.path.basename(f)[:-4]
        im = Image.open(f).convert("RGB")
        x = p(images=im, return_tensors="pt")
        imf = m.get_image_features(**x)
        imf = getattr(imf, "pooler_output", imf)
        logits = (imf / imf.norm(dim=-1, keepdim=True)) @ (tf / tf.norm(dim=-1, keepdim=True)).T * m.logit_scale.exp() + m.logit_bias
        prob = torch.sigmoid(logits)[0].tolist()
        rows[n] = {k: round(v, 3) for k, v in zip(concepts, prob)}
for n, r in rows.items():
    hits = sorted([k for k, v in r.items() if v >= 0.1])
    tr = sorted(truth.get(n, set())) if n in truth else "?"
    print(f"{n:18s} detected(>=0.1)={hits} | truth={tr} | top={max(r, key=r.get)}:{max(r.values())}")
json.dump(rows, open(f"{S}/bench/siglip_rows.json", "w"), indent=1)
