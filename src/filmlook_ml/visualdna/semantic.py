"""Semantic description of a photograph: what is in it (zero-shot SigLIP) and how it is rendered (pixel measurements).

Three kinds of output are kept apart, because they mean different things:

- ``detected``: objects, scenes and composition tags from the shared vocabulary, scored by SigLIP zero-shot
  (``google/siglip-base-patch16-224``, Apache-2.0). A score is SigLIP's sigmoid output for "a photo of ...". It is a
  model estimate, uncalibrated until ``thresholds`` are fitted on labelled data; tags are reported only above their
  threshold.
- ``measured``: tonal attributes computed from pixels (monochrome, contrast, dark or bright tones, saturation). These
  are facts about the file, not interpretations.
- ``interpretation``: mood words scored the same zero-shot way and always labelled as a subjective reading.

Nothing here generates free text, and nothing here names a photographer.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from PIL import Image

from filmlook_ml import paths as P

MODEL = "google/siglip-base-patch16-224"
VOCAB_PATH = P.ROOT / "knowledge" / "visual_dna" / "_vocabulary.json"
MEASURED_TAGS = ("monochrome", "colour", "high contrast", "low contrast", "dark tones", "bright airy exposure", "muted colour", "saturated colour")
MOODS = ("melancholic", "lonely", "calm", "joyful", "tense", "nostalgic", "dreamy")

# Prompts written for zero-shot retrieval. Objects and scenes use the tag itself; the few abstract tags get a phrase.
PROMPT = {
    "mother and child": "a photo of a mother holding her child",
    "face close-up": "a close-up photo of a person's face",
    "city street at night": "a photo of a city street at night",
    "farm field": "a photo of a farm field",
    "rural road": "a photo of a rural dirt road",
    "old building": "a photo of an old stone building",
    "vernacular house": "a photo of a simple wooden house",
    "shop window": "a photo of a shop window display",
    "construction site": "a photo of construction workers on a steel frame",
    "visible grain": "a grainy film photograph",
    "motion blur": "a blurry photo with motion blur",
    "out of focus": "an out of focus photo",
    "sharp detail": "a very sharp detailed photo",
    "soft light": "a photo in soft diffuse light",
    "backlight": "a backlit photo with glowing light",
    "night": "a photo taken at night",
    "flash": "a photo taken with direct flash",
    "soft pastel colour": "a photo with soft pastel colours",
    "reflection": "a photo of a reflection in a window or puddle",
    "through glass": "a photo taken through a wet window",
    "layered frame": "a photo with layered foreground objects partly blocking the view",
    "silhouette": "a photo of dark silhouettes against a bright sky",
    "negative space": "a minimalist photo with large empty space",
    "tilted frame": "a photo with a tilted, skewed horizon",
    "frontal view": "a straight frontal photo of a building facade",
    "close-up": "a close-up photo of a small object",
    "geometry": "a photo with strong geometric lines and shapes",
    "square format": "a square format photo",
    "empty scene": "a photo of an empty street with no people",
    "candid gesture": "a candid street photo of a person in mid-step",
    "posed portrait": "a posed portrait photo of a person looking at the camera",
    "crow": "a photo of a crow or raven",
}


def vocabulary() -> dict:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))["tags"]


def zero_shot_tags() -> list[str]:
    return [t for t in vocabulary() if t not in MEASURED_TAGS]


def prompt(tag: str) -> str:
    return PROMPT.get(tag, f"a photo of {'an' if tag[0] in 'aeiou' else 'a'} {tag}" if " " not in tag else f"a photo of {tag}")


def measure(img: Image.Image) -> dict[str, float]:
    """Pixel facts in [0, 1]; thresholds below are conventional and documented, not learned."""
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    luma = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    chroma = a.max(axis=2) - a.min(axis=2)
    p5, p50, p95 = np.percentile(luma, [5, 50, 95])
    sat95 = float(np.percentile(chroma, 95))
    return {"chroma_p95": sat95, "luma_p5": float(p5), "luma_median": float(p50), "luma_p95": float(p95), "contrast_range": float(p95 - p5), "luma_std": float(luma.std())}


def measured_tags(m: dict[str, float]) -> dict[str, bool]:
    mono = m["chroma_p95"] < 0.10
    return {
        "monochrome": mono, "colour": not mono,
        "high contrast": m["contrast_range"] > 0.80 or m["luma_std"] > 0.28,
        "low contrast": m["contrast_range"] < 0.45,
        "dark tones": m["luma_median"] < 0.28,
        "bright airy exposure": m["luma_median"] > 0.65 and m["luma_p5"] > 0.25,
        "muted colour": (not mono) and m["chroma_p95"] < 0.35,
        "saturated colour": (not mono) and m["chroma_p95"] > 0.60,
    }


class Tagger:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(MODEL).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.tags = zero_shot_tags()
        self.text = self._encode_text([prompt(t) for t in self.tags])
        self.mood_text = self._encode_text([f"a {m} photo" for m in MOODS])

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.proc(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            t = self.model.get_text_features(**inp)
            t = getattr(t, "pooler_output", t)
            return self.torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()

    def image_embeddings(self, images: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            inp = self.proc(images=[i.convert("RGB") for i in images], return_tensors="pt").to(self.device)
            v = self.model.get_image_features(**inp)
            v = getattr(v, "pooler_output", v)
            return self.torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

    def scores(self, emb: np.ndarray, text: np.ndarray) -> np.ndarray:
        scale = float(self.model.logit_scale.exp()); bias = float(self.model.logit_bias)
        return 1.0 / (1.0 + np.exp(-(emb @ text.T * scale + bias)))

    def describe(self, img: Image.Image, thresholds: dict[str, float] | None = None) -> dict:
        emb = self.image_embeddings([img])
        s = self.scores(emb, self.text)[0]; mood = self.scores(emb, self.mood_text)[0]
        th = thresholds or {}
        detected = {t: round(float(v), 4) for t, v in zip(self.tags, s) if v >= th.get(t, 0.10)}
        m = measure(img)
        return {
            "detected": dict(sorted(detected.items(), key=lambda kv: -kv[1])),
            "measured": {k: v for k, v in measured_tags(m).items() if v}, "measurements": m,
            "interpretation": {"mood": {k: round(float(v), 4) for k, v in zip(MOODS, mood) if v >= 0.05}, "note": "subjective visual reading"},
            "model": MODEL,
        }


@lru_cache(maxsize=1)
def default_tagger() -> Tagger:
    return Tagger()
