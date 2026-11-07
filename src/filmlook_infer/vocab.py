"""Label definitions shared by training, teachers, and inference (output schema v1).

Direction convention: for EVERY attribute, higher = more convincingly film-like / more natural
rendering. Higher is not "more of an effect": heavy fake grain scores LOW on grain_naturalness,
crushed or HDR-flattened shadows score LOW on tonal_depth.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
MODEL_VERSION = "filmlook-ml-v1.0.0"

ATTRIBUTES = [
    "highlight_rolloff",   # highlight roll-off and tonal transitions
    "color_response",      # colour response and palette coherence (null for monochrome)
    "microcontrast",       # organic acutance (high) vs crunchy digital sharpening (low)
    "grain_naturalness",   # texture / grain naturalness
    "tonal_depth",         # tonal depth and shadow rendering
    "film_coherence",      # overall film-aesthetic coherence
]
COLOR_ATTRIBUTES = {"color_response"}  # not applicable to monochrome images

ATTRIBUTE_NAMES = {
    "highlight_rolloff": {"en": "Highlight roll-off & tonal transitions", "zh": "高光过渡与影调衔接"},
    "color_response": {"en": "Color response & palette coherence", "zh": "色彩响应与色调统一"},
    "microcontrast": {"en": "Microcontrast & sharpening", "zh": "微反差与锐化"},
    "grain_naturalness": {"en": "Texture & grain naturalness", "zh": "质感与颗粒自然度"},
    "tonal_depth": {"en": "Tonal depth & shadow rendering", "zh": "影调层次与暗部表现"},
    "film_coherence": {"en": "Overall film-aesthetic coherence", "zh": "整体胶片质感统一度"},
}

# Rubric used by the Claude teacher, by human reviewers, and in documentation.
RUBRIC = {
    "film_score": "0-100. How convincingly the image exhibits the rendering characteristics of analog "
                  "film. 85+: indistinguishable from a good film scan. 60-84: largely film-like with minor "
                  "digital tells. 40-59: neutral or mixed. 20-39: clearly digital or a heavy-handed filter. "
                  "<20: strongly digital/over-processed. A clean, subtle film look can outscore an "
                  "exaggerated film filter. Do not reward grain, warmth, faded blacks, halation, low "
                  "contrast, blur, or vintage subject matter by themselves.",
    "highlight_rolloff": "1 = highlights compress gradually into white with smooth gradients; 0 = hard "
                         "clipping, flat white patches, banding or HDR halos in transitions.",
    "color_response": "1 = coherent, dye-like palette with natural separation; 0 = clipped, garish or "
                      "incoherent colour (oversaturated channels, mismatched casts). null if monochrome.",
    "microcontrast": "1 = organic acutance, edges defined without halos; 0 = crunchy digital sharpening, "
                     "visible halos, brittle fine detail. Very soft/blurred images are NOT automatically 1.",
    "grain_naturalness": "1 = grain/noise that is fine, clumped, mostly luminance and tone-dependent like "
                         "film, or a clean image where no texture is expected; 0 = uniform per-pixel or "
                         "colour speckle noise, overlaid fake grain, or smeared noise reduction.",
    "tonal_depth": "1 = rich shadow gradation, dense but not crushed blacks, believable midtone "
                   "separation; 0 = crushed blacks, flat HDR shadows, or milky lifted blacks with no density.",
    "film_coherence": "1 = all rendering properties agree on one plausible analog look; 0 = contradictory "
                      "cues (e.g. heavy grain on a clinically sharp, clipped digital file).",
}

# Deterministic style tags computed from MEASURED statistics (see style.py).
# Groups are mutually exclusive; a group may be left empty when the evidence is neutral.
STYLE_TAGS = {
    "monochrome": {"en": "black & white", "zh": "黑白"},
    "warm": {"en": "warm", "zh": "暖调"},
    "cool": {"en": "cool", "zh": "冷调"},
    "neutral": {"en": "neutral color", "zh": "中性色"},
    "muted": {"en": "muted", "zh": "低饱和"},
    "vivid": {"en": "vivid", "zh": "高饱和"},
    "soft": {"en": "soft", "zh": "柔和"},
    "crisp": {"en": "crisp", "zh": "锐利"},
    "high_key": {"en": "high-key", "zh": "高调"},
    "low_key": {"en": "low-key", "zh": "低调"},
    "high_contrast": {"en": "high contrast", "zh": "高反差"},
    "low_contrast": {"en": "low contrast", "zh": "低反差"},
    "no_visible_grain": {"en": "no visible grain", "zh": "无明显颗粒"},
    "fine_grain": {"en": "fine grain", "zh": "细颗粒"},
    "coarse_grain": {"en": "coarse grain", "zh": "粗颗粒"},
}
STYLE_GROUPS = {
    "color": ["monochrome"],
    "temperature": ["warm", "cool", "neutral"],
    "saturation": ["muted", "vivid"],
    "sharpness": ["soft", "crisp"],
    "key": ["high_key", "low_key"],
    "contrast": ["high_contrast", "low_contrast"],
    "grain": ["no_visible_grain", "fine_grain", "coarse_grain"],
}
# Mood tags (cinematic / documentary / editorial / nostalgic) are part of the teacher schema but are
# NOT emitted by model v1: no approved labels exist to learn them (see REPORT.md).
MOOD_TAGS = ["cinematic", "documentary", "editorial", "nostalgic"]
