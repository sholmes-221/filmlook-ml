"""Deterministic bilingual explanations from predicted attributes + measured statistics.

Rules:
* A strength/weakness is emitted only from a predicted attribute score (>= STRONG / <= WEAK).
* A *specific cause* (e.g. "clipped highlights") is named only when the corresponding measured
  statistic supports it; otherwise a generic phrase for that attribute is used.
* Colour phrases are never emitted for monochrome images.
"""
from __future__ import annotations

STRONG, WEAK = 0.68, 0.42

# attribute -> (generic strength, generic weakness)
GENERIC = {
    "highlight_rolloff": ({"en": "smooth highlight roll-off and tonal transitions", "zh": "高光过渡平滑，影调衔接自然"},
                          {"en": "abrupt highlight and tonal transitions", "zh": "高光与影调过渡生硬"}),
    "color_response": ({"en": "coherent, natural color palette", "zh": "色彩统一自然"},
                       {"en": "incoherent or harsh color rendering", "zh": "色彩表现生硬或不统一"}),
    "microcontrast": ({"en": "natural edge definition without sharpening halos", "zh": "边缘清晰自然，无锐化白边"},
                      {"en": "brittle, over-processed fine detail", "zh": "细节显得生硬、处理过度"}),
    "grain_naturalness": ({"en": "organic, film-like texture", "zh": "质感自然，接近胶片颗粒"},
                          {"en": "texture that looks digital or artificial", "zh": "质感偏数码或显得人工"}),
    "tonal_depth": ({"en": "rich shadow gradation and tonal depth", "zh": "暗部层次丰富，影调有深度"},
                    {"en": "shallow tonal depth in the shadows", "zh": "暗部层次单薄"}),
    "film_coherence": ({"en": "consistent, believable analog look", "zh": "整体胶片感统一可信"},
                       {"en": "mixed digital and film-like cues", "zh": "数码感与胶片感混杂"}),
}


def _specific(attr: str, weak: bool, f: dict) -> dict | None:
    """Return a more specific phrase only when a measured statistic supports it."""
    if weak:
        if attr == "highlight_rolloff":
            if f["clip_high"] > 0.02 or f["highlight_shoulder"] > 0.25:
                return {"en": "clipped highlights", "zh": "高光溢出"}
            if f["hist_gap_ratio"] > 0.3:
                return {"en": "banding in tonal gradients", "zh": "渐变区域出现断层"}
            if f["local_contrast_ratio"] > 0.2:
                return {"en": "HDR-style halos in tonal transitions", "zh": "影调过渡有HDR式光晕"}
        if attr == "color_response" and f["sat_clip"] > 0.03:
            return {"en": "clipped, oversaturated color", "zh": "色彩过饱和、溢色"}
        if attr == "microcontrast" and f["overshoot"] > 4.0:
            return {"en": "visible sharpening halos", "zh": "可见锐化白边"}
        if attr == "grain_naturalness" and f["noise_std"] >= 0.006:
            if f["chroma_noise_ratio"] > 1.0:
                return {"en": "color speckle noise", "zh": "彩色噪点"}
            if f["noise_autocorr1"] < 0.1:
                return {"en": "uniform overlay grain that looks artificial", "zh": "颗粒均匀如叠加滤镜，显得人工"}
        if attr == "tonal_depth":
            if f["clip_low"] > 0.03:
                return {"en": "crushed shadows", "zh": "暗部死黑"}
            if f["p01"] > 0.14:
                return {"en": "milky, lifted blacks without density", "zh": "黑位发灰，缺乏密度"}
            if f["local_contrast_ratio"] > 0.2:
                return {"en": "flattened, HDR-like shadows", "zh": "暗部被HDR压平"}
    else:
        if attr == "grain_naturalness" and f["noise_std"] < 0.006:
            return {"en": "clean texture without digital noise", "zh": "画面干净，无数码噪点"}
    return None


def strengths_weaknesses(attrs: dict, f: dict, lang: str, k: int = 3) -> tuple[list[str], list[str]]:
    s, w = [], []
    for a, v in attrs.items():
        if v is None:
            continue
        if v >= STRONG:
            s.append((v - STRONG, (_specific(a, False, f) or GENERIC[a][0])[lang]))
        elif v <= WEAK:
            w.append((WEAK - v, (_specific(a, True, f) or GENERIC[a][1])[lang]))
    s.sort(key=lambda t: -t[0])
    w.sort(key=lambda t: -t[0])
    return [t[1] for t in s[:k]], [t[1] for t in w[:k]]


SUMMARY = [  # (min score, en, zh)
    (80, "Strongly film-like rendering.", "画面呈现出强烈而可信的胶片质感。"),
    (65, "Largely film-like, with minor digital tells.", "整体接近胶片质感，仅有少量数码痕迹。"),
    (45, "Mixed: some film-like qualities alongside digital characteristics.", "胶片特征与数码特征并存。"),
    (25, "Reads mostly as digital or heavily processed.", "整体更像数码或后期处理较重的画面。"),
    (0, "Strongly digital or over-processed rendering.", "数码感或过度处理感明显。"),
]

LOW_CONFIDENCE_NOTE = {"en": " Assessment confidence is low for this image.", "zh": "该图片的评估置信度较低。"}

DISCLAIMER = {
    "en": "Automated visual assessment of rendering characteristics. It does not identify the camera, film stock "
          "or photographer, and it is not a judgement of artistic merit.",
    "zh": "本结果为对画面渲染特征的自动视觉评估，不识别相机、胶卷型号或摄影师，也不代表艺术价值评判。",
}


def summary(score: int, lang: str, low_confidence: bool) -> str:
    for lo, en, zh in SUMMARY:
        if score >= lo:
            txt = en if lang == "en" else zh
            break
    if low_confidence:
        txt += LOW_CONFIDENCE_NOTE[lang]
    return txt
