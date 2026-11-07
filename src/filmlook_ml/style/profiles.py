"""Interpretable style dimensions + photographer-style profiles (v1).

Dimensions map JS-reproducible features to [0,1] with a logistic on (x - center) / scale. Centers/scales are
the median and ~IQR/2 of the v1 real-photo pool (819 CommonCatalog CC-BY originals), so 0.5 means "typical".

Profiles are HYPOTHESES built from widely published descriptions of each photographer's visual approach
(museum, publisher and critical texts). They were NOT learned from the photographers' images: no licensed
reference images are used. A match means "shares these measurable characteristics", never authorship.
"""
from __future__ import annotations

import math

PROFILE_VERSION = "style-profiles-v1"

# dim: (feature, center, scale, sign)   value = sigmoid(sign * (x - center) / scale)
DIMS = {
    "saturation": ("sat_mean", 0.28, 0.12, 1),
    "accent": ("sat_concentration", 0.21, 0.04, 1),
    "contrast": ("y_std", 0.237, 0.045, 1),
    "darkness": ("y_mean", 0.438, 0.10, -1),
    "shadow_mass": ("dark_cells", 0.11, 0.12, 1),
    "softness": ("lap_logvar", -3.97, 0.75, -1),
    "grain": ("noise", 0.0133, 0.0045, 1),
    "warmth": ("warmth", 0.024, 0.05, 1),
    "red_yellow": ("hue_ry", 0.5, 0.25, 1),
    "high_key": ("bright_frac", 0.095, 0.08, 1),
    "complexity": ("edge_frac", 0.12, 0.065, 1),
    "spotlight": ("center_ratio", 1.0, 0.18, 1),
}
DIM_NAMES = ["monochrome", *DIMS]
COLOR_DIMS = {"saturation", "accent", "warmth", "red_yellow"}

PHRASES = {  # dim: {lang: (high, low)}
    "monochrome": {"en": ("black-and-white rendering", "colour rendering"), "zh": ("黑白影调", "彩色表现")},
    "saturation": {"en": ("rich, saturated colour", "restrained, muted colour"), "zh": ("浓郁饱和的色彩", "克制低饱和的色彩")},
    "accent": {"en": ("colour concentrated in a few accent areas", "colour spread evenly across the frame"),
               "zh": ("色彩集中在少数点缀区域", "色彩均匀分布于画面")},
    "contrast": {"en": ("strong tonal contrast", "gentle, low contrast"), "zh": ("强烈的明暗反差", "柔和的低反差")},
    "darkness": {"en": ("low-key, dark exposure", "bright, open exposure"), "zh": ("低调偏暗的曝光", "明亮通透的曝光")},
    "shadow_mass": {"en": ("large areas of deep shadow", "few large shadow areas"), "zh": ("大面积深重阴影", "少有大块阴影")},
    "softness": {"en": ("soft, diffused detail", "crisp, precise detail"), "zh": ("柔和朦胧的细节", "清晰锐利的细节")},
    "grain": {"en": ("visible grain / texture", "clean, fine texture"), "zh": ("明显的颗粒质感", "干净细腻的质感")},
    "warmth": {"en": ("warm colour balance", "cool colour balance"), "zh": ("偏暖的色调", "偏冷的色调")},
    "red_yellow": {"en": ("palette led by reds and yellows", "palette led by greens and blues"),
                   "zh": ("以红黄为主的色彩", "以绿蓝为主的色彩")},
    "high_key": {"en": ("airy, high-key highlights", "few bright highlight areas"), "zh": ("通透的高调亮部", "亮部区域较少")},
    "complexity": {"en": ("busy, layered frame with many edges", "simple, uncluttered frame"),
                   "zh": ("层次繁复、元素密集的画面", "简洁不杂乱的画面")},
    "spotlight": {"en": ("light concentrated in the centre", "light spread to the edges"),
                  "zh": ("光线集中于画面中央", "光线分布至画面边缘")},
}

# profile: {dim: (target, weight)}; descriptions are deliberately about measurable rendering, not subject matter.
PROFILES = {
    "saul_leiter": {"name": {"en": "Saul Leiter", "zh": "索尔·雷特"},
                    "basis": {"en": "muted colour with small saturated accents, soft focus, large dark masses",
                              "zh": "低饱和基调中的小块浓烈色彩、柔焦、大面积暗部"},
                    "profile": {"monochrome": (0.0, 2.0), "saturation": (0.35, 1.0), "accent": (0.8, 1.5),
                                "softness": (0.7, 1.0), "shadow_mass": (0.65, 1.0), "darkness": (0.6, 0.7),
                                "red_yellow": (0.7, 0.6)}},
    "william_eggleston": {"name": {"en": "William Eggleston", "zh": "威廉·埃格尔斯顿"},
                          "basis": {"en": "saturated, dye-transfer-like colour in even, everyday light",
                                    "zh": "如染印法般饱和的色彩与平实的日常光线"},
                          "profile": {"monochrome": (0.0, 2.0), "saturation": (0.8, 1.5), "red_yellow": (0.65, 0.8),
                                      "contrast": (0.5, 0.6), "softness": (0.3, 0.6), "shadow_mass": (0.3, 0.6),
                                      "accent": (0.45, 0.5)}},
    "daido_moriyama": {"name": {"en": "Daido Moriyama", "zh": "森山大道"},
                       "basis": {"en": "high-contrast black and white, heavy grain, blur (are-bure-boke)",
                                 "zh": "高反差黑白、粗颗粒与晃动模糊（粗・晃・暈）"},
                       "profile": {"monochrome": (1.0, 2.5), "contrast": (0.9, 1.5), "grain": (0.85, 1.2),
                                   "softness": (0.6, 0.6), "darkness": (0.65, 0.8), "shadow_mass": (0.7, 0.8)}},
    "alex_webb": {"name": {"en": "Alex Webb", "zh": "亚历克斯·韦伯"},
                  "basis": {"en": "intense warm colour against deep, hard-edged shadows in layered frames",
                            "zh": "强烈暖色与深重硬朗阴影并置的多层次画面"},
                  "profile": {"monochrome": (0.0, 2.0), "saturation": (0.8, 1.2), "contrast": (0.85, 1.2),
                              "shadow_mass": (0.8, 1.3), "warmth": (0.75, 0.8), "complexity": (0.7, 0.8),
                              "softness": (0.3, 0.5)}},
    "fan_ho": {"name": {"en": "Fan Ho", "zh": "何藩"},
               "basis": {"en": "luminous black and white, dramatic light shafts, clean geometric separation",
                         "zh": "光影戏剧化的黑白画面与清晰的几何分隔"},
               "profile": {"monochrome": (1.0, 2.5), "contrast": (0.8, 1.2), "shadow_mass": (0.65, 1.0),
                           "complexity": (0.3, 0.8), "grain": (0.35, 0.6), "spotlight": (0.7, 0.6)}},
    "stephen_shore": {"name": {"en": "Stephen Shore", "zh": "斯蒂芬·肖尔"},
                      "basis": {"en": "neutral, even colour, crisp large-format detail, gentle contrast",
                                "zh": "中性平和的色彩、大画幅般清晰的细节与柔和反差"},
                      "profile": {"monochrome": (0.0, 2.0), "saturation": (0.45, 0.8), "contrast": (0.35, 1.0),
                                  "softness": (0.15, 1.2), "grain": (0.15, 0.8), "darkness": (0.35, 0.6),
                                  "shadow_mass": (0.2, 0.6)}},
    "fred_herzog": {"name": {"en": "Fred Herzog", "zh": "弗雷德·赫尔佐格"},
                    "basis": {"en": "Kodachrome-like saturated reds and warm street colour",
                              "zh": "类似柯达克罗姆的饱和红色与温暖的街头色彩"},
                    "profile": {"monochrome": (0.0, 2.0), "saturation": (0.7, 1.0), "red_yellow": (0.8, 1.2),
                                "warmth": (0.7, 0.8), "contrast": (0.6, 0.6), "softness": (0.35, 0.5)}},
    "rinko_kawauchi": {"name": {"en": "Rinko Kawauchi", "zh": "川内伦子"},
                       "basis": {"en": "airy, high-key light with soft pastel colour and low contrast",
                                 "zh": "通透的高调光线、柔和的粉彩色与低反差"},
                       "profile": {"monochrome": (0.0, 2.0), "high_key": (0.85, 1.5), "darkness": (0.1, 1.2),
                                   "contrast": (0.25, 1.0), "saturation": (0.3, 0.8), "softness": (0.65, 0.8)}},
    "ernst_haas": {"name": {"en": "Ernst Haas", "zh": "恩斯特·哈斯"},
                   "basis": {"en": "saturated colour dissolved by motion blur and soft focus",
                             "zh": "被动态模糊和柔焦晕开的饱和色彩"},
                   "profile": {"monochrome": (0.0, 2.0), "saturation": (0.75, 1.0), "softness": (0.85, 1.4),
                               "complexity": (0.35, 0.6), "contrast": (0.5, 0.4)}},
}
PROFILE_IDS = list(PROFILES)
SIGMA = 0.22


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def dims_from_features(f: dict) -> dict[str, float]:
    x = dict(f)
    x["hue_ry"] = f["hue_red"] + f["hue_yellow"]
    out = {"monochrome": max(f["mono"], _sig((0.12 - f["sat_p95"]) / 0.02))}
    for d, (feat, c, s, sign) in DIMS.items():
        out[d] = _sig(sign * (x[feat] - c) / s)
    return out


def profile_similarity(dims: dict, prof: dict, sigma: float = SIGMA) -> float:
    num = sum(w * (dims[k] - t) ** 2 for k, (t, w) in prof.items())
    den = sum(w for _, w in prof.values())
    return math.exp(-num / den / (2 * sigma * sigma))


def profile_scores(dims: dict) -> dict[str, float]:
    return {pid: profile_similarity(dims, p["profile"]) for pid, p in PROFILES.items()}


def dim_progress(d: float, t: float) -> float:
    """How far the photo moved from typical (0.5) toward the target t, in [-1, 1].
    Overshoot beyond the target is penalised (p > 1 -> 2 - p), so an extreme photo is not a "perfect" match.
    Near-neutral targets (|t - 0.5| < 0.1) are scored by closeness instead."""
    if abs(t - 0.5) < 0.1:
        return max(-1.0, min(1.0, 1.0 - abs(d - t) / 0.25))
    p = (d - 0.5) / (t - 0.5)
    if p > 1.0:
        p = 2.0 - p
    return max(-1.0, min(1.0, p))


def profile_evidence(dims: dict, prof: dict) -> dict:
    """v1 match score. monochrome is a GATE (compat = 1 - |mono - target|); other dims give weighted progress
    toward the profile's distinctive targets. A typical photo scores ~0; score = compat * max(evidence, 0)."""
    t_m, _ = prof.get("monochrome", (None, 0))
    compat = 1.0 if t_m is None else 1.0 - abs(dims["monochrome"] - t_m)
    num = den = 0.0
    support = {}
    for k, (t, w) in prof.items():
        if k == "monochrome":
            continue
        pr = dim_progress(dims[k], t)
        num += w * pr
        den += w
        support[k] = pr
    ev = num / den if den > 0 else 0.0
    return {"score": compat * max(ev, 0.0), "evidence": ev, "compat": compat, "support": support}


def calibrate_rarity(scores_by_profile: dict[str, list[float]], n_q: int = 101) -> dict[str, list[float]]:
    """Per-profile score quantiles on a REFERENCE pool of ordinary photos -> percentile lookup."""
    import numpy as np
    return {pid: [float(x) for x in np.quantile(np.asarray(v), np.linspace(0, 1, n_q))]
            for pid, v in scores_by_profile.items()}


def percentile(score: float, q: list[float]) -> float:
    """Fraction of reference photos scoring below `score` (piecewise-linear on the quantile table)."""
    n = len(q) - 1
    if score <= q[0]:
        return 0.0
    if score >= q[-1]:
        return 1.0
    lo = 0
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if q[mid] <= score:
            lo = mid
        else:
            hi = mid
    span = q[hi] - q[lo]
    frac = (score - q[lo]) / span if span > 0 else 0.0
    return (lo + frac) / n
