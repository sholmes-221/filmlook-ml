"""Photographer-style similarity engine v1 (Python reference; js/filmlook-style/engine.js mirrors it exactly).

Pipeline: style features (JS-reproducible) -> interpretable dims -> per-profile evidence (distinctiveness, with a
monochrome gate) -> rarity percentile vs a reference pool of ordinary photos -> top-3 + "no reliable match".
Optionally a learned model (linear / random forest / MLP over the same features) can re-rank profiles; v1 ships
WITHOUT one because no independent labels exist to train or validate it (see REPORT.md).
"""
from __future__ import annotations

import math

from filmlook_ml.style.features import FEATURE_NAMES, FEATURE_VERSION
from filmlook_ml.style.profiles import (
    COLOR_DIMS,
    PHRASES,
    PROFILE_VERSION,
    PROFILES,
    dims_from_features,
    percentile,
    profile_evidence,
)

STYLE_SCHEMA_VERSION = "style-1.0"
STYLE_MODEL_VERSION = "filmlook-style-v1.0.0"
THRESHOLDS = {"min_score": 0.5, "min_percentile": 0.9, "min_compat": 0.7, "strong_score": 0.7,
              "strong_percentile": 0.97, "min_contrast": 0.03}

TEXT = {
    "uncertain": {"en": "No reliable style match: this photo does not show a distinctive enough combination of the "
                        "measured characteristics.",
                  "zh": "未找到可靠的风格匹配：这张照片在可测量的特征上没有足够鲜明的组合。"},
    "low_info": {"en": "No reliable style match: the image has too little tonal information to compare.",
                 "zh": "未找到可靠的风格匹配：画面的影调信息不足，无法比较。"},
    "match": {"en": "Closest measurable style resemblance: {name}.", "zh": "最接近的可测量风格：{name}。"},
    "disclaimer": {"en": "Style resemblance is computed from measurable characteristics (colour, tone, texture, "
                         "composition). It does not identify the photographer or authenticate authorship.",
                   "zh": "风格相似度基于可测量的特征（色彩、影调、质感、构图）计算，不代表识别摄影师或鉴定作品归属。"},
    "differs": {"en": "this style typically shows {x}; this photo does not", "zh": "该风格通常具有{x}，本图则不同"},
}
STRENGTH_TEXT = {"strong": {"en": "strong", "zh": "较强"}, "moderate": {"en": "moderate", "zh": "中等"},
                 "weak": {"en": "weak", "zh": "较弱"}}


def rnd(x: float, n: int) -> float:
    """floor(x*10^n + 0.5)/10^n - identical in JS (Python round() is banker's rounding)."""
    m = 10 ** n
    return math.floor(x * m + 0.5) / m


def _phrase(dim: str, target: float, lang: str) -> str:
    hi, lo = PHRASES[dim][lang]
    return hi if target >= 0.5 else lo


def explain(res: dict, prof: dict, mono: bool, lang: str, k: int = 3) -> tuple[list[str], list[str]]:
    """Supporting dims (progress >= 0.6), strongest first; one contradicting dim (progress < 0) if any.
    Colour dims are never cited for monochrome images."""
    sup = [(p, d) for d, p in res["support"].items() if p >= 0.6 and not (mono and d in COLOR_DIMS)]
    sup.sort(key=lambda x: (-x[0], x[1]))
    con = [(p, d) for d, p in res["support"].items() if p < 0 and not (mono and d in COLOR_DIMS)]
    con.sort(key=lambda x: (x[0], x[1]))
    p = prof["profile"]
    return [_phrase(d, p[d][0], lang) for _, d in sup[:k]], [_phrase(d, p[d][0], lang) for _, d in con[:1]]


def analyze_features(f: dict, quantiles: dict, lang: str = "en", learned=None) -> dict:
    if lang not in ("en", "zh"):
        raise ValueError("unsupported language")
    dims = dims_from_features(f)
    mono = dims["monochrome"] >= 0.5
    rows = []
    for pid, prof in PROFILES.items():
        r = profile_evidence(dims, prof["profile"])
        pct = percentile(r["score"], quantiles[pid])
        if learned is not None:
            pct = learned.get(pid, pct)
        reliable = (r["score"] >= THRESHOLDS["min_score"] and pct >= THRESHOLDS["min_percentile"]
                    and r["compat"] >= THRESHOLDS["min_compat"])
        strength = ("strong" if reliable and r["score"] >= THRESHOLDS["strong_score"]
                    and pct >= THRESHOLDS["strong_percentile"] else "moderate" if reliable else "weak")
        rows.append((pid, prof, r, pct, reliable, strength))
    rows.sort(key=lambda x: (-(x[2]["score"] * (0.5 + 0.5 * x[3])), x[0]))
    low_info = f["y_std"] < THRESHOLDS["min_contrast"]
    matches = []
    for pid, prof, r, pct, reliable, strength in rows[:3]:
        ev, con = explain(r, prof, mono, lang)
        matches.append({"id": pid, "name": prof["name"][lang], "similarity": rnd(r["score"], 3),
                        "percentile": rnd(pct, 3), "reliable": bool(reliable and not low_info),
                        "strength": STRENGTH_TEXT[strength if not low_info else "weak"][lang],
                        "evidence": ev, "counter_evidence": [TEXT["differs"][lang].format(x=c) for c in con],
                        "basis": prof["basis"][lang]})
    status = "match" if matches[0]["reliable"] else "uncertain"
    msg = (TEXT["low_info"][lang] if low_info else TEXT["uncertain"][lang]) if status == "uncertain" \
        else TEXT["match"][lang].format(name=matches[0]["name"])
    return {"schema_version": STYLE_SCHEMA_VERSION, "model_version": STYLE_MODEL_VERSION, "language": lang,
            "status": status, "message": msg, "monochrome": bool(mono), "matches": matches,
            "dims": {k: rnd(v, 4) for k, v in dims.items()}, "disclaimer": TEXT["disclaimer"][lang]}


def model_payload(quantiles: dict, learned_model: dict | None = None) -> dict:
    """Everything the JS engine needs (exported as model-data.js)."""
    from filmlook_ml.style.profiles import DIMS, SIGMA
    return {"model_version": STYLE_MODEL_VERSION, "schema_version": STYLE_SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION, "profile_version": PROFILE_VERSION, "feature_names": FEATURE_NAMES,
            "dims": {k: list(v) for k, v in DIMS.items()}, "color_dims": sorted(COLOR_DIMS), "sigma": SIGMA,
            "profiles": {pid: {"name": p["name"], "basis": p["basis"],
                               "profile": {d: list(tw) for d, tw in p["profile"].items()}}
                         for pid, p in PROFILES.items()},
            "quantiles": quantiles, "thresholds": THRESHOLDS, "phrases": {k: {lg: list(v) for lg, v in d.items()}
                                                                         for k, d in PHRASES.items()},
            "text": TEXT, "strength_text": STRENGTH_TEXT, "learned": learned_model}


def _finite(x: float) -> bool:
    return math.isfinite(x)
