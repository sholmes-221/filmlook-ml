"""Deterministic style tags from measured statistics.

These describe what the image IS (measured), not what the model thinks of it, so they cannot
contradict the pixels (e.g. no colour-temperature tag on a monochrome image). Thresholds are set
from the distribution of the v1 training originals (see REPORT.md) and are part of the model version.
"""
from __future__ import annotations

import math

THRESHOLDS = {
    "warm": 0.055, "cool": -0.015,            # mean(R - B)
    "muted": 0.17, "vivid": 0.40,              # mean HSV-like saturation
    "soft_lap": 4.0e-4, "crisp_lap": 4.0e-3,   # Laplacian variance (median-filtered, 1024 px)
    "high_key": 0.60, "low_key": 0.30,         # mean luma
    "high_contrast": 0.30, "low_contrast": 0.15,  # luma std
    "grain_none": 0.006, "grain_coarse": 0.028,   # high-pass residual std in flat regions
}


def is_monochrome(f: dict) -> bool:
    return bool(f["is_mono"])


def style_tags(f: dict, th: dict = THRESHOLDS) -> list[str]:
    tags: list[str] = []
    mono = is_monochrome(f)
    if mono:
        tags.append("monochrome")
    else:
        if f["warmth"] > th["warm"]:
            tags.append("warm")
        elif f["warmth"] < th["cool"]:
            tags.append("cool")
        else:
            tags.append("neutral")
        if f["sat_mean"] < th["muted"]:
            tags.append("muted")
        elif f["sat_mean"] > th["vivid"]:
            tags.append("vivid")
    if f["lap_var"] < th["soft_lap"]:
        tags.append("soft")
    elif f["lap_var"] > th["crisp_lap"]:
        tags.append("crisp")
    if f["y_mean"] > th["high_key"]:
        tags.append("high_key")
    elif f["y_mean"] < th["low_key"]:
        tags.append("low_key")
    if f["y_std"] > th["high_contrast"]:
        tags.append("high_contrast")
    elif f["y_std"] < th["low_contrast"]:
        tags.append("low_contrast")
    if f["noise_std"] < th["grain_none"]:
        tags.append("no_visible_grain")
    elif f["noise_std"] > th["grain_coarse"]:
        tags.append("coarse_grain")
    else:
        tags.append("fine_grain")
    return tags


def finite(f: dict) -> bool:
    return all(math.isfinite(v) for v in f.values())
