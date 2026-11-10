"""Human-readable colour characteristics of each preset, derived from the parsed parameters (no model, no pixels).

    python -m research.lut.explain [data/lut/presets.json]

Used for the "why this label" layer: every sentence names a parameter that is in the XMP, so it is auditable.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from research.lut.forward_model import BANDS, Preset, load_presets

HUE_WORDS = [(15, "red"), (45, "orange"), (70, "yellow"), (150, "green"), (200, "teal"), (260, "blue"), (300, "purple"), (345, "magenta"), (361, "red")]


def hue_word(h: float) -> str:
    h = h % 360
    return next(w for lim, w in HUE_WORDS if h < lim)


def describe(p: Preset) -> list[str]:
    s = []
    lut = p.curve_lut(256)
    black = lut[0].mean() * 255
    white = lut[-1].mean() * 255
    mid = lut[128].mean() * 255
    if black >= 8:
        s.append(f"lifted blacks (shadows start at {black:.0f}/255: faded, matte shadows)")
    if white <= 250:
        s.append(f"compressed whites (highlights stop at {white:.0f}/255)")
    if mid - 128 > 6:
        s.append(f"brighter midtones (+{mid - 128:.0f}/255 at mid grey)")
    elif 128 - mid > 6:
        s.append(f"darker midtones ({mid - 128:.0f}/255 at mid grey)")
    slope_sh = (lut[64].mean() - lut[16].mean()) / (48 / 255)
    slope_hi = (lut[240].mean() - lut[176].mean()) / (64 / 255)
    if slope_sh < 0.85:
        s.append("soft shadow contrast")
    if slope_hi < 0.85:
        s.append("rolled-off highlights")
    rgb_tint = lut[64] * 255 - lut[64].mean() * 255
    if np.abs(rgb_tint).max() > 3:
        c = ["red", "green", "blue"][int(np.argmax(rgb_tint))]
        s.append(f"{c}-tinted shadows from the channel curves")
    for zone, hk, sk in (("shadows", "SplitToningShadowHue", "SplitToningShadowSaturation"), ("midtones", "ColorGradeMidtoneHue", "ColorGradeMidtoneSat"),
                         ("highlights", "SplitToningHighlightHue", "SplitToningHighlightSaturation")):
        sat = p.grading.get(sk, 0) or 0
        if sat >= 5:
            strength = "strong" if sat >= 35 else ("noticeable" if sat >= 15 else "slight")
            s.append(f"{strength} {hue_word(p.grading.get(hk, 0) or 0)} tint in the {zone} (colour grade {sat:.0f})")
    for b in BANDS:
        h = p.hsl[b]
        v = h.get("saturation", 0) or 0
        if v <= -25:
            s.append(f"{b.lower()}s muted ({v:+.0f})")
        elif v >= 15:
            s.append(f"{b.lower()}s boosted ({v:+.0f})")
        v = h.get("luminance", 0) or 0
        if v <= -20:
            s.append(f"{b.lower()}s darkened ({v:+.0f})")
        elif v >= 18:
            s.append(f"{b.lower()}s lightened ({v:+.0f})")
        v = h.get("hue", 0) or 0
        if abs(v) >= 20:
            s.append(f"{b.lower()} hue shifted towards {'warmer/yellower' if v > 0 else 'cooler/redder'} ({v:+.0f})")
    g = p.grain.get("GrainAmount", 0) or 0
    if g >= 45:
        s.append(f"heavy grain ({g:.0f})")
    elif g >= 20:
        s.append(f"moderate grain ({g:.0f})")
    elif g == 0:
        s.append("no grain")
    if p.dehaze < 0:
        s.append(f"hazy, low-contrast veil (dehaze {p.dehaze:+.0f})")
    return s


def main() -> None:
    presets = load_presets(sys.argv[1] if len(sys.argv) > 1 else "data/lut/presets.json")
    out = {k: describe(p) for k, p in presets.items()}
    for k, v in out.items():
        print(f"{k}: " + "; ".join(v))
    json.dump(out, open("reports/lut/preset_descriptions.json", "w"), indent=1)


if __name__ == "__main__":
    main()
