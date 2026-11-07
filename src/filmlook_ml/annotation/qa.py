"""Quality assurance for categorization annotations (weak labels, not ground truth).

- Consistency checks compare annotator claims with deterministic measurements (filmlook_infer.features)
  to catch hallucinated attributes, e.g. "strong grain" on a noise-free image or "monochrome" on a colour image.
- Confidence thresholds and unknown labels route images to manual review.
- Near-duplicate detection uses perceptual hashes from the manifest.
- Review sampling: every flagged image plus a random share of unflagged ones.
"""
from __future__ import annotations

import random
from collections import Counter

from filmlook_infer.style import THRESHOLDS, style_tags
from filmlook_ml.annotation import taxonomy as T

MIN_CONFIDENCE = 0.5
NEAR_DUP_HAMMING = 6
# Flags that only inform. Every other flag sends the image to manual review.
INFO_FLAGS = {"style_none_of_the_above", "film_none_of_the_above", "uploader_stock_tag_disagrees"}


def measured_summary(f: dict | None) -> dict:
    if not f:
        return {}
    return {"tags": style_tags(f), "is_mono": bool(f["is_mono"]), "noise_std": round(float(f["noise_std"]), 5),
            "y_std": round(float(f["y_std"]), 4), "y_mean": round(float(f["y_mean"]), 4),
            "sat_mean": round(float(f["sat_mean"]), 4), "warmth": round(float(f["warmth"]), 4)}


def check(result, measured: dict, metadata=None) -> list[str]:
    """Flags for one CategorizationResult. ``measured`` comes from measured_summary()."""
    flags: list[str] = []
    ps, fa, at = result.photographer_style, result.film_aesthetic, result.attributes
    for name, call in (("style", ps), ("film", fa)):
        if call.primary == T.UNKNOWN:
            flags.append(f"{name}_unknown")
        elif call.primary == T.NONE:
            flags.append(f"{name}_none_of_the_above")
        elif call.confidence < MIN_CONFIDENCE:
            flags.append(f"{name}_low_confidence")
    mono_claim = at.color.saturation == "monochrome"
    if fa.primary in T.BW_STOCKS and not mono_claim:
        flags.append("bw_stock_on_colour_image")
    if fa.primary in T.FILM_STOCKS and fa.primary not in T.BW_STOCKS and mono_claim:
        flags.append("colour_stock_on_monochrome_image")
    if "not_a_photograph" in at.image_flags or "artwork_reproduction" in at.image_flags:
        flags.append("not_a_photograph")
    if measured:
        tags = set(measured["tags"])
        if mono_claim != measured["is_mono"]:
            flags.append("monochrome_disagrees_with_measurement")
        if not mono_claim and not measured["is_mono"]:
            t = at.color.temperature
            if (t == "warm" and "cool" in tags) or (t == "cool" and "warm" in tags):
                flags.append("temperature_disagrees_with_measurement")
            if (at.color.saturation == "high" and "muted" in tags) or (at.color.saturation == "low" and "vivid" in tags):
                flags.append("saturation_disagrees_with_measurement")
        g = at.texture.visible_grain
        if g in ("moderate", "strong") and measured["noise_std"] < THRESHOLDS["grain_none"]:
            flags.append("grain_claim_unsupported")
        if g == "none" and measured["noise_std"] > THRESHOLDS["grain_coarse"]:
            flags.append("grain_missed")
        c = at.tone.contrast
        if (c == "high" and "low_contrast" in tags) or (c == "low" and "high_contrast" in tags):
            flags.append("contrast_disagrees_with_measurement")
        k = at.lighting.key
        if (k == "high_key" and "low_key" in tags) or (k == "low_key" and "high_key" in tags):
            flags.append("key_disagrees_with_measurement")
    if metadata is not None and metadata.film_stock_metadata and fa.primary in T.FILM_STOCKS:
        tagged = {m.value for m in metadata.film_stock_metadata}
        if fa.primary not in tagged and not tagged & set(fa.alternatives):
            flags.append("uploader_stock_tag_disagrees")
    return flags


def needs_review(flags: list[str]) -> bool:
    return any(f not in INFO_FLAGS for f in flags)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def near_duplicates(records: list[dict], max_dist: int = NEAR_DUP_HAMMING) -> list[tuple[str, str, int, bool]]:
    """(id_a, id_b, distance, crosses_split) for originals whose perceptual hashes are within ``max_dist`` bits."""
    out = []
    rs = [r for r in records if r.get("phash")]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            d = hamming(rs[i]["phash"], rs[j]["phash"])
            if d <= max_dist:
                out.append((rs[i]["image_id"], rs[j]["image_id"], d, rs[i].get("split") != rs[j].get("split")))
    return out


def review_sample(anns: list, frac_unflagged: float = 0.1, seed: int = 0) -> list[str]:
    flagged = [a.image_id for a in anns if a.qa.needs_review]
    rest = [a.image_id for a in anns if not a.qa.needs_review]
    k = min(len(rest), max(1, round(frac_unflagged * len(rest)))) if rest else 0
    return sorted(flagged) + sorted(random.Random(seed).sample(rest, k))


def distributions(anns: list) -> dict:
    def c(get):
        return dict(Counter(get(a) for a in anns).most_common())

    def multi(get):
        return dict(Counter(x for a in anns for x in get(a)).most_common())
    return {
        "photographer_primary": c(lambda a: a.photographer_style.primary),
        "photographer_any": multi(lambda a: [a.photographer_style.primary, *a.photographer_style.alternatives]),
        "film_primary": c(lambda a: a.film_aesthetic.primary),
        "film_any": multi(lambda a: [a.film_aesthetic.primary, *a.film_aesthetic.alternatives]),
        "genre": multi(lambda a: a.attributes.genre),
        "aesthetic": multi(lambda a: a.attributes.overall_aesthetic),
        "temperature": c(lambda a: a.attributes.color.temperature),
        "contrast": c(lambda a: a.attributes.tone.contrast),
        "visible_grain": c(lambda a: a.attributes.texture.visible_grain),
        "qa_flags": multi(lambda a: a.qa.flags),
        "needs_review": sum(a.qa.needs_review for a in anns),
        "training_eligible": sum(a.training_eligible for a in anns),
    }
