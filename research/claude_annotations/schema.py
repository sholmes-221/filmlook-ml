"""Controlled vocabulary for structured photographic-style annotations, its JSON schema, and its vector form.

Every field is a closed list, so an annotation is a fixed-length vector and two annotators can be compared field
by field. Nothing in the schema can hold a photographer's name, a title or a date: the annotation describes what
is visible, and catalogue attribution stays the only ground truth for "who made this".
"""
from __future__ import annotations

import numpy as np

VERSION = "style-annotation-v1"

SUBJECT = ["person_single", "people_group", "crowd", "street_scene", "architecture_exterior", "interior", "landscape", "rural_scene",
           "industrial", "still_life", "object_detail", "sign_or_text", "vehicle", "animal", "plant_nature", "water_or_sky",
           "abstract_pattern", "other"]
SINGLE = {
    "subject_primary": SUBJECT,
    "setting": ["urban", "town", "rural", "wilderness", "indoor_domestic", "indoor_work", "indoor_public", "studio", "unclear"],
    "people_count": ["none", "one", "two", "small_group", "many"],
    "people_awareness": ["none_present", "posed_to_camera", "aware_unposed", "unaware_candid"],
    "framing": ["extreme_closeup", "closeup", "medium", "full_figure", "wide", "very_wide"],
    "vantage": ["eye_level", "low_angle", "high_angle", "overhead", "oblique"],
    "depth": ["flat", "shallow_layers", "deep_space"],
    "light_source": ["daylight_direct", "daylight_overcast", "window_light", "artificial", "flash", "mixed", "unclear"],
    "light_direction": ["front", "side", "back", "top", "diffuse", "unclear"],
    "light_quality": ["hard", "medium", "soft"],
    "key": ["low", "mid", "high"],
    "contrast": ["low", "medium", "high"],
    "tonal_treatment": ["neutral_bw", "warm_toned_bw", "cool_toned_bw", "colour_muted", "colour_natural", "colour_saturated", "hand_coloured", "faded"],
    "focus": ["sharp_overall", "selective_focus", "soft_overall", "motion_blur"],
    "texture": ["fine", "moderate_grain", "coarse_grain", "print_surface_artifacts"],
    "moment": ["posed_still", "candid_moment", "decisive_action", "empty_stillness", "constructed_arrangement"],
    "physical_condition": ["clean", "minor_wear", "heavy_damage_or_visible_border"],
    "confidence": ["low", "medium", "high"],
}
MULTI = {
    "subjects_secondary": SUBJECT,
    "composition": ["centered_subject", "off_center_subject", "symmetrical", "strong_diagonals", "frontal_flat", "layered_depth",
                    "frame_within_frame", "negative_space", "dense_all_over", "leading_lines", "horizon_dominant", "repeated_forms", "tight_crop"],
    "mood": ["formal", "austere", "tender", "melancholic", "tense", "playful", "monumental", "intimate", "detached", "lyrical", "gritty", "serene"],
}
# The eight narrative axes of the website's Photographer DNA (seed_data/axes.json), which pixels cannot measure.
ORDINAL = ["human_presence", "documentary_character", "abstraction", "humour", "melancholy", "intimacy", "urban_setting", "stillness"]
LEVELS = ["0", "1", "2", "3", "4"]
# Fields that describe the scan or print rather than the photograph; kept, but excluded from the style vector.
ARTEFACT_FIELDS = ("physical_condition", "confidence")


def json_schema() -> dict:
    props: dict = {k: {"type": "string", "enum": v} for k, v in SINGLE.items()}
    props.update({k: {"type": "array", "items": {"type": "string", "enum": v}} for k, v in MULTI.items()})
    props.update({k: {"type": "string", "enum": LEVELS} for k in ORDINAL})
    props["widely_reproduced"] = {"type": "boolean"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def validate(a: dict) -> list[str]:
    """Problems with one annotation (empty list = valid)."""
    bad = [f"{k}: {a.get(k)!r}" for k, v in SINGLE.items() if a.get(k) not in v]
    bad += [f"{k}: {a.get(k)!r}" for k, v in MULTI.items() if not isinstance(a.get(k), list) or any(x not in v for x in a[k])]
    bad += [f"{k}: {a.get(k)!r}" for k in ORDINAL if a.get(k) not in LEVELS]
    if not isinstance(a.get("widely_reproduced"), bool):
        bad.append("widely_reproduced")
    return bad


def feature_names(include_artefacts: bool = False) -> list[str]:
    names = [f"{k}={x}" for k, v in SINGLE.items() if include_artefacts or k not in ARTEFACT_FIELDS for x in v]
    names += [f"{k}:{x}" for k, v in MULTI.items() for x in v]
    return names + ORDINAL


def vectorise(a: dict, include_artefacts: bool = False) -> np.ndarray:
    """One-hot for single fields, multi-hot for lists, 0..1 for the ordinal axes."""
    out: list[float] = []
    for k, v in SINGLE.items():
        if include_artefacts or k not in ARTEFACT_FIELDS:
            out += [1.0 if a.get(k) == x else 0.0 for x in v]
    for k, v in MULTI.items():
        chosen = set(a.get(k) or [])
        out += [1.0 if x in chosen else 0.0 for x in v]
    out += [int(a.get(k, "2")) / 4.0 for k in ORDINAL]
    return np.asarray(out, dtype=np.float32)


SYSTEM_PROMPT = f"""You are annotating photographs for a research dataset about photographic style.

Purpose, stated plainly: these labels are structured visual descriptions (content categorisation and information
extraction). They are used to evaluate, and may be used to train, a small specialised photography-style retrieval
model. They are not used to build a general-purpose model.

For each photograph, return one JSON object in the required schema ({VERSION}). Rules:

1. Describe only what is visible in the image. Do not name, guess or hint at the photographer, the title, the
   collection, the place or the date, and do not let a guess about authorship influence any field.
2. Choose exactly one value for each single-choice field. Use "unclear" or the closest value when unsure, and set
   "confidence" to "low" when several fields were hard to decide.
3. List fields: "subjects_secondary" at most 3 values (do not repeat the primary subject), "composition" at most 4,
   "mood" at most 3. An empty list is allowed.
4. The eight narrative fields are levels "0" (absent) to "4" (dominant):
   human_presence: how much people and their traces carry the picture.
   documentary_character: observed record of the world ("4") versus staged or constructed image ("0").
   abstraction: forms, tones and patterns matter more than the recognisable subject.
   humour: visual wit, irony or absurdity.
   melancholy: sadness, solitude, loss.
   intimacy: closeness between camera and subject, emotional or physical.
   urban_setting: how urban the scene is.
   stillness: quiet, static, contemplative ("4") versus movement and action ("0").
5. "tonal_treatment" describes the image as presented: a sepia or albumen print is "warm_toned_bw".
6. "physical_condition" describes the scan or print (scratches, stains, a visible film border or mount), not the
   photograph's style. It is recorded so that archive artefacts can be separated from style.
7. "widely_reproduced" is true only if this exact picture looks like a very frequently reproduced photograph.
8. "framing" is the distance to the main subject; "vantage" the camera position; "depth" the spatial structure;
   "moment" the kind of instant (a posed sitting, a candid moment, an action at its peak, an empty quiet scene, an
   arrangement made for the camera).

Return only the JSON object."""

USER_TEXT = "Annotate this photograph."
