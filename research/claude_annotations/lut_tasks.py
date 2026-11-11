"""Claude as a second reader of the film-look descriptions the website shows.

The question is not "which film is this" (no model, and no person, can tell that from one picture). It is narrower
and testable: the website describes each LUT-derived look in one or two sentences generated from measured numbers
("Warm midtones with a yellow cast, green shadows..."). Given the untouched picture A and the same scene with a look
applied (B), can a careful reader pick the applied look's description out of four? Ground truth is the preset that
was applied, which the pipeline knows and the model does not. A description that readers cannot match to its own
look is either wrong or not specific enough, and pairs of looks that are confused for each other have descriptions
(or renderings) that do not differ visibly.
"""
from __future__ import annotations

import json
from pathlib import Path

VERSION = "lut-description-check-v1"
CATALOG = Path.home() / "misc" / "film-evaluation" / "filmlook" / "data" / "filmstyle" / "catalog_map.json"
# manifest preset name -> catalog style key
KEY = lambda preset: preset.lower().replace("'", "").replace(" ", "_")  # noqa: E731


def descriptions() -> dict[str, str]:
    styles = json.loads(CATALOG.read_text())["styles"]
    by_key = {k.replace("'", ""): v["description"]["en"] for k, v in styles.items()}
    import collections
    presets = collections.OrderedDict()
    for line in (Path(__file__).resolve().parents[2] / "data" / "lut" / "synth" / "manifest.jsonl").read_text().splitlines():
        preset = json.loads(line)["preset"]
        if preset != "none" and KEY(preset) in by_key:
            presets[preset] = by_key[KEY(preset)]
    return dict(presets)


SYSTEM_PROMPT = """You are checking written descriptions of colour grades for a research evaluation.

You will see two versions of the same photograph. Image 1 is the untouched version. Image 2 is the same scene after
a colour grade was applied (the framing, exposure and compression may differ slightly; ignore that). Then four
candidate descriptions of a grade are listed. Exactly one of them was written for the grade that was applied.

Compare the colour and tone of image 2 with image 1: colour cast in shadows, midtones and highlights, saturation,
black level, contrast. Choose the description that best matches the change you can see. Also report how visible the
change is. Answer in the required JSON schema only. These answers are used to evaluate the descriptions and may be
used to improve a small specialised film-look model; they are not used to build a general-purpose model."""


def user_text(options: list[str]) -> str:
    return "Candidate descriptions:\n" + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(options)) + "\nWhich one describes the change from image 1 to image 2?"


def json_schema() -> dict:
    props = {"choice": {"type": "string", "enum": ["1", "2", "3", "4"]}, "visible_change": {"type": "string", "enum": ["none", "subtle", "clear"]},
             "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def validate(a: dict) -> list[str]:
    s = json_schema()["properties"]
    return [f"{k}: {a.get(k)!r}" for k in s if a.get(k) not in s[k]["enum"]]
