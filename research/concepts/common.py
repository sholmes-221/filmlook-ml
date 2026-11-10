"""Shared paths, etiquette and label helpers for the badge-concept study."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "concepts"
META = DATA / "oi_meta"
IMAGES = DATA / "images"          # <=384 px JPEG thumbnails made locally from the CVDF copy; never committed
DIST = ROOT / "dist" / "concepts"
REPORTS = ROOT / "reports" / "concepts"
USER_AGENT = "FilmAnalysisResearch/1.0 (badge concept study; small subset)"   # generic, no personal contact
CVDF = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"

# Classes whose *positive* label on an image makes it an unreliable negative for a head (a parent, a near-synonym or a
# usual co-occurrence the raters may have verified instead). Such images are dropped from that head's negatives only.
IGNORE: dict[str, list[str]] = {
    "night": ["Darkness", "Evening", "Dusk", "Neon", "Neon sign", "Street light", "Moon", "Full moon", "Star", "Milky way", "Lamp"],
    "blue_hour": ["Night", "Evening", "Sunset", "Sunrise", "Afterglow", "Darkness"],
    "golden_hour": ["Dusk", "Evening", "Sunlight", "Afterglow"],
    "street": ["Road", "City", "Urban area", "Pedestrian", "Neighbourhood", "Town", "Lane"],
    "bird": ["Animal", "Wildlife"],
    "small_animal": ["Animal", "Mammal", "Pet", "Bird"],
    "flowers": ["Plant", "Garden", "Flowering plant", "Petal", "Botany", "Lotus family"],
    "boat": ["Watercraft", "Ship", "Harbor", "Marina", "Vehicle"],
    "duck": ["Ducks, geese and swans", "Water bird"],
    "moon": ["Astronomical object", "Celestial event"],
    "starry": ["Astronomical object", "Night", "Nebula", "Galaxy"],
    "ice": ["Snow", "Frost", "Winter", "Freezing"],
    "fog": ["Atmospheric phenomenon", "Cloud"],
    "wave": ["Ocean", "Sea"],
    "dock": ["Harbor", "Marina"],
    "industrial": ["Industry", "Building"],
    "construction": ["Crane (Machine)"],
    "suburb": ["Neighbourhood", "House", "Home"],
    "fishing": ["Fish", "Fishing vessel"],
    "maple": ["Autumn", "Leaf", "Deciduous"],
    "wheat": ["Grain", "Cereal", "Field", "Crop"],
    "lotus_pond": ["Water lily (Plant)", "Aquatic plant"],
    "pink": ["Magenta", "Purple", "Red", "Peach"],
    "corridor": ["Room", "Building"],
    "cabin": ["Cottage", "House"],
    "diner": ["Restaurant", "Fast food"],
}


def class_maps() -> tuple[dict[str, str], dict[str, str]]:
    cd = pd.read_csv(META / "oidv7-class-descriptions.csv")
    return dict(zip(cd.DisplayName, cd.LabelName)), dict(zip(cd.LabelName, cd.DisplayName))


def related_mids() -> set[str]:
    n2m, _ = class_maps()
    return {n2m[n] for names in IGNORE.values() for n in names if n in n2m}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []
