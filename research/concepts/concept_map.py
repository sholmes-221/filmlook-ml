"""Badge concepts (the website's `_WORDS` keys in filmlook/collection/context.py) mapped to source labels.

Each supported-candidate concept lists the Open Images V7 class names whose *human-verified* image-level label
(Confidence 1) makes an image a positive. A verified 0 for any of those classes makes it a verified hard negative.
Concepts with no adequate legal source label, or that are inherently ambiguous as a whole-image property, are listed in
UNSUPPORTED with the reason; they are never guessed.

`key` is the exact `_WORDS` key (the badge constant's string value in filmlook/collection/badges.py or lessons.py).
Several heads may feed one key (e.g. bird and small-animal both feed "birds").
"""
from __future__ import annotations

# head id -> (exact _WORDS key, [Open Images class display names])
CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "night": ("night", ["Night"]),
    "blue_hour": ("blue_hour", ["Dusk"]),
    "golden_hour": ("golden_hour", ["Sunset", "Sunrise", "Afterglow"]),
    "street": ("street", ["Street", "Alley", "Sidewalk"]),
    "bird": ("birds", ["Bird", "Duck", "Mallard", "Owl", "Parrot", "Heron", "Egret"]),
    "small_animal": ("birds", ["Cat", "Dog", "Squirrel", "Wildlife"]),
    "school_uniform": ("sailor", ["School uniform"]),
    "gas_station": ("gas_station", ["Filling station", "Gas pump"]),
    "car_window": ("car_window", ["Rear-view mirror", "Automotive side-view mirror", "Windshield"]),
    "suburb": ("suburb", ["Suburb", "Residential area"]),
    "industrial": ("industrial", ["Warehouse", "Parking lot", "Factory"]),
    "pumpkin": ("pumpkin", ["Pumpkin"]),
    "desert": ("desert", ["Desert", "Dune", "Erg"]),
    "power_plant": ("power_plant", ["Power station", "Cooling tower"]),
    "construction": ("construction", ["Construction", "Scaffolding", "Construction worker"]),
    "motorcycle": ("motorcycle", ["Motorcycle"]),
    "starry": ("starry", ["Star", "Milky way"]),
    "water_lily": ("water_lily", ["Water lily (Plant)", "Fragrant white water lily", "Nymphaea lotus"]),
    "sunflower": ("sunflower", ["Sunflower", "Common sunflower"]),
    "diner": ("diner", ["Diner", "Convenience store", "Fast food restaurant"]),
    "wave": ("wave", ["Wind wave", "Surfing"]),
    "fog": ("fog", ["Fog", "Mist", "Haze"]),
    "pink": ("pink", ["Pink"]),
    "corridor": ("corridor", ["Hall", "Aisle"]),
    "umbrella": ("umbrella", ["Umbrella"]),
    "scooter": ("scooter", ["Scooter", "Vespa", "Moped"]),
    "moon": ("moon", ["Moon", "Full moon"]),
    "fishing": ("fishing", ["Fishing", "Recreational fishing", "Angling", "Fisherman", "Fly fishing", "Fishing rod"]),
    "maple": ("maple", ["Maple", "Maple leaf"]),
    "willow": ("willow", ["Willow", "Weeping willow"]),
    "dragonfly": ("dragonfly", ["Dragonfly", "Dragonflies and damseflies", "Damselfly"]),
    "duck": ("duck", ["Duck", "Mallard"]),
    "boat": ("boat", ["Boat"]),
    "flowers": ("flowers", ["Flower", "Blossom", "Rose", "Tulip", "Peony", "Sunflower", "Lily", "Cherry blossom", "Sacred lotus"]),
    "oranges": ("oranges", ["Mandarin orange", "Tangerine", "Orange (fruit)", "Clementine"]),
    "tunnel": ("tunnel", ["Tunnel"]),
    "dock": ("dock", ["Pier", "Dock"]),
    "cabin": ("cabin", ["Log cabin", "Hut", "Shack"]),
    "windmill": ("windmill", ["Windmill", "Wind turbine"]),
    "wheat": ("wheat", ["Wheat", "Barley", "Rye"]),
    "ice": ("ice", ["Ice", "Icicle"]),
    "fox": ("fox", ["Fox", "Red fox"]),
    "rickshaw": ("rickshaw", ["Rickshaw"]),
    "lotus_pond": ("lotus_pond", ["Sacred lotus", "Lotus family", "Lotus (Plant)"]),
    "spider_web": ("spider_web", ["Spider web"]),
    "beetle": ("beetle", ["Beetle", "Ladybug", "Stag beetle", "Rhinoceros beetle", "Ground beetle", "Japanese beetle"]),
    "reflection": ("reflection", ["Reflection"]),
}

UNSUPPORTED: dict[str, str] = {
    "window_light": "no source label for light falling through a window (Window + Sunlight is not the same thing)",
    "grid": "pattern/composition concept; no source label that means a grid composition",
    "round_glasses": "source labels have Glasses but not the frame shape",
    "qipao": "no Open Images class for qipao/cheongsam; no licensed labelled source found in budget",
    "tatami": "no Open Images class; no licensed labelled source found in budget",
    "rain_alley": "compound scene (oil-paper umbrella / rainy alley); no source label",
    "peach_blossom": "source labels cannot separate peach from cherry/plum blossom",
    "two_trees": "a count of trees, not a class; no source label",
    "frame": "frame-within-frame is a composition judgement; no source label",
}
