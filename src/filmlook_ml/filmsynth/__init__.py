"""Film-inspired synthetic augmentation (versioned presets in configs/film_synth_v1.yaml).

Synthetic variants are labelled with the parameters actually applied. They are never real film scans."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from filmlook_ml.paths import CONFIG_DIR


class SyntheticAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_image_id: str
    variant_id: str
    augmentation_family: str
    augmentation_version: str
    parameters: dict[str, float | bool]           # every engine parameter as applied (0 / False = off)
    nuisance: dict[str, float | int | str]        # jpeg_quality, exposure_ev, crop_fraction, resample
    seed: int
    label_source: str = "deterministic_synthetic_augmentation"
    is_real_film_scan: bool = False
    approximate_references: list[str] = Field(default_factory=list)   # database aesthetic ids, orientation only
    split: str
    group_id: str
    source_license: str | None = None
    source_dataset: str | None = None
    source_attribution: str | None = None
    path: str
    sha256: str


def load_presets(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "film_synth_v1.yaml").read_text())


def variant_seed(source_id: str, family: str, k: int, version: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{version}|{source_id}|{family}|{k}".encode()).digest()[:4], "little")


def sample_params(family_cfg: dict, rng: np.random.Generator) -> dict:
    from filmlook_ml.filmsynth.engine import PARAM_NAMES
    out: dict = {k: 0.0 for k in PARAM_NAMES}
    out["monochrome"] = False
    for k, v in (family_cfg.get("params") or {}).items():
        if isinstance(v, list):
            out[k] = round(float(rng.uniform(v[0], v[1])), 4)
        else:
            out[k] = v
    return out


def sample_nuisance(cfg: dict, rng: np.random.Generator) -> dict:
    n = cfg["nuisance"]
    return {"jpeg_quality": int(rng.integers(n["jpeg_quality"][0], n["jpeg_quality"][1] + 1)),
            "exposure_ev": round(float(rng.uniform(*n["exposure_ev"])), 4),
            "crop_fraction": round(float(rng.uniform(*n["crop_fraction"])), 4),
            "resample": str(rng.choice(n["resample"]))}
