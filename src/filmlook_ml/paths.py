from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FILMLOOK_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "configs"
DATA = ROOT / "data"
RAW, PROCESSED, LABELS = DATA / "raw", DATA / "processed", DATA / "labels"
EMBEDDINGS, SPLITS, MANIFESTS, QUARANTINE = DATA / "embeddings", DATA / "splits", DATA / "manifests", DATA / "quarantine"
CHECKPOINTS, REPORTS = ROOT / "checkpoints", ROOT / "reports"


def load_yaml(name: str) -> dict:
    import yaml
    return yaml.safe_load((CONFIG_DIR / name).read_text())
