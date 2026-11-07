"""Quarantined images (configs/quarantine.yaml): files stay on disk; annotation, training and evaluation skip them."""
from __future__ import annotations

from pathlib import Path

import yaml

from filmlook_ml.paths import CONFIG_DIR


def load_quarantine(path: Path | None = None) -> dict[str, dict]:
    p = Path(path or CONFIG_DIR / "quarantine.yaml")
    if not p.exists():
        return {}
    cfg = yaml.safe_load(p.read_text()) or {}
    return {e["image_id"]: e for e in cfg.get("images", [])}


def is_quarantined(record: dict, quarantine: dict[str, dict]) -> bool:
    """True for a quarantined image, and for every member of its family when ``family_excluded`` is set."""
    if record["image_id"] in quarantine:
        return True
    fam = record.get("family_id")
    return bool(fam) and any(e.get("family_excluded") and q.startswith(f"{fam}_") for q, e in quarantine.items())


def filter_manifest(manifest: list[dict], quarantine: dict[str, dict] | None = None) -> list[dict]:
    q = load_quarantine() if quarantine is None else quarantine
    return [r for r in manifest if not is_quarantined(r, q)]
