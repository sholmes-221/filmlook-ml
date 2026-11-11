"""Shared helpers for C08: every LUT we may analyse (owner presets + licensed library), colour maths, CIEDE2000."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.lut.engine.lut3d import LUT3D
from research.lut.forward_model import load_presets, srgb_to_lab

OUT = Path("data/c08")
COLOUR_TYPES = {"colour_negative", "colour_reversal", "instant", "instant_colour", "camera_simulation", "colour_negative_cine", "colour_negative_special"}


def lab(rgb01: np.ndarray) -> np.ndarray:
    return np.asarray(srgb_to_lab(np.clip(np.asarray(rgb01, dtype=np.float64), 0, 1)))


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005), vectorised over the leading axes."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, float), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, float), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = 1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm)) + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def _library_records() -> list[dict]:
    return json.loads(Path("research/lut_library/manifest.json").read_text())["luts"]


def all_luts(include_unanalysable: bool = False) -> list[dict]:
    """Owner presets (13 Lightroom looks, the live engine's classes) and every library record. Each item: metadata plus
    `lut` (LUT3D on sRGB) or `lut=None` with `skip` when it cannot be read as an sRGB-in/sRGB-out colour transform."""
    items = []
    for name, p in load_presets().items():
        items.append({"id": f"preset__{name}", "name": name, "source": "owner_presets", "role": "film_look", "format": "lightroom_xmp",
                      "license": "owner's own presets (research and production use by the owner)", "film_stock": None, "film_type": "colour_negative",
                      "original_size": None, "domain": [[0, 0, 0], [1, 1, 1]], "lut": LUT3D(p.lattice(33), name=name)})
    for r in _library_records():
        item = {"id": r["id"].replace("/", "__"), "name": r["name"], "source": r["source_slug"], "role": r.get("role"), "format": r.get("format"),
                "license": r.get("license"), "film_stock": r.get("film_stock"), "film_type": r.get("film_type"), "original_size": r.get("lut_size"),
                "input_color_space": r.get("input_color_space"), "variant_tokens": r.get("variant_tokens") or [], "push_pull": r.get("push_pull_stops"),
                "path": r["path"], "lut": None, "skip": None}
        if r.get("format") == "rawtherapee_pp3":
            item["skip"] = "RawTherapee profile, not a LUT"
        elif not str(r.get("input_color_space", "")).lower().startswith("srgb"):
            item["skip"] = "input is not sRGB (log / camera space): sRGB patch analysis would be meaningless"
        elif not r.get("validation", {}).get("ok", False):
            item["skip"] = "library validation failed"
        else:
            try:
                lut = LUT3D.from_hald(r["path"]) if r["format"].startswith("haldclut") else LUT3D.from_cube(r["path"])
                item["domain"] = [lut.domain_min.tolist(), lut.domain_max.tolist()]
                item["lut"] = lut.resample(33) if lut.size != 33 else lut
            except Exception as exc:  # noqa: BLE001
                item["skip"] = f"parse error: {exc}"
        if item["lut"] is not None or include_unanalysable:
            items.append(item)
    return items


def lut_kind(item: dict) -> str:
    role = item.get("role")
    if role in ("identity", "negative_inversion") or item.get("skip", "") and "not sRGB" in (item.get("skip") or ""):
        return "technical"
    if role == "creative_look":
        return "creative"
    if item.get("film_type") in COLOUR_TYPES:
        return "film_emulation_colour"
    if item.get("film_type"):
        return "film_emulation_bw"
    return "film_emulation"
