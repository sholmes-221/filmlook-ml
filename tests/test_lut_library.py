"""LUT-library ingestion: HaldCLUT reader, manifest filters, stock/line mapping (no library files needed)."""
import numpy as np
import pytest
from PIL import Image

from research.lut.engine.lut3d import LUT3D
from research.lut.library_db import COLOUR_TYPES, ingest
from research.lut.real_eval_library import FAMILY, STOCK_TO_LINE


def write_hald(path, level=4, fn=lambda rgb: rgb):
    """Identity-or-transformed HaldCLUT image of the given level (n = level^2 steps, red fastest)."""
    n = level * level
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")           # red varies fastest in memory order
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = np.clip(fn(rgb), 0, 1)
    side = level ** 3
    Image.fromarray((out.reshape(side, side, 3) * 255 + 0.5).astype(np.uint8)).save(path)


def test_hald_identity_and_ordering(tmp_path):
    f = tmp_path / "identity.png"
    write_hald(f)
    lut = LUT3D.from_hald(f)
    assert lut.size == 16 and lut.validate()["ok"] and lut.validate()["identity_rms"] < 0.01
    # a channel-specific transform must land on the right axis
    f2 = tmp_path / "redgain.png"
    write_hald(f2, fn=lambda rgb: rgb * np.array([0.5, 1.0, 1.0], np.float32))
    lut2 = LUT3D.from_hald(f2)
    x = np.array([[1.0, 0.2, 0.6]], np.float32)
    y = lut2.apply(x)[0]
    assert abs(y[0] - 0.5) < 0.02 and abs(y[1] - 0.2) < 0.02 and abs(y[2] - 0.6) < 0.02
    assert lut2.resample(9).size == 9


def test_hald_rejects_non_cube_width(tmp_path):
    f = tmp_path / "bad.png"
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(f)
    with pytest.raises(ValueError):
        LUT3D.from_hald(f)


def test_ingest_filters(tmp_path):
    f = tmp_path / "ok.png"
    write_hald(f)
    base = {"path": str(f), "format": "haldclut_png", "input_color_space": "sRGB (display-referred, 8-bit)", "validation": {"ok": True},
            "name": "X", "source_slug": "s", "license": "CC-BY-SA-4.0", "film_stock": "kodak_portra_400", "film_type": "colour_negative"}
    recs = [{**base, "id": "s/a", "role": "film_look"},
            {**base, "id": "s/b", "role": "creative_look"},
            {**base, "id": "s/c", "role": "film_look", "format": "rawtherapee_pp3"},
            {**base, "id": "s/d", "role": "film_look", "input_color_space": "DaVinci Intermediate / DWG"},
            {**base, "id": "s/e", "role": "film_look", "validation": {"ok": False}},
            {**base, "id": "s/f", "role": "film_look", "license": "CC0-1.0"},
            {**base, "id": "s/a", "role": "film_look"}]
    styles, luts, skipped = ingest(recs)
    assert [s["id"] for s in styles] == ["s__a", "s__f"] and all(l.size == 33 for l in luts)
    reasons = {s["id"]: s["reason"] for s in skipped}
    assert "role" in reasons["s/b"] and "pp3" in reasons["s/c"] and "sRGB" in reasons["s/d"] and "validation" in reasons["s/e"]
    assert sum(s["reason"] == "duplicate id" for s in skipped) == 1
    free_styles, _, _ = ingest(recs, only_free=True)
    assert [s["id"] for s in free_styles] == ["s__f"]
    assert styles[0]["label_basis"] == "LUT-derived aesthetic" and "approximation" in styles[0]


def test_mappings_consistent():
    assert set(STOCK_TO_LINE.values()) == set(FAMILY)
    assert "colour_negative" in COLOUR_TYPES and "bw_negative" not in COLOUR_TYPES
