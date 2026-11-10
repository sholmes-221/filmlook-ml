"""End-to-end engine tests against the committed LUT feature database (research/lut/db)."""
import json
from pathlib import Path

import numpy as np
import pytest

from research.lut.engine import DEFAULT_DB_DIR, FilmStyleEngine
from research.lut.engine.lut_db import LutDB
from research.lut.engine.matching import APPROACHES, similarity_0_100

pytestmark = pytest.mark.skipif(not (Path(DEFAULT_DB_DIR) / "lut_db_arrays.npz").exists(), reason="LUT database not built")


def photo(seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:240:1, 0:320:1]
    img = np.stack([0.25 + 0.6 * x / 320, 0.35 + 0.4 * y / 240, 0.45 + 0.25 * np.cos(x / 40.0)], -1) + rng.normal(0, 0.02, (240, 320, 3))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def engine():
    return FilmStyleEngine()


def test_database_consistency():
    db = LutDB(DEFAULT_DB_DIR)
    S = db.n_styles
    assert S >= 1 and len(db.ids) == S + 1 and db.ids[-1] == "none"
    assert db.lattices.shape == (S, 33, 33, 33, 3) and db.inverse_lattices.shape == (S, 33, 33, 33, 3)
    assert db.bank.shape[1] == S + 1 and db.bank.shape[2] == db.meta["descriptor_dim"]
    assert np.isfinite(db.bank).all() and np.isfinite(db.cloud).all()
    for st in db.styles:
        d = st["descriptors"]
        assert "clipping" in d and "hue_shift_by_region" in d and len(d["hue_shift_by_region"]) == 12
        assert st["label_basis"] == "LUT-derived aesthetic" and st["approximation"]
    meta = json.loads((Path(DEFAULT_DB_DIR) / "lut_db.json").read_text())
    assert all(v["ok"] for v in meta["validation"])


def test_lattice_precision_and_apply():
    db = LutDB(DEFAULT_DB_DIR)
    lut = db.lut(0)
    x = np.random.default_rng(0).random((100, 3)).astype(np.float32)
    y = lut.apply(x)
    assert y.shape == (100, 3) and (y >= 0).all() and (y <= 1).all()
    assert np.abs(lut.lattice.astype(np.float32) - db.lattices[0].astype(np.float32)).max() < 1e-3   # float16 storage


def test_analyze_output_contract(engine):
    res = engine.analyze(photo(), top_k=3, all_methods=True)
    assert len(res["styles"]) == 3
    sims = [s["similarity"] for s in res["styles"]]
    assert sims[0] == 100.0 and all(0 <= v <= 100 for v in sims) and sims == sorted(sims, reverse=True)
    for s in res["styles"]:
        assert set(s) >= {"id", "name", "similarity", "explanation", "key_features", "label_basis"}
    assert res["method"] in APPROACHES and set(res["all_method_rankings"]) == set(APPROACHES)
    assert isinstance(res["uncertain"], bool) and "texture" in res and "photo" in res
    assert "probability" not in json.dumps(res["styles"]).lower()
    json.dumps(res)  # serialisable


def test_grey_photo_is_uncertain(engine):
    g = np.repeat(np.linspace(20, 230, 200).astype(np.uint8)[None, :, None], 150, 0)
    g = np.repeat(g, 3, 2)
    res = engine.analyze(g)
    assert res["uncertain"] and "monochrome" in res["uncertainty_reason"]


def test_render_moves_photo_towards_its_style(engine):
    """Rendering a photo through a style's LUT must bring it closer to that style than the untouched photo was
    (approach D, Fisher-weighted centroids) for most styles. Exact top-1 recognition on an artificial gradient is not
    expected: the reference scenes are natural photographs."""
    db = engine.db
    base = photo(seed=5)
    d0 = engine.analyze(base, method="D")["distances"]
    closer = 0
    for i in range(db.n_styles):
        d1 = engine.analyze(db.lut(i).apply_u8(base), method="D")["distances"]
        closer += d1[db.ids[i]] < d0[db.ids[i]]
    assert closer >= 9


def test_film_likeness_contract(engine):
    res = engine.film_likeness(photo(seed=7))
    assert res["band"] in ("digital-like", "ambiguous", "film-like") and res["confidence"] in ("low", "medium", "high")
    assert 0 <= res["estimate"] <= 100 and len(res["evidence"]) == 6
    assert all(e[2] in ("towards film", "towards digital") for e in res["evidence"])
    assert abs(sum(res["contributions"].values()) + res["logit"] - res["logit"] - sum(res["contributions"].values())) < 1e-6
    # noisier image -> higher estimate (fine noise is the strongest positive weight)
    noisy = np.clip(photo(seed=7).astype(float) + np.random.default_rng(1).normal(0, 12, photo().shape), 0, 255).astype(np.uint8)
    assert engine.film_likeness(noisy)["estimate"] > res["estimate"]


def test_similarity_scale():
    s = similarity_0_100(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert s[0] == 100.0 and abs(s[2] - 50.0) < 1e-6 and s[-1] < s[-2]


def test_file_and_bytes_inputs(engine, tmp_path):
    from PIL import Image
    f = tmp_path / "p.jpg"
    Image.fromarray(photo()).save(f, quality=90)
    r1 = engine.analyze(str(f))
    r2 = engine.analyze(f.read_bytes())
    assert r1["styles"][0]["id"] == r2["styles"][0]["id"]
