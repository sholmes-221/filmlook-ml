"""Database-driven taxonomy, quarantine, categorization-3.0 schema, and the DINOv2 baseline interface."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.schema_v3 import CategorizationV3
from filmlook_ml.datasets.quarantine import filter_manifest, is_quarantined, load_quarantine
from filmlook_ml.taxonomy_db import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
HAS_TAX = (ROOT / "taxonomy" / "photographers.v1.json").exists()


# ------------------------------------------------------------------ taxonomy (database-driven)
@pytest.mark.skipif(not HAS_TAX, reason="taxonomy files not extracted")
def test_taxonomy_loads_with_stable_unique_ids_and_provenance():
    t = load_taxonomy()
    s = t.summary()
    assert s["photographers"] == 139 and s["aesthetic_directions"] == 67
    assert s["stock_entity_types"]["film_stock"] >= 300
    assert t.source["source_git_commit"] and not t.source["source_dirty"]
    for p in t.photographers.values():
        assert p["names"]["en"] and p["names"]["zh"] and "provenance" in p
        assert "ja" not in p["names"]                                  # not in the source: never invented
    assert all(a["reference_stock_id"] in t.stocks for a in t.aesthetics.values())


@pytest.mark.skipif(not HAS_TAX, reason="taxonomy files not extracted")
def test_taxonomy_lookup_aliases_and_languages():
    t = load_taxonomy()
    assert t.photographer_id("Saul Leiter") == t.photographer_id("索尔·雷特") == "saul_leiter"
    assert t.stock_id("Portra-like") == t.stock_id("Kodak Portra 400")
    assert t.stock_id("Fujifilm Superia X-TRA 400") == t.stock_id("Fuji Superia X-TRA 400")
    assert t.stock_id("Kodak Portra 160") != t.stock_id("Kodak Portra 400")     # distinct stocks never merged
    assert t.display("photographer", "daido_moriyama", "zh") == "森山大道"
    # every category the previous hard-coded taxonomies used exists in the database
    for name in T.PHOTOGRAPHERS:
        assert t.photographer_id(name), name


@pytest.mark.skipif(not HAS_TAX, reason="taxonomy files not extracted")
def test_shortlist_gates_on_palette():
    t = load_taxonomy()
    mono, colour = set(t.shortlist_photographers(True)), set(t.shortlist_photographers(False))
    assert all(t.photographers[p]["palette"] != "colour" for p in mono)
    assert all(t.photographers[p]["palette"] != "monochrome" for p in colour)
    assert all(t.aesthetics[a]["monochrome"] for a in t.shortlist_aesthetics(True))


def test_extractor_norm_key_merges_spelling_not_distinct_stocks():
    spec = importlib.util.spec_from_file_location("extract_taxonomy", ROOT / "scripts" / "extract_taxonomy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.norm_key("Fuji Velvia 50") == m.norm_key("Fujifilm Velvia 50") == m.norm_key("Velvia-like".replace(
        "Velvia", "Fujifilm Velvia 50"))
    assert m.norm_key("Ilford HP5 Plus") == m.norm_key("Ilford HP5")
    assert m.norm_key("Kodak Portra 160") != m.norm_key("Kodak Portra 400")


# ------------------------------------------------------------------ quarantine
def test_quarantine_excludes_image_and_family_but_keeps_file():
    q = load_quarantine()
    assert "f993c25b62208870_original" in q
    man = [{"image_id": "f993c25b62208870_original", "family_id": "f993c25b62208870"},
           {"image_id": "f993c25b62208870_film_emulation", "family_id": "f993c25b62208870"},
           {"image_id": "aaaa_original", "family_id": "aaaa"}]
    assert [r["image_id"] for r in filter_manifest(man, q)] == ["aaaa_original"]
    assert is_quarantined(man[1], q)
    p = ROOT / "data" / "raw" / "commoncatalog" / "cc_9733492355.jpg"
    if p.parent.exists():
        assert p.exists()                                              # original file preserved


# ------------------------------------------------------------------ categorization-3.0
V3 = {"photographer": {"analyzable": True, "closest_photographer": "Saul Leiter",
                       "top_3_photographers": [{"name": "Saul Leiter", "similarity": 0.4},
                                               {"name": "Rinko Kawauchi", "similarity": 0.3},
                                               {"name": "Fan Ho", "similarity": 0.1}],
                       "style_match_supported": False, "none_of_the_above": True, "style_evidence": ["generic snapshot"]},
      "film": {"analyzable": True, "top_3_film_stocks": [{"name": "Kodak Portra 400", "similarity": 0.5},
                                                         {"name": "Kodak Gold 200", "similarity": 0.4},
                                                         {"name": "Fujifilm C200", "similarity": 0.2}],
               "stock_match_supported": False, "aesthetic_category": "classic_color_negative",
               "apparent_capture": "film_look", "film_evidence": ["soft contrast"]},
      "attributes": {"color": {"temperature": "warm", "saturation": "low", "dominant_colors": ["red"], "color_cast": "none"},
                     "lighting": {"source": "natural", "quality": "soft", "key": "balanced", "direction": "diffuse"},
                     "tone": {"contrast": "low", "highlight_rolloff": "soft", "shadow_depth": "lifted"},
                     "texture": {"visible_grain": "subtle", "grain_appearance": "fine", "sharpness": "moderate"},
                     "genre": ["street"], "image_flags": ["misoriented"]}}


def _v3(**patch):
    d = json.loads(json.dumps(V3))
    for path, val in patch.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    return d


def test_v3_ranking_is_kept_apart_from_match_support():
    r = CategorizationV3(**V3)
    assert r.photographer.closest_photographer == "Saul Leiter" and not r.photographer.style_match_supported
    for bad in (_v3(**{"photographer.style_match_supported": True}),           # both supported and none
                _v3(**{"photographer.none_of_the_above": False}),              # neither
                _v3(**{"photographer.closest_photographer": "Fan Ho"}),        # closest != first ranked
                _v3(**{"photographer.top_3_photographers": V3["photographer"]["top_3_photographers"][::-1]}),
                _v3(**{"film.aesthetic_category": "bw_negative"}),             # B&W category on colour image
                _v3(**{"film.stock_match_supported": True, "film.aesthetic_category": "clean_digital"})):
        with pytest.raises(ValidationError):
            CategorizationV3(**bad)


# ------------------------------------------------------------------ baseline predictor
def test_baseline_predictor_structured_output(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from filmlook_ml import dinov2_baseline as B
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 768))
    heads = {}
    for name, ncls in (("temperature", 3), ("grain", 2)):
        y = np.array([f"c{i % ncls}" for i in range(60)])
        sc = StandardScaler().fit(X)
        heads[name] = (sc, LogisticRegression(max_iter=500).fit(sc.transform(X), y), sorted(set(y)))
    B.save_heads(tmp_path, heads, X.mean(0), {"baseline_version": "test"})
    out = B.BaselinePredictor(tmp_path).predict_embedding(X[0])
    assert set(out["visual_attributes"]) == {"temperature", "grain"}
    assert out["photographer_style"]["status"] == "not_available" and out["film_aesthetic"]["status"] == "not_available"
    sc = out["visual_attributes"]["grain"]["scores"]
    assert abs(sum(sc.values()) - 1) < 1e-3 and len(sc) == 2


def test_attribute_labels_are_deterministic():
    from filmlook_ml.dinov2_baseline import attribute_labels
    f = {"is_mono": 0, "warmth": 0.1, "sat_mean": 0.5, "y_std": 0.1, "y_mean": 0.7, "noise_std": 0.001, "lap_var": 0.01}
    assert attribute_labels(f) == {"temperature": "warm", "saturation": "high", "contrast": "low", "key": "high_key",
                                   "grain": "none", "sharpness": "crisp"}


@pytest.mark.skipif(not HAS_TAX, reason="taxonomy files not extracted")
def test_v3_validates_against_database_ids_via_context():
    from filmlook_ml.taxonomy_db import candidates_block, validation_context
    t = load_taxonomy()
    d = _v3(**{"photographer.closest_photographer": "saul_leiter",
               "photographer.top_3_photographers": [{"name": "saul_leiter", "similarity": 0.4},
                                                    {"name": "takashi_homma", "similarity": 0.3},
                                                    {"name": "luigi_ghirri", "similarity": 0.2}],
               "film.top_3_film_stocks": [{"name": "portra_like", "similarity": 0.5},
                                          {"name": "gold_like", "similarity": 0.4},
                                          {"name": "pro_400h_like", "similarity": 0.3}]})
    ctx = validation_context(t)
    CategorizationV3.model_validate(d, context=ctx)
    with pytest.raises(ValidationError):                                # names outside the database are rejected
        CategorizationV3.model_validate(_v3(**{"photographer.closest_photographer": "nobody",
                                               "photographer.top_3_photographers": [
                                                   {"name": "nobody", "similarity": 0.4},
                                                   {"name": "saul_leiter", "similarity": 0.3},
                                                   {"name": "fan_ho", "similarity": 0.2}]}), context=ctx)
    short = validation_context(t, ["saul_leiter", "fan_ho", "alex_webb"], ["portra_like", "gold_like", "hp5_like"])
    with pytest.raises(ValidationError):                                # must stay inside the shown shortlist
        CategorizationV3.model_validate(d, context=short)
    block = candidates_block(t, ["saul_leiter"], ["portra_like"])
    assert "saul_leiter" in block and "portra_like" in block and "fan_ho" not in block
