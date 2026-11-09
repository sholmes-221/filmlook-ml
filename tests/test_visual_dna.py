"""Visual DNA knowledge graph: validation rules, completeness, and the counterexamples from the owner's brief."""
from __future__ import annotations

import copy

import pytest

from filmlook_ml.visualdna import build as B
from filmlook_ml.visualdna import dna


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    db = tmp_path_factory.mktemp("vdna") / "vd.sqlite"
    B.build(db)
    dna.knowledge.cache_clear()
    return dna.knowledge(db)


def test_every_profile_validates_and_cites_sources():
    vocab = B.load_vocabulary()
    for p in B.load_profiles():
        assert B.validate(p, vocab) == [], p["id"]


def test_signature_motif_needs_two_sources_and_tags_must_be_in_vocabulary():
    vocab = B.load_vocabulary()
    p = copy.deepcopy(next(p for p in B.load_profiles() if p["id"] == "saul_leiter"))
    p["motifs"][0]["src"] = ["SLF_B"]
    p["motifs"][1]["tags"] = ["not-a-tag"]
    errors = B.validate(p, vocab)
    assert any("fewer than two sources" in e for e in errors) and any("not in vocabulary" in e for e in errors)


def test_incomplete_profiles_are_reported_not_filled_in():
    comp = {p["id"]: B.completeness(p) for p in B.load_profiles()}
    assert comp["saul_leiter"]["status"] == "complete"
    assert "major_photobooks_researched" in comp["lewis_hine"]["missing"]  # No photobook verified: stays missing.


def _top(kb, det, meas):
    return sorted(((dna.knowledge_score(p, det, meas, kb), p) for p in kb["photographers"]), key=lambda x: -x[0]["total"])[0]


def test_crow_with_matching_rendering_points_to_ravens(kb):
    k, p = _top(kb, {"crow": 0.6, "bird": 0.7, "silhouette": 0.4}, {"monochrome", "dark tones"})
    e = dna.explain(p, k, dna.strength_label({"first_in": ["dna"]}, k), ["crow", "monochrome"], kb)
    assert p == "masahisa_fukase" and e["label"] == "subject_and_style" and "《鸦》系列" in e["zh"]


def test_cartoon_crow_is_a_subject_association_only(kb):
    k = dna.knowledge_score("masahisa_fukase", {"crow": 0.5, "bird": 0.6}, {"colour", "saturated colour"}, kb)
    label = dna.strength_label({"first_in": ["dna"]}, k)
    e = dna.explain("masahisa_fukase", k, label, ["crow"], kb)
    assert label == "subject_only" and "只是题材上的联想" in e["zh"] and "subject association only" in e["en"]


def test_pastel_still_life_without_watermelon_can_still_match_kawauchi_on_style(kb):
    k, p = _top(kb, {"backlight": 0.3}, {"colour", "bright airy exposure"})
    assert p == "rinko_kawauchi" and dna.strength_label({"first_in": ["dna"]}, k) == "style_only"


def test_generic_street_or_pedestrians_never_produce_a_match(kb):
    k, p = _top(kb, {"street": 0.7, "crowd": 0.5, "candid gesture": 0.05}, {"colour"})
    assert dna.strength_label({"first_in": ["dna"]}, k) == "no_match"


def test_weak_associations_never_score(kb):
    # Kawauchi's insects and birds are weak (title-only evidence): alone they must not create a match.
    k = dna.knowledge_score("rinko_kawauchi", {"insect": 0.9, "bird": 0.9}, set(), kb)
    assert k["subject"] == 0


def test_one_signal_is_never_strong():
    r = dna.fuse(["a", "b"], {"dinov2": [0.9, 0.1], "semantic": None, "dna": None}, {"dinov2": 0.1, "semantic": 0.1, "dna": 0.1})
    assert r["signals_used"] == ["dinov2"] and r["weights"] == {"dinov2": 1.0}
    k = {"subject": 0.0, "style": 0.0, "total": 0.0}
    assert dna.strength_label(r["ranked"][0], k) != "strong"
    r2 = dna.fuse(["a", "b"], {"dinov2": [0.9, 0.1], "semantic": [0.8, 0.2], "dna": [0.1, 0.3]}, {"dinov2": 0.1, "semantic": 0.1, "dna": 0.1})
    assert r2["ranked"][0]["photographer_id"] == "a" and dna.strength_label(r2["ranked"][0], {"subject": 1, "style": 1, "total": 1}) == "strong"
