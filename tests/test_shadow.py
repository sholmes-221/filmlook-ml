"""Shadow-mode record: separate fields for aesthetics, authenticity, attributes and embedding; no field doubles as another."""
import json
from pathlib import Path

import pytest

from filmlook_infer.shadow import BADGE_RULES, ShadowAnalyzer

ROOT = Path(__file__).resolve().parents[1]
FILM = ROOT / "checkpoints" / "best"
AUTH = ROOT / "checkpoints" / "student_sup"
STYLE = ROOT / "checkpoints" / "student_style_fs"
SAMPLE = ROOT / "tests" / "assets" / "sample_synthetic.jpg"


@pytest.mark.skipif(not (FILM / "model.onnx").exists(), reason="no film checkpoint")
def test_shadow_record_keeps_concepts_apart():
    an = ShadowAnalyzer(FILM, AUTH if (AUTH / "model.onnx").exists() else None, STYLE if (STYLE / "model.onnx").exists() else None)
    out = an.analyze(SAMPLE)
    json.dumps(out)                                              # serialisable
    assert out["schema_version"] == "shadow-1.0"
    fa = out["film_aesthetics"]
    assert 0 <= fa["film_score"] <= 100 and "film_authenticity" not in fa
    va = out["visual_attributes"]
    assert set(va["badge_candidates"]) <= set(BADGE_RULES) and "measured" in va
    if "film_authenticity" in out:
        auth = out["film_authenticity"]
        assert 0 <= auth["p_film_scan"] <= 1 and auth["status"] == "experimental" and "not calibrated" in auth["note"]
        assert auth["p_film_scan"] != fa["film_score"] / 100   # never the same number re-used as the aesthetic score
    if "style_embedding" in out:
        v = out["style_embedding"]["vector"]
        assert abs(sum(x * x for x in v) - 1) < 1e-3 and out["style_embedding"]["status"] == "experimental"
    assert out["uncertainty"]["authenticity_calibrated"] is False and out["uncertainty"]["human_validation"] == "none"


def test_badge_rules_are_deterministic_and_explicit():
    f = {"is_mono": 0, "warmth": 0.1, "sat_mean": 0.1, "y_mean": 0.6, "y_std": 0.15, "clip_high": 0.0, "clip_low": 0.0,
         "p01": 0.02, "noise_std": 0.012, "noise_autocorr1": 0.5}
    hits = {k for k, r in BADGE_RULES.items() if r(f)}
    assert {"warm_palette", "muted_colours", "pastel_tones", "soft_highlights", "deep_shadows", "fine_grain"} <= hits
    assert "cool_palette" not in hits and "high_contrast" not in hits
    assert hits == {k for k, r in BADGE_RULES.items() if r(dict(f))}
