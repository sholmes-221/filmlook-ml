"""Photographer-style module: features, scoring rules, learned-model export, evaluation harness, JS suite."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from filmlook_ml.style import learned as L
from filmlook_ml.style.engine import analyze_features
from filmlook_ml.style.features import (FEATURE_NAMES, box2x, feature_vector, resize_for_style, style_features,
                                        target_size)
from filmlook_ml.style.profiles import (DIM_NAMES, PROFILE_IDS, PROFILES, dims_from_features, percentile,
                                        profile_evidence)

ROOT = Path(__file__).resolve().parent.parent
Q = {pid: list(np.linspace(0, 1, 101)) for pid in PROFILE_IDS}   # synthetic uniform rarity tables


@pytest.fixture(scope="module")
def scene_feats(fixture_dir):
    return [style_features(np.asarray(Image.open(p).convert("RGB"))) for p in sorted(fixture_dir.glob("*.png"))]


def test_features_finite_and_complete(scene_feats):
    for f in scene_feats:
        assert set(f) == set(FEATURE_NAMES) and len(f) == len(FEATURE_NAMES) and np.isfinite(feature_vector(f)).all()


def test_resize_two_stage_and_box2x():
    a = np.random.default_rng(0).integers(0, 256, (1000, 1500, 3), dtype=np.uint8)
    out = resize_for_style(a)
    assert out.shape[:2][::-1] == target_size(1500, 1000)
    b = np.array([[[0], [1]], [[2], [4]]], np.uint8).repeat(3, 2)
    assert box2x(b)[0, 0, 0] == (0 + 1 + 2 + 4 + 2) // 4
    small = a[:200, :300]
    assert resize_for_style(small) is small   # no upscaling


def test_mono_feature():
    g = np.repeat(np.random.default_rng(1).integers(0, 256, (100, 120, 1), dtype=np.uint8), 3, 2)
    f = style_features(g)
    assert f["mono"] == 1.0 and f["sat_mean"] == 0.0 and dims_from_features(f)["monochrome"] >= 0.99


def test_typical_photo_matches_nothing_and_targets_match_fully():
    typical = {d: 0.5 for d in DIM_NAMES}
    typical["monochrome"] = 0.0
    for pid, p in PROFILES.items():
        on_target = dict(typical)
        for d, (t, _) in p["profile"].items():
            on_target[d] = t
        assert profile_evidence(on_target, p["profile"])["score"] == pytest.approx(1.0), pid
        s = profile_evidence(typical, p["profile"])["score"]
        assert s < 0.35, (pid, s)   # near-neutral targets give a little credit, never a match


def test_monochrome_gate_blocks_colour_profiles():
    d = {k: 0.5 for k in DIM_NAMES}
    d["monochrome"] = 1.0
    for pid in ("saul_leiter", "william_eggleston", "alex_webb", "fred_herzog"):
        assert profile_evidence(d, PROFILES[pid]["profile"])["score"] == 0.0


def test_percentile_lookup():
    q = [0.0, 0.1, 0.2, 0.5, 1.0]
    assert percentile(-1, q) == 0 and percentile(2, q) == 1
    assert percentile(0.15, q) == pytest.approx(0.375)


def test_engine_contract(scene_feats):
    for f in scene_feats:
        for lang in ("en", "zh"):
            r = analyze_features(f, Q, lang)
            assert r["status"] in ("match", "uncertain") and len(r["matches"]) == 3
            assert (r["status"] == "match") == r["matches"][0]["reliable"]
            assert "photographer" in r["disclaimer"] or "摄影师" in r["disclaimer"]
            assert analyze_features(f, Q, lang) == r   # deterministic
            jsonschema = pytest.importorskip("jsonschema")
            jsonschema.validate(r, json.loads((ROOT / "js/filmlook-style/schema/style_output_v1.json").read_text()))
    with pytest.raises(ValueError):
        analyze_features(scene_feats[0], Q, "fr")


@pytest.mark.parametrize("kind", L.KINDS)
def test_learned_export_matches_sklearn(kind, scene_feats):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATURE_NAMES)))
    y = np.where(X[:, 0] + X[:, 3] > 0.5, "saul_leiter", np.where(X[:, 5] < -0.3, "fan_ho", "none")).tolist()
    sc, m = L.fit(kind, X, y, seed=0)
    ex = json.loads(json.dumps(L.export(kind, sc, m)))   # JSON round-trip, as shipped to JS
    for x in X[:40]:
        np.testing.assert_allclose(L.predict_exported(x, ex), m.predict_proba(sc.transform(x[None]))[0], atol=1e-9)


def test_evaluation_harness_runs_on_synthetic_judgements(tmp_path):
    """Code-path test only: judgements are synthetic, so the numbers mean nothing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("se", ROOT / "scripts" / "style_evaluate.py")
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    rng = np.random.default_rng(0)
    feats, groups, js = {}, {}, []
    for i in range(160):
        f = {k: float(v) for k, v in zip(FEATURE_NAMES, rng.normal(0.3, 0.1, len(FEATURE_NAMES)))}
        f["mono"] = 0.0
        iid = f"img{i}"
        feats[iid], groups[iid] = f, f"g{i % 40}"
        lab = "saul_leiter" if f["sat_mean"] > 0.3 else "none"
        for src in ("claude", "human"):
            js.append({"image_id": iid, "source": src, "matches": [] if lab == "none" else [lab],
                       "no_match": lab == "none"})
    res = se.run(js, feats, groups, Q, "claude", "human", emb=None, seeds=(0,))
    for k in ("profile", "linear", "forest", "mlp"):
        assert 0 <= (res[k]["top1"] if k == "profile" else res[k]["top1"]["mean"]) <= 1
    assert "BLOCKED" in se.run(js[:10], feats, groups, Q, "claude", "human")["status"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_suite():
    """Runs the Node test suite (parity with Python fixtures, ES2017/sandbox checks, behaviour)."""
    fx = ROOT / "js" / "filmlook-style" / "test" / "fixtures" / "golden.json"
    if not fx.exists():
        pytest.skip("run scripts/style_make_fixtures.py first")
    r = subprocess.run(["node", "--test", str(ROOT / "js" / "filmlook-style" / "test" / "style.test.js")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
