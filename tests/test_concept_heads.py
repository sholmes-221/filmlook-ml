"""The NumPy-only concept-heads reference (research/concepts/predict.py) and the shipped artefact, when present."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.concepts import predict as P
from research.concepts.train import threshold, wilson

ROOT = Path(__file__).resolve().parents[1]


def _art(tmp_path, W, b, thresholds, signals):
    art = {"version": "t", "kind": "concepts", "embedding": P.EMBEDDING, "activation": "sigmoid", "classes": [f"c{i}" for i in range(len(b))],
           "weights": np.asarray(W).tolist(), "bias": list(b),
           "outputs": [{"signal": s, "classes": [i], "threshold": t} for i, (s, t) in enumerate(zip(signals, thresholds))]}
    p = tmp_path / "a.json"; p.write_text(json.dumps(art))
    return P.load(p)


def test_predict_matches_logistic_formula(tmp_path):
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(3, 768)) * 0.1, [0.5, -1.0, 0.0]
    art = _art(tmp_path, W, b, [0.5, 0.5, 0.99], ["umbrella", "birds", "birds"])
    v = rng.normal(size=768).astype(np.float32)
    got = P.predict(art, v)
    want = 1 / (1 + np.exp(-(np.asarray(W, np.float32) @ v + np.asarray(b, np.float32))))
    assert np.allclose([got[c] for c in art["classes"]], want, atol=1e-6)


def test_signals_use_thresholds_and_merge_keys(tmp_path):
    W = np.zeros((3, 768)); W[0, 0] = 10; W[1, 1] = 10; W[2, 2] = 10
    art = _art(tmp_path, W, [0, 0, 0], [0.9, 0.9, 0.9], ["umbrella", "birds", "birds"])
    v = np.zeros(768, np.float32); v[1] = 1.0   # only class 1 fires (sigmoid(10) > 0.9); others sit at 0.5
    assert P.signals(art, v) == {"birds"}
    v[0] = 1.0
    assert P.signals(art, v) == {"umbrella", "birds"}


def test_rejects_wrong_shapes(tmp_path):
    art = _art(tmp_path, np.zeros((1, 768)), [0], [0.5], ["moon"])
    with pytest.raises(ValueError):
        P.predict(art, np.zeros(384))
    bad = json.loads((tmp_path / "a.json").read_text()); bad["embedding"] = "siglip"
    (tmp_path / "b.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        P.load(tmp_path / "b.json")


def test_threshold_rule_is_precision_first():
    # 20 positives score above 10 negatives, then more positives: 21 fires keep precision 20/21 >= 0.95, 22 do not
    p = np.r_[np.linspace(0.99, 0.80, 20), np.linspace(0.79, 0.70, 10), np.linspace(0.69, 0.5, 10)]
    y = np.r_[np.ones(20), np.zeros(10), np.ones(10)].astype(int)
    t = threshold(p, y)
    assert t is not None and abs(t - 0.79) < 1e-9
    assert threshold(np.linspace(1, 0, 10), np.zeros(10, int)) is None
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.95 < hi


def test_shipped_artefact_reference(tmp_path):
    arts = sorted((ROOT / "dist" / "concepts").glob("concept-heads-*.json"))
    if not arts:
        pytest.skip("no artefact built (dist/ is git-ignored)")
    art = P.load(arts[-1])
    words = {"night", "blue_hour", "golden_hour", "street", "birds", "sailor", "gas_station", "car_window", "suburb", "industrial",
             "pumpkin", "desert", "power_plant", "construction", "motorcycle", "starry", "water_lily", "sunflower", "diner", "wave",
             "fog", "pink", "corridor", "umbrella", "scooter", "moon", "fishing", "maple", "willow", "dragonfly", "duck", "boat",
             "flowers", "oranges", "tunnel", "dock", "cabin", "windmill", "wheat", "ice", "fox", "rickshaw", "lotus_pond",
             "spider_web", "beetle", "lesson_reflection"}
    for o in art["outputs"]:
        assert o["signal"] in words
        assert o["test"]["precision"] >= 0.90 and o["test"]["recall"] >= 0.30
    assert P.signals(art, np.zeros(768, np.float32)) <= words
