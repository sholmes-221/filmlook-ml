"""The statistics-first MVP (filmlook_infer.retrieval_mvp): deterministic, schema-valid, honest about what it claims,
bounded in time and memory, and respectful of the caller's look allowlist. Skips when the artifacts are absent."""

from __future__ import annotations

import io
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V1, SUP, ART = ROOT / "checkpoints" / "best", ROOT / "checkpoints" / "student_sup", ROOT / "checkpoints" / "retrieval_mvp"
pytestmark = pytest.mark.skipif(not (V1 / "model.onnx").exists() or not (SUP / "model.onnx").exists() or not (ART / "calibration.json").exists(), reason="MVP artifacts not present")


def photo(seed=0, size=(900, 700), mono=False, warm=0.0, sat=1.0, dark=False) -> io.BytesIO:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    base = 0.25 + 0.5 * (x / size[0]) * (0.6 + 0.4 * np.sin(y / 40.0)) + rng.normal(0, 0.01, size=(size[1], size[0]))
    if dark:
        base *= 0.35
    r = np.clip(base + warm, 0, 1); g = np.clip(base, 0, 1); b = np.clip(base - warm, 0, 1)
    rgb = np.stack([r, g, b], -1)
    if sat != 1.0:
        lum = rgb.mean(-1, keepdims=True); rgb = np.clip(lum + (rgb - lum) * sat, 0, 1)
    if mono:
        rgb = np.repeat(rgb.mean(-1, keepdims=True), 3, -1)
    im = Image.fromarray((rgb * 255).astype(np.uint8)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=92); buf.seek(0); return buf


@pytest.fixture(scope="module")
def mvp():
    from filmlook_infer.retrieval_mvp import RetrievalMVP
    return RetrievalMVP(V1, SUP, ART)


def test_output_matches_schema_v2_in_both_languages(mvp):
    import jsonschema
    schema = json.loads((ROOT / "src" / "filmlook_infer" / "schema" / "filmlook_output_v2.json").read_text())
    for lang in ("zh", "en"):
        out = mvp.analyze(photo(), lang)
        jsonschema.validate(out, schema)
        assert out["film_likeness"]["calibrated"] is True and 0 <= out["film_likeness"]["estimate_pct"] <= 100
        assert "film_stock_identity" in out["not_claimed"] and "photographer_identity" in out["not_claimed"]
        assert "使用了" not in out["look_suggestion"]["text"] and "shot on" not in out["look_suggestion"]["text"].lower()
        assert out["disclaimer"] and out["summary"]


def test_deterministic_for_identical_bytes(mvp):
    a = mvp.analyze(photo(3), "zh"); b = mvp.analyze(photo(3), "zh")
    assert a == b


def test_allowlist_is_respected_and_abstention_is_possible(mvp):
    mono = mvp.analyze(photo(mono=True), "zh", allowed_looks={"tri_x_like", "gold_like"})
    assert mono["monochrome"] is True and mono["look_suggestion"]["look_id"] == "tri_x_like"
    assert mono["characteristics"]["temperature"]["label"] == "monochrome" and mono["rendering_attributes"]["color_response"] is None
    none_allowed = mvp.analyze(photo(mono=True), "zh", allowed_looks={"gold_like"})
    assert none_allowed["look_suggestion"]["abstained"] is True and none_allowed["look_suggestion"]["look_id"] is None
    assert none_allowed["look_suggestion"]["text"] == "暂时没有找到足够接近的胶卷风格"
    full = mvp.analyze(photo(mono=True), "en", allowed_looks=None)
    assert full["look_suggestion"]["look_id"] == "tri_x_like" and "Tri-X 400" in full["look_suggestion"]["text"]


def test_invalid_input_and_language_raise_typed_errors(mvp):
    from filmlook_infer.analyzer import InvalidImage, UnsupportedLanguage
    with pytest.raises(InvalidImage):
        mvp.analyze(io.BytesIO(b"not an image"), "zh")
    with pytest.raises(UnsupportedLanguage):
        mvp.analyze(photo(), "fr")


def test_latency_and_memory_are_bounded_on_this_host(mvp):
    mvp.analyze(photo(1), "zh")  # warm
    tracemalloc.start(); t = time.perf_counter()
    for i in range(3):
        mvp.analyze(photo(10 + i, size=(1600, 1200)), "en")
    elapsed = (time.perf_counter() - t) / 3; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    assert elapsed < 3.0, elapsed  # the E2 Micro core is several times slower; the gate there is 3 s p95
    assert peak < 400 * 1024 * 1024


def test_calibration_artifact_is_the_student_head_with_recorded_reference(mvp):
    cal = json.loads((ART / "calibration.json").read_text())
    assert cal["inputs"] == ["sup_p_film"] and cal["read_auroc"] >= 0.75 and cal["read_ci95"][0] >= 0.65 and cal["read_ece"] <= 0.10
    meta = json.loads((ART / "meta.json").read_text())
    assert meta["rendering_family_prototypes"]["status"].startswith("experimental")
