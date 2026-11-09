"""Serving-side photographer matcher: integrity check, encoder compatibility rule, output shape."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ONNX = Path(__file__).resolve().parents[1] / "checkpoints" / "refdb" / "dinov2_small_cls.onnx"
pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="exported ONNX model not present")


def _index(tmp: Path) -> Path:
    d = tmp / "idx"; d.mkdir()
    (d / "dinov2_small_cls.onnx").symlink_to(ONNX)
    rng = np.random.default_rng(0)
    V = rng.normal(size=(30, 384)).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.save(d / "vectors_cls.f16.npy", V.astype(np.float16))
    db = sqlite3.connect(d / "works.sqlite")
    db.execute("CREATE TABLE works (row INTEGER PRIMARY KEY, work_id TEXT, photographer_id TEXT, title TEXT, year TEXT, source_url TEXT, license TEXT, license_evidence_url TEXT, credit TEXT)")
    for i in range(30):
        db.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?)", (i, f"t:{i}", ["a", "b", "c"][i % 3], f"w{i}", "1936", "https://example.org", "PD", "https://example.org", ""))
    db.commit(); db.close()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    (d / "manifest.json").write_text(json.dumps({"index_version": "test", "encoder": "facebook/dinov2-small@abc", "files": files,
                                                 "calibration": {"knn_k": 5, "temperature_dinov2": 0.05, "abstain_below_dinov2": 0.5}}))
    return d


def test_matcher_refuses_incompatible_vectors_and_tampered_files(tmp_path):
    from filmlook_infer.style_match import EncoderMismatch, StyleMatcher
    d = _index(tmp_path)
    m = StyleMatcher(d)
    v = np.ones(384, np.float32)
    with pytest.raises(EncoderMismatch):
        m.match_vector(v, "openai/clip-vit-b32")  # A text or other-encoder vector is never compared.
    out = m.match_vector(v, m.encoder)
    assert len(out["matches"]) == 3 and abs(sum(x["relative_closeness"] for x in out["matches"]) - 1) < 1e-3
    assert "not a probability" in out["note"] and isinstance(out["abstain"], bool)
    np.save(d / "vectors_cls.f16.npy", np.zeros((30, 384), np.float16))
    with pytest.raises(ValueError, match="index file changed"):
        StyleMatcher(d)


def test_matcher_embeds_an_image_end_to_end(tmp_path):
    from filmlook_infer.style_match import StyleMatcher
    m = StyleMatcher(_index(tmp_path))
    out = m.match_image(Image.new("RGB", (320, 240), (120, 110, 100)))
    assert out["index_version"] == "test" and len(out["nearest_works"]) == 3
