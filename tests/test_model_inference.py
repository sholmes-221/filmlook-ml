import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.labeling.features import N_FEATURES, compute_features, feature_vector
from filmlook_ml.models.filmlook_model import Ensemble, FilmLookHeads, load_ensemble, save_ensemble

SCHEMA = Path(__file__).resolve().parent.parent / "src" / "filmlook_infer" / "schema" / "filmlook_output_v1.json"


@pytest.mark.parametrize("kind", ["texture", "dino", "fused"])
def test_heads_forward_shapes(kind):
    h = FilmLookHeads(kind=kind, hidden=16)
    o = h(torch.randn(4, 768) if kind != "texture" else None, torch.randn(4, N_FEATURES))
    assert o["score"].shape == (4,) and o["attrs"].shape == (4, len(ATTRIBUTES)) and o["medium_logit"].shape == (4,)
    assert torch.all((o["score"] >= 0) & (o["score"] <= 1))


def test_ensemble_roundtrip_and_spread(tmp_path):
    ms = []
    for s in range(3):
        torch.manual_seed(s)
        ms.append(FilmLookHeads(kind="texture", hidden=16).eval())
    ms[0].t_medium.fill_(2.0)
    save_ensemble(tmp_path, ms, {"kind": "texture"})
    ens, meta = load_ensemble(tmp_path)
    f = torch.randn(5, N_FEATURES)
    a, b = Ensemble(ms).eval()(None, f), ens(None, f)
    assert torch.allclose(a["score"], b["score"]) and meta["kind"] == "texture"
    assert (b["score_std"] > 0).all() and ens.members[0].t_medium.item() == 2.0


def test_feature_vector_finite(fixture_dir):
    v = feature_vector(compute_features(Image.open(fixture_dir / "scene_0.png")))
    assert v.shape == (N_FEATURES,) and np.isfinite(v).all()


def test_backbone_forward_and_freeze():
    from filmlook_ml.models.filmlook_model import DinoEmbedder
    m = DinoEmbedder().eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 112, 112)).shape == (2, 768)   # works at reduced resolution
    m.freeze(2)
    trainable = {n.split(".")[2] for n, p in m.model.named_parameters() if p.requires_grad and n.startswith("encoder")}
    assert trainable == {"10", "11"}


def test_onnx_parity_texture(texture_onnx, fixture_dir):
    from filmlook_ml.export.onnx_export import parity
    r = parity(texture_onnx, texture_onnx / "model.onnx", n=2, images=sorted(fixture_dir.glob("*.png"))[:2])
    assert all(v["max_abs"] < 1e-4 for v in r.values()), r


def test_onnx_parity_fused_and_int8(fused_onnx, fixture_dir):
    from filmlook_ml.export.onnx_export import parity
    imgs = sorted(fixture_dir.glob("*.png"))[:2]
    assert all(v["max_abs"] < 1e-3 for v in parity(fused_onnx, fused_onnx / "model.onnx", n=2, images=imgs).values())
    assert (fused_onnx / "model.int8.onnx").stat().st_size < (fused_onnx / "model.onnx").stat().st_size * 0.5


def _validate(obj):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(obj, json.loads(SCHEMA.read_text()))


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("model", ["texture_onnx", "fused_onnx"])
def test_analyze_schema_bilingual(request, model, fixture_dir, lang):
    from filmlook_infer import FilmLookAnalyzer
    an = FilmLookAnalyzer(request.getfixturevalue(model))
    r = an.analyze(fixture_dir / "scene_1.png", language=lang)
    _validate(r)
    assert r["language"] == lang and isinstance(r["film_score"], int)
    has_cjk = bool(re.search(r"[一-鿿]", json.dumps({k: v for k, v in r.items() if k != "model_version"},
                                                    ensure_ascii=False)))
    assert has_cjk == (lang == "zh")
    assert "photographer_dna" not in r and "film_stock_matches" not in r


def test_deterministic_output(texture_onnx, fixture_dir):
    from filmlook_infer import FilmLookAnalyzer
    a = FilmLookAnalyzer(texture_onnx).analyze(fixture_dir / "scene_2.png", "en")
    b = FilmLookAnalyzer(texture_onnx).analyze(fixture_dir / "scene_2.png", "en")
    assert a == b


def test_monochrome_has_no_color_claims(texture_onnx, fixture_dir, tmp_path):
    from filmlook_infer import FilmLookAnalyzer
    p = tmp_path / "mono.png"
    Image.open(fixture_dir / "scene_3.png").convert("L").save(p)
    for lang in ("en", "zh"):
        r = FilmLookAnalyzer(texture_onnx).analyze(p, lang)
        _validate(r)
        assert r["monochrome"] and r["attributes"]["color_response"] is None
        assert not {"warm", "cool", "neutral", "muted", "vivid"} & set(r["style_tags"])
        txt = " ".join(r["strengths"] + r["weaknesses"]).lower()
        assert "color" not in txt and "色彩" not in txt and "彩色" not in txt


def test_error_handling(texture_onnx, tmp_path, fixture_dir):
    from filmlook_infer import FilmLookAnalyzer, InvalidImage, UnsupportedLanguage
    an = FilmLookAnalyzer(texture_onnx)
    bad = tmp_path / "x.jpg"
    bad.write_text("nope")
    with pytest.raises(InvalidImage):
        an.analyze(bad)
    with pytest.raises(InvalidImage):
        an.analyze(tmp_path / "missing.jpg")
    with pytest.raises(UnsupportedLanguage):
        an.analyze(fixture_dir / "scene_0.png", language="fr")
    with pytest.raises(FileNotFoundError):
        FilmLookAnalyzer(tmp_path)


def test_cli_json_and_runtime_isolation(texture_onnx, fixture_dir, tmp_path):
    """The runtime must not import torch or training code."""
    src = Path(__file__).resolve().parent.parent / "src"
    code = ("import sys, runpy; sys.argv=['x', sys.argv[1], '--model', sys.argv[2], '--lang', 'zh'];"
            "import filmlook_infer.cli as c; rc=c.main(sys.argv[1:]);"
            "assert 'torch' not in sys.modules and 'filmlook_ml' not in sys.modules; sys.exit(rc)")
    env = {"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code, str(fixture_dir / "scene_0.png"), str(texture_onnx)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["language"] == "zh"
    bad = tmp_path / "bad.jpg"
    bad.write_text("x")
    r = subprocess.run([sys.executable, "-c", code, str(bad), str(texture_onnx)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and json.loads(r.stdout)["error"] == "invalid_image"
