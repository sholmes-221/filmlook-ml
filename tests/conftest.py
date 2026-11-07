from pathlib import Path

import numpy as np
import pytest

from filmlook_ml.datasets.synthetic import make_scene


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """Small synthetic fixture dataset: 6 procedural scenes."""
    d = tmp_path_factory.mktemp("fixture")
    for i in range(6):
        make_scene(1000 + i, (640, 448)).save(d / f"scene_{i}.png")
    return d


@pytest.fixture(scope="session")
def raw_records(fixture_dir):
    return [{"path": str(p), "dataset_id": "synthetic_fixture", "license": "synthetic", "source_id": p.stem}
            for p in sorted(fixture_dir.glob("scene_*.png"))]


def _ckpt(tmp_path_factory, kind: str, input_size=None):
    import torch

    from filmlook_ml.constants import ATTRIBUTES, MODEL_VERSION
    from filmlook_ml.labeling.features import N_FEATURES
    from filmlook_ml.models.filmlook_model import FilmLookHeads, save_ensemble

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp(f"ckpt_{kind}")
    members = []
    for s in range(3):
        torch.manual_seed(s)
        m = FilmLookHeads(kind=kind, hidden=32).eval()
        m.feat_std.fill_(0.1)
        members.append(m)
    meta = {"model_version": MODEL_VERSION, "kind": kind, "backbone": "facebook/dinov2-small" if kind != "texture" else None,
            "input_size": input_size, "max_side": 640, "texture_crop": 256, "attributes": ATTRIBUTES,
            "feat_mean": [0.0] * N_FEATURES, "feat_std": [0.1] * N_FEATURES}
    save_ensemble(d, members, meta)
    return d


@pytest.fixture(scope="session")
def texture_onnx(tmp_path_factory):
    from filmlook_ml.export.onnx_export import export_onnx
    d = _ckpt(tmp_path_factory, "texture")
    export_onnx(d)
    return d


@pytest.fixture(scope="session")
def fused_onnx(tmp_path_factory):
    from filmlook_ml.export.onnx_export import export_onnx, quantize_int8
    d = _ckpt(tmp_path_factory, "fused", input_size=112)
    quantize_int8(export_onnx(d))
    return d


@pytest.fixture
def rng():
    return np.random.default_rng(0)
