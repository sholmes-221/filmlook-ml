"""Film-synth engine: determinism, neutral no-op, parameter monotonicity, preset/taxonomy consistency, schema."""
from pathlib import Path

import numpy as np
import pytest

from filmlook_ml.filmsynth import SyntheticAnnotation, load_presets, sample_nuisance, sample_params, variant_seed
from filmlook_ml.filmsynth.engine import PARAM_NAMES, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:96, 0:128]
    base = np.stack([x * 2, y * 2.5, 255 - x * 1.5], 2) % 256
    return np.clip(base + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)


def _luma(a):
    return (a.astype(float) @ [0.2126, 0.7152, 0.0722])


def test_neutral_parameters_are_a_no_op(img):
    assert np.array_equal(render(img, {k: 0.0 for k in PARAM_NAMES}, seed=1), img)


def test_render_is_deterministic_per_seed(img):
    p = {"grain_strength": 0.5, "grain_size": 0.4, "temperature_shift": 0.3}
    assert np.array_equal(render(img, p, 7), render(img, p, 7))
    assert not np.array_equal(render(img, p, 7), render(img, p, 8))      # grain pattern depends on the seed


def test_parameters_move_measurements_in_the_documented_direction(img):
    warm, cool = render(img, {"temperature_shift": 0.8}, 0), render(img, {"temperature_shift": -0.8}, 0)
    rb = lambda a: a[..., 0].mean() - a[..., 2].mean()  # noqa: E731
    assert rb(warm) > rb(img) > rb(cool)
    assert _luma(render(img, {"contrast_adjustment": 0.8}, 0)).std() > _luma(img).std() > \
        _luma(render(img, {"contrast_adjustment": -0.8}, 0)).std()
    assert render(img, {"shadow_lift": 1.0}, 0).min() >= 25                # black raised to about 0.12
    assert render(img, {"highlight_compression": 1.0}, 0).max() < 255
    m = render(img, {"monochrome": True}, 0)
    assert np.abs(m[..., 0].astype(int) - m[..., 2]).max() <= 1
    g = render(img, {"grain_strength": 1.0}, 0).astype(float) - img
    assert 3 < g.std() < 20                                                # visible but not destructive


def test_presets_reference_existing_database_aesthetics():
    from filmlook_ml.taxonomy_db import load_taxonomy
    if not (ROOT / "taxonomy" / "film_aesthetics.v1.json").exists():
        pytest.skip("taxonomy not extracted")
    t = load_taxonomy()
    cfg = load_presets()
    for fam, c in cfg["families"].items():
        assert fam.endswith("_v1")
        for ref in c["approximate_references"]:
            assert ref in t.aesthetics, (fam, ref)
        for k in c["params"]:
            assert k in PARAM_NAMES or k == "monochrome", k


def test_sampling_and_nuisance_are_seeded_and_label_independent():
    cfg = load_presets()
    s = variant_seed("img_1", "warm_negative_v1", 3, cfg["version"])
    a = sample_params(cfg["families"]["warm_negative_v1"], np.random.default_rng(s))
    b = sample_params(cfg["families"]["warm_negative_v1"], np.random.default_rng(s))
    assert a == b and set(PARAM_NAMES) <= set(a)
    lo, hi = cfg["families"]["warm_negative_v1"]["params"]["temperature_shift"]
    assert lo <= a["temperature_shift"] <= hi
    n = sample_nuisance(cfg, np.random.default_rng(0))
    assert set(n) == {"jpeg_quality", "exposure_ev", "crop_fraction", "resample"}


def test_annotation_schema_marks_synthetic_as_not_real_film():
    ann = SyntheticAnnotation(source_image_id="x_original", variant_id="x__warm_negative_v1__01",
                              augmentation_family="warm_negative_v1", augmentation_version="film-synth-v1",
                              parameters={"temperature_shift": 0.3, "monochrome": False},
                              nuisance={"jpeg_quality": 90, "exposure_ev": 0.0, "crop_fraction": 0.01, "resample": "bicubic"},
                              seed=1, split="train", group_id="g", path="p", sha256="s")
    assert ann.is_real_film_scan is False and ann.label_source == "deterministic_synthetic_augmentation"
