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


def test_v1_rendering_is_unchanged_by_grain_model_option(img):
    """The v1 default (correlated) must stay bit-identical to outputs generated before grain models existed."""
    from filmlook_ml.filmsynth.engine import GRAIN_MODELS
    p = {"grain_strength": 0.5, "grain_size": 0.4, "chroma_grain": 0.2}
    a = render(img, p, 3)
    b = render(img, {**p, "grain_model": "correlated"}, 3)
    assert np.array_equal(a, b)
    fields = {m: render(img, {**p, "grain_model": m}, 3).astype(float) - img for m in GRAIN_MODELS}
    for m, f in fields.items():
        assert 2 < f.std() < 25, m                              # every model applies visible, bounded grain
    # the models differ in spatial correlation (lag-1 autocorrelation of the luma residual)
    def ac1(f):
        y = f @ [0.2126, 0.7152, 0.0722]; y = y - y.mean()
        return float((y[:, 1:] * y[:, :-1]).mean() / (y.var() + 1e-9))
    assert ac1(fields["gaussian"]) < ac1(fields["correlated"]) and ac1(fields["correlated"]) < ac1(fields["multiscale"])
    lum = np.clip(_luma(img) / 255, 0, 1)
    r = (fields["poisson"] @ [0.2126, 0.7152, 0.0722])
    assert r[lum > 0.6].std() > r[lum < 0.3].std()             # poisson grain grows with luminance


def test_v2_config_randomises_grain_independently_of_family():
    cfg = load_presets(ROOT / "configs" / "film_synth_v2.yaml")
    from filmlook_ml.filmsynth.engine import GRAIN_MODELS
    assert cfg["version"] == "film-synth-v2"
    for fam, c in cfg["families"].items():
        assert set(c["params"]["grain_model"]) == set(GRAIN_MODELS), fam
    ident = cfg["families"]["identity_v2"]["params"]
    assert ident["grain_strength"][1] > 0                      # the untreated control also carries grain
    rng = np.random.default_rng(0)
    models = {sample_params(cfg["families"]["warm_negative_v2"], rng)["grain_model"] for _ in range(40)}
    assert len(models) == 4


def test_v1_render_pinned_against_verified_output(img):
    """Pinned 2026-11-07 after the full film-synth-v1 dataset regenerated to its original sha set (d1b3f36e...).
    If this changes, v1 outputs are no longer reproducible and every v1 result must be re-run."""
    import hashlib
    p = {"temperature_shift": 0.3, "contrast_adjustment": -0.2, "shadow_lift": 0.3, "highlight_compression": 0.5,
         "saturation_adjustment": -0.2, "grain_strength": 0.4, "grain_size": 0.5, "chroma_grain": 0.2,
         "grain_midtone_weight": 0.6, "halation": 0.4, "vignette": 0.3, "local_contrast": 0.2}
    assert hashlib.sha256(render(img, p, 11).tobytes()).hexdigest() == \
        "1b1ddb2babcac2e40d619e1b23a7c606326d2f628b79b0ca4dc052c3db150845"
