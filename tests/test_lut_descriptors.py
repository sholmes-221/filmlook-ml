"""Photo descriptors of the LUT engine: shape, determinism, ranges, sensitivity to colour, resolution robustness."""
import numpy as np

from research.lut.engine.descriptors import COLOUR_IDX, DIM, NAMES, TEXTURE_IDX, TONE_IDX, describe, describe_pixels, neutralise, srgb_to_lab


def synthetic_photo(seed=0, size=(300, 420), tint=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size[0], 0:size[1]]
    base = np.stack([0.3 + 0.5 * x / size[1], 0.2 + 0.6 * y / size[0], 0.5 + 0.3 * np.sin(x / 25.0)], -1)
    base += rng.normal(0, 0.03, base.shape)
    base += np.array(tint)
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def test_shape_names_determinism():
    img = synthetic_photo()
    d1, d2 = describe(img), describe(img)
    assert d1.shape == (DIM,) and len(NAMES) == DIM
    assert np.array_equal(d1, d2) and np.isfinite(d1).all()
    assert len(COLOUR_IDX) + len(TONE_IDX) + len(TEXTURE_IDX) == DIM


def test_histograms_normalised_and_ranges():
    d = describe(synthetic_photo())
    assert abs(d[0:16].sum() - 1) < 1e-4            # L histogram
    assert abs(d[16:116].sum() - 1) < 1e-3          # ab histogram
    i = NAMES.index("black_level")
    assert 0 <= d[i] <= 1 and 0 <= d[NAMES.index("white_level")] <= 1
    assert -1 <= d[NAMES.index("corr_rg")] <= 1


def test_colour_sensitivity_and_neutralise():
    plain = describe(synthetic_photo())
    warm = describe(synthetic_photo(tint=(0.08, 0.02, -0.08)))
    assert warm[NAMES.index("warm_b_mean")] > plain[NAMES.index("warm_b_mean")]
    assert warm[NAMES.index("mid_b")] > plain[NAMES.index("mid_b")]
    n = neutralise(synthetic_photo(tint=(0.08, 0.02, -0.08)))
    assert n.dtype == np.uint8 and n.shape == (300, 420, 3)
    lab = srgb_to_lab(n.reshape(-1, 3) / 255.0)
    assert abs(lab[:, 1].mean()) < 8 and abs(lab[:, 2].mean()) < 8   # grey-world cast mostly removed (gradient image)


def test_resolution_robustness():
    img = synthetic_photo(size=(600, 840))
    small = img[::2, ::2]
    a, b = describe(img)[COLOUR_IDX], describe(small)[COLOUR_IDX]
    d = np.abs(a - b)
    assert np.percentile(d, 95) < 0.03 and d.max() < 0.3     # sparse-band statistics (few shadow pixels) may move more


def test_pixel_cloud_descriptor():
    px = (np.random.default_rng(3).random((10000, 3)) * 255).astype(np.uint8)
    d = describe_pixels(px)
    assert d.shape == (DIM,) and np.all(d[TEXTURE_IDX] == 0)
