"""Reproducibility tests for the session-6 helpers (E46-E51): pooled-layer reductions, perturbations, neighbour coherence."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")
from e43_penultimate_pca import PCA64  # noqa: E402
from e47_pooled_reductions import Int8, RProj, Whiten, reductions  # noqa: E402
from e49_robustness import INVARIANT, PERTURB, perturb  # noqa: E402
from e50_aesthetic_vs_content import coherence, mono_agreement, onehots  # noqa: E402


def test_reductions_shapes_and_fit_set_independence():
    rng = np.random.default_rng(0); Zfit = rng.standard_normal((300, 576)).astype(np.float32) @ np.diag(np.linspace(0.1, 3, 576)).astype(np.float32); Z = rng.standard_normal((20, 576)).astype(np.float32)
    red = reductions(Zfit); dims = {"raw576": 576, "pca8": 8, "pca16": 16, "pca32": 32, "pca64": 64, "pca128": 128, "pcaw64": 64, "rproj64": 64, "pca64_int8": 64, "pca32_int8": 32}
    for name, f in red.items():
        assert f(Z).shape == (20, dims[name]), name
    W = Whiten(PCA64(Zfit, 16), Zfit)(Zfit); assert np.allclose(W.std(0), 1, atol=0.05)
    q = Int8(PCA64(Zfit, 8), Zfit); Y = PCA64(Zfit, 8)(Z); assert np.abs(q(Z) - Y).max() <= np.abs(q.scale).max() * 0.51 + 1e-6      # within half a quantisation step on the fit range
    assert len(np.unique(np.round(q(Z) / q.scale))) <= 255
    assert RProj(Zfit, 64, seed=0)(Z).shape == (20, 64) and not np.allclose(RProj(Zfit, 64, seed=0)(Z), RProj(Zfit, 64, seed=1)(Z))


def test_perturbations_keep_dtype_and_change_geometry_as_intended():
    rng = np.random.default_rng(0); a = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    for k in PERTURB:
        b = perturb(a, k); assert b.dtype == np.uint8 and b.ndim == 3 and b.shape[2] == 3, k
    assert perturb(a, "crop80").shape == (160, 240, 3) and perturb(a, "crop70_corner").shape == (140, 210, 3)
    assert np.array_equal(perturb(a, "flip"), a[:, ::-1]) and perturb(a, "jpeg50").shape == a.shape and perturb(a, "rescale0.5").shape == a.shape
    assert set(INVARIANT) < set(PERTURB) and "crop70_corner" not in INVARIANT


def test_coherence_ratio_below_one_when_neighbours_share_attributes():
    rng = np.random.default_rng(0); n, k = 120, 4; lab = np.repeat(np.arange(k), n // k)
    attr = np.repeat(rng.standard_normal((k, 28)), n // k, 0) + 0.1 * rng.standard_normal((n, 28))          # attributes cluster with the label
    E = np.repeat(rng.standard_normal((k, 16)), n // k, 0) + 0.05 * rng.standard_normal((n, 16)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    group = np.arange(n) % 7
    r = coherence(E, attr, group); assert r["all28"] < 0.5 and r["colour_tone"] < 0.6 and r["texture"] < 0.6
    Er = rng.standard_normal((n, 16)); Er /= np.linalg.norm(Er, axis=1, keepdims=True); r2 = coherence(Er, attr, group); assert 0.8 < r2["all28"] < 1.2
    mono = lab % 2 == 0; m = mono_agreement(E, mono, group); assert m["agreement"] > 0.95 and abs(m["prior"] - 0.5) < 0.1


def test_onehots_one_column_per_level():
    X = onehots([np.array(["a", "b", "a"]), np.array(["x", "x", "y"])]); assert X.shape == (3, 4) and np.allclose(X.sum(1), 2)


def test_trainer_exposes_pooled_vicreg_options():
    src = (ROOT / "scripts" / "train_style_student.py").read_text()
    assert '"--vic-pooled"' in src and '"--vic-mode"' in src and 'choices=["batch", "source"]' in src and "vic_src" in src
