"""Reproducibility tests for the real-photograph evaluation helpers added in session 5 (E32-E38)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")
from e29_audit import eff_rank, retrieval, whiten  # noqa: E402
from frozen_head_choice import PCAW, share  # noqa: E402
from head_photographer_invariant import Remover  # noqa: E402
from within_photographer_stock import within_auc  # noqa: E402


def _clustered(n_per=10, k=4, d=16, seed=0, noise=0.05):
    rng = np.random.default_rng(seed); centres = rng.standard_normal((k, d))
    Z = np.concatenate([centres[i] + noise * rng.standard_normal((n_per, d)) for i in range(k)]); lab = np.repeat(np.arange(k), n_per)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True), lab


def test_retrieval_recovers_clusters_and_excludes_group():
    Z, lab = _clustered(); group = np.arange(len(Z)) % 5                     # groups cut across clusters
    r = retrieval(Z, lab.astype(str), group)
    assert r["precision@5"] > 0.95 and r["ci95"][0] > r["chance"] and r["n"] == len(Z)
    same = (group[:, None] == group[None]); r2 = retrieval(Z, lab.astype(str), np.arange(len(Z)), allowed=~same)   # mask removes same-group candidates
    assert r2["precision@5"] > 0.95
    rng = np.random.default_rng(1); r3 = retrieval(Z, rng.permutation(lab).astype(str), group)             # shuffled labels: near chance
    assert abs(r3["precision@5"] - r3["chance"]) < 0.15


def test_retrieval_allowed_mask_can_make_queries_unanswerable():
    Z, lab = _clustered(); none = np.zeros((len(Z), len(Z)), bool)
    assert retrieval(Z, lab.astype(str), np.arange(len(Z)), allowed=none)["n"] == 0


def test_eff_rank_and_whitening():
    rng = np.random.default_rng(0); iso = rng.standard_normal((500, 32)); collapsed = np.outer(rng.standard_normal(500), rng.standard_normal(32)) + 0.01 * rng.standard_normal((500, 32))
    assert eff_rank(iso) > 25 and eff_rank(collapsed) < 3
    W = whiten(collapsed); assert eff_rank(W) > 10 and np.allclose(np.linalg.norm(W, axis=1), 1, atol=1e-5)


def test_pcaw_whitens_fit_set_and_applies_to_new_data():
    rng = np.random.default_rng(0); A = rng.standard_normal((400, 20)) @ np.diag(np.linspace(0.1, 5, 20)); p = PCAW(8).fit(A)
    Y = p(A); C = np.cov(Y.T); assert Y.shape == (400, 8) and np.allclose(np.diag(C), 1, atol=0.05) and np.abs(C - np.diag(np.diag(C))).max() < 0.05
    assert p(rng.standard_normal((5, 20))).shape == (5, 8)


def test_remover_projects_out_subspace_and_modes_agree_on_shape():
    rng = np.random.default_rng(0); Z = rng.standard_normal((300, 64)); groups = np.repeat(np.arange(60), 5).astype(str)
    for mode in ("lda", "random", "pca"):
        rem = Remover(8, mode, n_pc=32).fit(Z, groups); Y = rem(Z)
        assert Y.shape == (300, 32) and np.abs(Y @ rem.Q).max() < 1e-6                               # nothing left along the removed directions
    assert Remover(0, "lda", n_pc=32).fit(Z, groups)(Z).shape == (300, 32)


def test_within_auc_separates_lines_only_inside_uploader_and_honours_day_and_colour_filters():
    rng = np.random.default_rng(0); n = 40; up = np.array(["a"] * 20 + ["b"] * 20); line = np.array((["p", "q"] * 10) * 2)
    E = np.where(line[:, None] == "p", 1.0, -1.0) * np.ones((n, 4)) + 0.01 * rng.standard_normal((n, 4)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    day = np.array([str(i % 3) for i in range(n)]); a, g = within_auc(E, line, up, day, False)
    assert len(a) == n and a.mean() > 0.99 and set(g) == {"a", "b"}
    a2, _ = within_auc(E, line, up, day, True); assert len(a2) == n and a2.mean() > 0.99
    colour = np.array([i % 4 < 2 for i in range(n)]); a3, _ = within_auc(E, line, up, day, True, colour); assert len(a3) == colour.sum()
    a4, _ = within_auc(E, rng.permutation(line), up, day, True); assert abs(a4.mean() - 0.5) < 0.2


def test_share_measures_group_clustering():
    Z, lab = _clustered(); assert share(Z, lab) > 0.95 and share(Z, np.arange(len(Z)) % 5) < 0.4


def test_student_matrix_reads_only_analysis_rows(monkeypatch):
    import json
    import student_real_matrix as M
    if not (M.AIC.with_suffix(".json").exists() and M.CCF.with_suffix(".json").exists()):
        pytest.skip("feature caches absent")
    ra, rg, rv, L = M.labels()
    rows_a = json.loads(M.AIC.with_suffix(".json").read_text())["rows"]; rows_g = json.loads(M.CCF.with_suffix(".json").read_text())["rows"]
    assert all(rows_a[i]["split"] == "analysis" for i in L["ia"]) and all(rows_g[i]["split"] == "analysis" for i in L["ig"])
    assert all(r["split"] == "analysis" for r in ra) and all(r["split"] == "analysis" for r in rg)
    assert len(ra) < len(rows_a) and len(rg) < len(rows_g)


def test_train_style_student_exposes_kd_target_choices():
    src = (ROOT / "scripts" / "train_style_student.py").read_text()
    assert '"--kd-target"' in src and 'choices=["supcon", "supcon_vic3", "pcaw64"]' in src
