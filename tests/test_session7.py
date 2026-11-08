"""Session 7 helpers: fresh-holdout ingest adapters, the frozen candidate, the holdout metrics and the abstention rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))


def test_met_select_caps_artists_and_groups_unknown():
    import ingest_met_photographs as I
    cands = [{"Object ID": str(i), "Artist Display Name": "Unknown" if i % 3 == 0 else f"A{i % 4}", "Constituent ID": str(i % 4), "Object Begin Date": "1860"} for i in range(60)]
    sel = I.select(cands, target=30, per_artist=5)
    from collections import Counter
    per = Counter(I.artist_key(c) for c in sel)
    assert max(per.values()) <= 5 and len(sel) <= 30 and "unknown" in per
    assert I.half_of("a") in ("analysis", "test") and I.half_of("a") == I.half_of("a")


def test_commoncatalog_configure_switches_every_path_and_caps():
    import ingest_commoncatalog_film as C
    C.configure("commoncatalog_film_hd", "30,10,150", 768)
    assert C.DS == "commoncatalog_film_hd" and C.CAPS == {"per_uploader": 30, "per_uploader_stock": 10, "per_stock": 150} and C.MIN_SIDE == 768
    assert C.CAND.name == "commoncatalog_film_hd_candidates.jsonl" and C.MANIFEST.name == "commoncatalog_film_hd_manifest.jsonl" and C.RAW.name == "commoncatalog_film_hd"
    rows = [{"photoid": str(i), "uid": f"u{i % 5}", "width": 2048, "height": 1500, "film_stock": "kodak_portra_400", "film_brand": "kodak"} for i in range(50)]
    sel = C.select(rows, caps=C.CAPS, exclude_uploaders={"u0"})
    assert all(r["uid"] != "u0" for r in sel) and len(sel) == 40
    C.configure("commoncatalog_film", "20,6,150", 1024)


def test_e53_hits_and_paired_diff_are_consistent():
    import e53_fresh_holdout as E
    rng = np.random.default_rng(0); n = 60; label = np.array(["a", "b", "c"] * 20); group = np.arange(n) // 3
    E1 = rng.standard_normal((n, 8)).astype(np.float32); E1[label == "a", 0] += 3; E1[label == "b", 1] += 3; E1[label == "c", 2] += 3
    from style_benchmark import unit
    h, g, pr, qi, top = E.hits(unit(E1), label, group)
    assert h.shape == g.shape == pr.shape == qi.shape and top.shape == (len(h), 10) and h.mean() > 0.8 and abs(pr.mean() - 1 / 3) < 0.05
    h0, *_ = E.hits(unit(rng.standard_normal((n, 8)).astype(np.float32)), label, group)
    d = E.paired_diff(h, h0, g)
    assert d["mean"] > 0 and d["ci95"][0] > 0 and d["n"] == len(h)


def test_e53_perturbations_keep_dtype_and_change_shape_only_for_crops():
    import e53_fresh_holdout as E
    a = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    for k in E.PERTURB:
        b = E.perturb(a, k); assert b.dtype == np.uint8 and b.ndim == 3
        if k.startswith("crop"):
            assert b.shape[0] < a.shape[0]
        else:
            assert b.shape == a.shape


@pytest.mark.skipif(not (ROOT / "checkpoints" / "frozen_session7" / "freeze.json").exists(), reason="frozen candidate not on this machine")
def test_frozen_reduce_is_nested_and_int8_bounded():
    from freeze_candidate import CANDIDATE, Frozen
    fz = Frozen(CANDIDATE[0]); F = np.random.default_rng(2).standard_normal((5, 576)).astype(np.float32)
    z64, z32, q = fz.reduce(F, 64, False), fz.reduce(F, 32, False), fz.reduce(F, 32, True)
    assert np.allclose(z64[:, :32], z32) and q.shape == (5, 32)
    assert np.all(np.abs(np.round(q / fz.scale32)) <= 127)


def test_e54_edits_identity_at_the_identity_level():
    import e54_characteristic_probe as E
    a = (np.random.default_rng(3).random((64, 80, 3)) * 255).astype(np.uint8)
    for attr, vals in E.ATTR.items():
        b = E.edit(a, attr, vals[E.IDENTITY[attr]])
        assert b.shape == a.shape and b.dtype == np.uint8 and np.abs(b.astype(int) - a.astype(int)).max() <= 1, attr
        c = E.edit(a, attr, vals[-1]); assert np.abs(c.astype(int) - a.astype(int)).mean() > 0.5, attr


def test_e55_calibration_and_evaluation_on_synthetic_records():
    import e55_abstention as E
    rng = np.random.default_rng(4); n = 200
    r = {"correct": (rng.random(n) < 0.6).astype(float), "agree": rng.choice([0.4, 0.6, 0.8, 1.0], n), "top1": rng.random(n), "groups": rng.integers(0, 20, n), "prior": np.full(n, 0.2), "n": n}
    r["correct"][r["top1"] > 0.7] = 1.0                                   # confident queries are right
    tau, chance = E.calibrate(r, 0.6); assert tau is not None and chance == 0.2
    ev = E.evaluate(r, tau, 0.6)
    assert 0 < ev["coverage"] <= 1 and ev["precision_answered"] >= 0.4 and ev["n_answered"] == int(((r["agree"] >= 0.6) & (r["top1"] >= tau)).sum())
    tau_hi, _ = E.calibrate(r, 0.6, target_mult=4.5)                      # a stricter target needs a higher threshold and buys precision
    ev_hi = E.evaluate(r, tau_hi, 0.6)
    assert tau_hi > tau and ev_hi["precision_answered"] >= 0.9 and ev_hi["coverage"] < ev["coverage"]
    assert E.calibrate({**r, "correct": np.zeros(n)}, 0.6)[0] is None
