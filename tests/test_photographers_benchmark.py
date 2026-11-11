"""The photographer retrieval benchmark's building blocks on toy data: metrics, leakage groups that never straddle
splits, and a photographer score that does not reward having more works in the bank."""
import numpy as np

from research.photographers import evaluate as E


def test_retrieval_metrics_on_a_known_ranking():
    S = np.array([[0.9, 0.8, 0.1, 0.0], [0.1, 0.2, 0.9, 0.3]], dtype=np.float32)
    yb, yq = np.array([0, 1, 1, 0]), np.array([1, 0])
    m = E.retrieval_metrics(S, yq, yb)
    # query 0 (class 1): ranking b0(0), b1(1), b2(1), b3(0) -> first hit at rank 2; query 1 (class 0): b2(1), b3(0), b1, b0 -> rank 2
    assert m["recall@1"] == 0.0 and m["recall@5"] == 1.0 and abs(m["mrr"] - 0.5) < 1e-6
    assert abs(m["map"] - np.mean([(1 / 2 + 2 / 3) / 2, (1 / 2 + 2 / 4) / 2])) < 1e-6


def test_groups_merge_sessions_and_near_duplicates_and_never_straddle_splits():
    rng = np.random.default_rng(0)
    X = E.l2(rng.normal(size=(40, 16)).astype(np.float32))
    X[1] = X[0]  # an alternate scan of the same negative
    rows = [{"work_id": f"w:{i:03d}", "photographer_id": "p" + str(i % 4), "title": f"title number {i}", "year": "1936", "phash": f"{rng.integers(0, 2**63):016x}"} for i in range(40)]
    rows[5]["title"] = rows[9]["title"] = "street corner in town"; rows[9]["photographer_id"] = rows[5]["photographer_id"]  # one sitting
    rows[20]["phash"] = rows[21]["phash"]  # identical perceptual hash
    g, audit = E.leakage_groups(rows, X)
    assert g[0] == g[1] and g[5] == g[9] and g[20] == g[21]
    assert audit["groups"] == len(set(g.tolist())) < 40
    for seed in range(5):
        split = E.split_groups(rows, g, seed)
        for group in set(g.tolist()):
            assert len(set(split[g == group].tolist())) == 1
        assert set(split.tolist()) <= {"bank", "val", "test"}


def test_the_photographer_score_is_not_a_head_count():
    # Class 0 has 50 mediocre works, class 1 has 3 close ones: the top-m mean must prefer class 1.
    Sq = np.concatenate([np.full(50, 0.55), np.full(3, 0.80)])[None, :].astype(np.float32)
    yb = np.array([0] * 50 + [1] * 3)
    assert E.topm_scores(Sq, yb, 2, 3).argmax(1)[0] == 1
    assert E.knn_scores(Sq, yb, 2, 3).argmax(1)[0] == 1
    P = E.softmax(np.array([[2.0, 1.0, 0.0]]), 1.0)
    assert abs(P.sum() - 1) < 1e-6 and E.ece(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 0])) < 0.11
