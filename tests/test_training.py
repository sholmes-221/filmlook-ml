"""Tiny end-to-end: fixture -> edits -> labels -> features/embeddings -> heads -> metrics."""
import numpy as np
import pytest
import torch

from filmlook_ml.datasets.edits import EDITS
from filmlook_ml.datasets.preprocess import group_split, prepare
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.evaluation.metrics import auroc, auroc_ci, evaluate
from filmlook_ml.labeling.pipeline import extract_features, feature_matrix, heuristic_labels, synthetic_labels
from filmlook_ml.labeling.store import build_targets, pair_arrays
from filmlook_ml.training.train import (TrainData, expected_calibration_error, extract_embeddings, fit_temperature,
                                        predict, train_ensemble, train_member)

CFG = {"backbone": "facebook/dinov2-small", "max_side": 640, "stage_a": {
    "epochs": 12, "batch_size": 16, "lr": 3e-3, "weight_decay": 1e-4, "hidden": 32, "dropout": 0.0, "rank_margin": 0.05,
    "loss_weights": {"score": 1, "attributes": 1, "ranking": 1, "medium": 0.5}, "early_stopping_patience": 50}}
W = {"human": 1.0, "synthetic_ordering": 0.7, "heuristic": 0.3}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, raw_records):
    tmp = tmp_path_factory.mktemp("e2e")
    man, rej = prepare(raw_records, Registry.load(), tmp / "proc", list(EDITS), max_side=640)
    assert not rej
    # pretend half the originals are film scans and half digital (metadata-style labels)
    for k, r in enumerate([r for r in man if r["edit_type"] == "original"]):
        r["medium"] = "film_scan" if k % 2 else "digital_camera"
    split = group_split(man, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0, group_key="group_id")
    ids = [r["image_id"] for r in man]
    feats = extract_features(man, tmp / "f.json", 256, workers=2, max_side=640)
    labels = {"heuristic": {x.image_id: x for x in heuristic_labels(man, feats, "v2")}}
    syn, pairs = synthetic_labels(man)
    labels["synthetic_ordering"] = {x.image_id: x for x in syn}
    t = build_targets(ids, labels, W, man)
    assert t.medium_w.sum() == 6 and t.attrs_w.sum() > 0
    return tmp, man, ids, feats, t, split, pairs


def test_end_to_end_texture_and_resume(tiny):
    tmp, man, ids, feats, t, split, pairs = tiny
    data = TrainData(ids, None, feature_matrix(ids, feats), t, np.array([split[i] for i in ids]),
                     pair_arrays(pairs, {i: k for k, i in enumerate(ids)}, W))
    members = train_ensemble(data, "texture", CFG, [0, 1], tmp / "ck", log=lambda *_: None)
    pred = predict(members, data)
    assert pred["score"].shape == (len(ids),) and (pred["score_members"].std(0) > 0).any()
    res = evaluate(pred, data, man, "test", {"synthetic_ordering": data.pairs})
    assert res["human_agreement"].startswith("unverified")
    assert "synthetic_ordering" in res["pairwise"]
    # resume: continuing a finished run reproduces the same best weights
    m2, hist = train_member(data, "texture", CFG, 0, tmp / "ck" / "last_texture_seed0.pt", resume=True,
                            log=lambda *_: None)
    assert torch.allclose(m2.score.weight, members[0].score.weight)


def test_end_to_end_fused_with_embeddings(tiny):
    tmp, man, ids, feats, t, split, pairs = tiny
    emb = extract_embeddings(man, CFG["backbone"], tmp / "e.npz", size=112, batch_size=16, max_side=640)
    data = TrainData(ids, np.stack([emb[i] for i in ids]), feature_matrix(ids, feats), t,
                     np.array([split[i] for i in ids]), pair_arrays(pairs, {i: k for k, i in enumerate(ids)}, W))
    m, _ = train_member(data, "fused", CFG, 0, log=lambda *_: None)
    assert predict([m], data)["film_capture"].shape == (len(ids),)
    assert extract_embeddings(man, CFG["backbone"], tmp / "e.npz", size=112).keys() == emb.keys()  # cache reused


def test_seed_determinism():
    from filmlook_ml.training.train import set_seed
    set_seed(3)
    a = torch.randn(3)
    set_seed(3)
    assert torch.equal(a, torch.randn(3))


def test_temperature_and_ece():
    torch.manual_seed(0)
    truth = (torch.rand(2000) > 0.5).float()
    logits = (truth * 2 - 1) * 6 + torch.randn(2000)        # very confident ...
    tgt = torch.where(torch.rand(2000) < 0.25, 1 - truth, truth)  # ... but 25% wrong -> over-confident
    t = fit_temperature(logits, tgt, torch.ones_like(tgt))
    assert t > 1.0
    p = torch.sigmoid(logits / t).numpy()
    assert expected_calibration_error(p, tgt.numpy()) <= expected_calibration_error(torch.sigmoid(logits).numpy(),
                                                                                    tgt.numpy()) + 1e-6


def test_auroc():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert auroc(np.array([0.5] * 4), y) == 0.5
    r = auroc_ci(np.random.default_rng(0).random(200), np.r_[np.zeros(100), np.ones(100)], n_boot=200)
    assert r["ci95"][0] < 0.5 < r["ci95"][1]
