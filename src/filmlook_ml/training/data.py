"""Assemble TrainData from manifest + labels + cached embeddings/features."""
from __future__ import annotations

import json

import numpy as np

from filmlook_ml import paths as P
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.labeling.pipeline import extract_features, feature_matrix
from filmlook_ml.labeling.store import build_targets, load_labels, load_pairs, pair_arrays
from filmlook_ml.training.train import TrainData, extract_embeddings

MANIFEST = P.MANIFESTS / "manifest.jsonl"
SPLITS = P.SPLITS / "splits.json"


def load_manifest() -> list[dict]:
    """Manifest rows minus quarantined images and families (configs/quarantine.yaml)."""
    from filmlook_ml.datasets.quarantine import filter_manifest
    return filter_manifest(_load_manifest_raw())


def _load_manifest_raw() -> list[dict]:
    rows = read_jsonl(MANIFEST)
    if not rows:
        raise SystemExit(f"empty manifest {MANIFEST}; run scripts/prepare_dataset.py first")
    return rows


def embedding_cache(size: int, max_side: int = 1280):
    return P.EMBEDDINGS / f"dinov2_small_{size}_ms{max_side}.npz"


def feature_cache(max_side: int, crop: int):
    return P.EMBEDDINGS / f"features_ms{max_side}_c{crop}.json"


def load_train_data(train_cfg: dict, label_cfg: dict, workers: int = 8, with_emb: bool = True,
                    emb_size: int = 224, feat_max_side: int | None = None) -> tuple[TrainData, list[dict]]:
    """Every training / evaluation entry point goes through here, so the compliance preflight runs first
    (docs/AI_DATA_POLICY.md). It raises SystemExit when provenance or permissions are missing."""
    from filmlook_ml.compliance.preflight import require_training_preflight
    require_training_preflight(label_cfg)
    manifest = load_manifest()
    split = json.loads(SPLITS.read_text())
    ids = [r["image_id"] for r in manifest]
    ms = feat_max_side or train_cfg["max_side"]
    feats = extract_features(manifest, feature_cache(ms, train_cfg["texture_crop"]), train_cfg["texture_crop"],
                             workers, ms)
    emb = None
    if with_emb:
        e = extract_embeddings(manifest, train_cfg["backbone"], embedding_cache(emb_size, train_cfg["max_side"]),
                               size=emb_size, max_side=train_cfg["max_side"])
        emb = np.stack([e[i] for i in ids]).astype(np.float32)
    labels = load_labels(P.LABELS)
    t = build_targets(ids, labels, label_cfg["source_weights"], manifest)
    pairs = pair_arrays(load_pairs(P.LABELS), {i: k for k, i in enumerate(ids)}, label_cfg["source_weights"])
    mp = train_cfg.get("medium_pairs", {})
    if mp.get("enabled"):
        pairs = _concat_pairs(pairs, medium_pairs(manifest, np.array([split[i] for i in ids]), mp.get("per_film", 4),
                                                  mp.get("weight", 0.5), mp.get("seed", 0)))
    data = TrainData(ids=ids, emb=emb, feats=feature_matrix(ids, feats), t=t,
                     split=np.array([split[i] for i in ids]), pairs=pairs)
    return data, manifest


def medium_pairs(manifest: list[dict], split: np.ndarray, per_film: int = 4, weight: float = 0.5, seed: int = 0):
    """Cross-image film_score pairs from capture metadata: a real film-scan ORIGINAL should score higher than
    an unfiltered digital ORIGINAL. Pairs are drawn within each split only (no leakage). Filtered digital photos
    are not used as negatives: a convincing film simulation may legitimately score high."""
    rng = np.random.default_rng(seed)
    ia, ib = [], []
    for sp in ("train", "val", "test"):
        film = [k for k, r in enumerate(manifest) if split[k] == sp and r["edit_type"] == "original"
                and r.get("medium") == "film_scan"]
        dig = [k for k, r in enumerate(manifest) if split[k] == sp and r["edit_type"] == "original"
               and r.get("medium") in ("digital_camera", "smartphone")]
        if not film or not dig:
            continue
        for f in film:
            for d in rng.choice(dig, min(per_film, len(dig)), replace=False):
                ia.append(f)
                ib.append(int(d))
    n = len(ia)
    return (np.array(ia, np.int64), np.array(ib, np.int64), np.full(n, -1, np.int64), np.ones(n, np.float32),
            np.full(n, weight, np.float32))


def _concat_pairs(a, b):
    return tuple(np.concatenate([x, y]).astype(x.dtype) for x, y in zip(a, b))
