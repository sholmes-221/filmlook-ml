"""Label storage + merging into training targets.

Priority (highest first): human > claude (only if usable_for_training and not excluded by flags)
> synthetic_ordering (partial) > heuristic. Mock labels are never used.
Capture-medium targets come from acquisition metadata (EXIF / scanner model), originals only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.labeling.schema import ImageLabel, PairLabel
from filmlook_ml.labeling.teachers import EXCLUDING_FLAGS

PRIORITY = ["human", "claude", "synthetic_ordering", "heuristic"]

LABEL_FILES = {
    "heuristic": "pseudo_heuristic.jsonl",
    "claude": "pseudo_claude.jsonl",
    "mock": "pseudo_mock.jsonl",
    "synthetic_ordering": "synthetic_labels.jsonl",
    "human": "human_labels.jsonl",
}
PAIR_FILES = {"synthetic_ordering": "synthetic_pairs.jsonl", "claude": "claude_pairs.jsonl",
              "human": "human_pairs.jsonl"}
FILM_MEDIA = {"film_scan"}
DIGITAL_MEDIA = {"digital_camera", "smartphone", "digital_filtered", "smartphone_filtered"}


def load_labels(label_dir: Path) -> dict[str, dict[str, ImageLabel]]:
    out: dict[str, dict[str, ImageLabel]] = {}
    for src, fn in LABEL_FILES.items():
        d: dict[str, ImageLabel] = {}
        for r in read_jsonl(Path(label_dir) / fn):
            lab = ImageLabel.model_validate(r)
            d[lab.image_id] = lab  # later lines (newer reviews) win
        out[src] = d
    return out


def load_pairs(label_dir: Path) -> list[PairLabel]:
    pairs = []
    for fn in PAIR_FILES.values():
        pairs += [PairLabel.model_validate(r) for r in read_jsonl(Path(label_dir) / fn)]
    return pairs


@dataclass
class Targets:
    ids: list[str]
    score: np.ndarray        # (N,) in [0,1]
    score_w: np.ndarray      # (N,)
    attrs: np.ndarray        # (N, A)
    attrs_w: np.ndarray      # (N, A) 0 = unsupervised
    medium: np.ndarray       # (N,) 1 = film capture, 0 = digital capture
    medium_w: np.ndarray     # (N,) 0 = unknown / edited
    source: list[str]        # highest-priority source per image (for reporting)


def medium_targets(manifest: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    m, w = np.zeros(len(manifest), np.float32), np.zeros(len(manifest), np.float32)
    for i, r in enumerate(manifest):
        if r.get("edit_type") != "original":
            continue
        if r.get("medium") in FILM_MEDIA:
            m[i], w[i] = 1.0, 1.0
        elif r.get("medium") in DIGITAL_MEDIA:
            m[i], w[i] = 0.0, 1.0
    return m, w


def build_targets(ids: list[str], labels: dict[str, dict[str, ImageLabel]], weights: dict[str, float],
                  manifest: list[dict] | None = None) -> Targets:
    n, A = len(ids), len(ATTRIBUTES)
    if manifest is not None:
        by_id = {r["image_id"]: r for r in manifest}
        med, med_w = medium_targets([by_id[i] for i in ids])
    else:
        med, med_w = np.zeros(n, np.float32), np.zeros(n, np.float32)
    t = Targets(ids, np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros((n, A), np.float32),
                np.zeros((n, A), np.float32), med, med_w, ["none"] * n)
    for i, iid in enumerate(ids):
        for src in reversed(PRIORITY):  # LOW -> HIGH priority so higher priority overwrites
            lab = labels.get(src, {}).get(iid)
            if lab is None or not lab.usable_for_training or lab.review_status == "rejected":
                continue
            if src != "human" and EXCLUDING_FLAGS & set(lab.flags):
                continue
            w = weights.get(src, 0.0)
            if w <= 0:
                continue
            sup = lab.supervised_attributes or ATTRIBUTES
            for j, a in enumerate(ATTRIBUTES):
                v = lab.label.attributes.get(a)
                if a in sup:
                    # null (not applicable, e.g. colour on monochrome) -> explicitly unsupervised
                    t.attrs[i, j], t.attrs_w[i, j] = (0.0, 0.0) if v is None else (v, w)
            if lab.supervised_attributes is None:
                t.score[i], t.score_w[i] = lab.label.film_score / 100.0, w
            t.source[i] = src
    return t


def pair_arrays(pairs: list[PairLabel], id_index: dict[str, int], weights: dict[str, float]):
    """-> (idx_a, idx_b, attr_idx (-1 = film_score), sign (+1 a wins), weight)."""
    rows = []
    for p in pairs:
        if p.winner == "tie" or p.image_a not in id_index or p.image_b not in id_index:
            continue
        ai = -1 if p.attribute == "film_score" else ATTRIBUTES.index(p.attribute)
        rows.append((id_index[p.image_a], id_index[p.image_b], ai, 1.0 if p.winner == "a" else -1.0,
                     weights.get(p.source, 0.0)))
    if not rows:
        return tuple(np.zeros(0, dt) for dt in (np.int64, np.int64, np.int64, np.float32, np.float32))
    a = np.array(rows, dtype=np.float64)
    return a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.int64), \
        a[:, 3].astype(np.float32), a[:, 4].astype(np.float32)


def append_label(label_dir: Path, source: str, lab: ImageLabel) -> None:
    p = Path(label_dir) / LABEL_FILES[source]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(lab.model_dump_json() + "\n")


def append_pair(label_dir: Path, pair: PairLabel) -> None:
    p = Path(label_dir) / PAIR_FILES[pair.source if pair.source in PAIR_FILES else "synthetic_ordering"]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(pair.model_dump_json() + "\n")


def pairs_by_source(label_dir: Path, id_index: dict[str, int], weights: dict[str, float]) -> dict:
    """Pair arrays grouped by label source (evaluation keeps synthetic / claude / human apart)."""
    by = {}
    for p in load_pairs(label_dir):
        by.setdefault(p.source, []).append(p)
    return {s: pair_arrays(ps, id_index, {s: 1.0}) for s, ps in by.items()}
