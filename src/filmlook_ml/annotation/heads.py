"""Photographer-style and film-aesthetic heads on frozen DINOv2-small embeddings, with baselines.

Methods, compared on the same uploader-grouped splits:
  colour_texture : logistic regression on the 28 measured statistics (filmlook_infer.features)
  knn_dinov2     : cosine k-nearest-neighbours on DINOv2-small embeddings
  linear_dinov2  : logistic regression head on DINOv2-small embeddings (the lightweight classifier)

Each head is trained separately (``photographer`` / ``film``). Labels are weak (model-generated); they are
not ground truth.
- Soft targets: the primary label gets weight 1 x self-reported confidence; each alternative gets
  ALT_WEIGHT x that.
- ``unknown`` rows are excluded from that head. ``none_of_the_above`` is a real class, so a head can decline to
  match.
- Only ``training_eligible`` annotations are used: permitted provenance and no pending review.
- A readiness gate refuses to train or report when classes are too sparse.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from filmlook_ml.annotation import taxonomy as T

ALT_WEIGHT = 0.5
MIN_TRAIN_PER_CLASS = 8
MIN_EVAL_PER_CLASS = 2
MIN_CLASSES = 3


@dataclass
class HeadData:
    head: str
    classes: list[str]
    split: dict[str, np.ndarray]          # name -> indices into rows
    y: np.ndarray                         # primary label index per row
    alts: list[list[int]]                 # alternative label indices per row
    conf: np.ndarray
    emb: np.ndarray
    feats: np.ndarray
    ids: list[str]


def _call(a, head):
    return a.photographer_style if head == "photographer" else a.film_aesthetic


def readiness(anns, head: str, splits: dict[str, str]) -> dict:
    """Per-class counts by split and whether the head can be trained and evaluated meaningfully."""
    counts = {s: Counter() for s in ("train", "val", "test")}
    for a in anns:
        if not a.training_eligible:
            continue
        p = _call(a, head).primary
        if p != T.UNKNOWN:
            counts[splits.get(a.image_id, "train")][p] += 1
    trainable = [c for c in counts["train"] if counts["train"][c] >= MIN_TRAIN_PER_CLASS
                 and counts["val"][c] >= MIN_EVAL_PER_CLASS and counts["test"][c] >= MIN_EVAL_PER_CLASS]
    ok = len(trainable) >= MIN_CLASSES and any(c != T.NONE for c in trainable)
    return {"head": head, "counts": {s: dict(c) for s, c in counts.items()}, "trainable_classes": sorted(trainable),
            "ready": ok, "rule": f">= {MIN_TRAIN_PER_CLASS} train and >= {MIN_EVAL_PER_CLASS} val/test examples per "
                                 f"class, >= {MIN_CLASSES} such classes incl. one real style"}


def build(anns, head: str, records: dict, emb_by_sha: dict, feat_by_sha: dict, classes: list[str]) -> HeadData:
    from filmlook_infer.features import feature_vector
    idx = {c: i for i, c in enumerate(classes)}
    rows = [a for a in anns if a.training_eligible and _call(a, head).primary in idx
            and a.sha256 in emb_by_sha and a.sha256 in feat_by_sha]
    split = {s: np.array([i for i, a in enumerate(rows) if records[a.image_id]["split"] == s], dtype=int)
             for s in ("train", "val", "test")}
    return HeadData(head=head, classes=classes, split=split,
                    y=np.array([idx[_call(a, head).primary] for a in rows], dtype=int),
                    alts=[[idx[x] for x in _call(a, head).alternatives if x in idx] for a in rows],
                    conf=np.array([_call(a, head).confidence for a in rows], dtype=float),
                    emb=np.stack([emb_by_sha[a.sha256] for a in rows]) if rows else np.zeros((0, 768)),
                    feats=np.stack([feature_vector(feat_by_sha[a.sha256]) for a in rows]) if rows else np.zeros((0, 28)),
                    ids=[a.image_id for a in rows])


def soft_rows(d: HeadData, idx: np.ndarray, X: np.ndarray):
    """Expand to (x, label, weight) rows: primary weight = confidence, alternatives ALT_WEIGHT x confidence."""
    xs, ys, ws = [], [], []
    for i in idx:
        xs.append(X[i]); ys.append(d.y[i]); ws.append(max(d.conf[i], 0.05))
        for a in d.alts[i]:
            xs.append(X[i]); ys.append(a); ws.append(ALT_WEIGHT * max(d.conf[i], 0.05))
    return np.array(xs), np.array(ys), np.array(ws)


class Method:
    def __init__(self, name: str, kind: str, k: int = 5):
        self.name, self.kind, self.k = name, kind, k

    def _X(self, d: HeadData):
        return d.feats if self.kind == "features" else d.emb

    def fit(self, d: HeadData):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = self._X(d)
        tr = d.split["train"]
        if self.kind == "knn":
            Z = X[tr] / (np.linalg.norm(X[tr], axis=1, keepdims=True) + 1e-8)
            self.bank, self.bank_y, self.bank_w = Z, d.y[tr], d.conf[tr]
            return self
        self.scaler = StandardScaler().fit(X[tr])
        xs, ys, ws = soft_rows(d, tr, self.scaler.transform(X))
        self.clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(xs, ys, sample_weight=ws)
        return self

    def proba(self, d: HeadData, idx: np.ndarray) -> np.ndarray:
        X = self._X(d)[idx]
        n_cls = len(d.classes)
        if self.kind == "knn":
            Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            sims = Z @ self.bank.T
            P = np.zeros((len(idx), n_cls))
            for r, s in enumerate(sims):
                nn = np.argsort(-s)[: self.k]
                for j in nn:
                    P[r, self.bank_y[j]] += max(s[j], 0) * self.bank_w[j]
            return P / np.clip(P.sum(1, keepdims=True), 1e-8, None)
        P = np.zeros((len(idx), n_cls))
        P[:, self.clf.classes_] = self.clf.predict_proba(self.scaler.transform(X))
        return P

    def size_bytes(self) -> int:
        if self.kind == "knn":
            return int(self.bank.nbytes + self.bank_y.nbytes)
        return int(self.clf.coef_.nbytes + self.clf.intercept_.nbytes + self.scaler.mean_.nbytes * 2)


def evaluate(P: np.ndarray, d: HeadData, idx: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    y = d.y[idx]
    pred = P.argmax(1)
    labels = list(range(len(d.classes)))
    present = sorted(set(y) | set(pred))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    top3 = np.argsort(-P, axis=1)[:, :3]
    none = d.classes.index(T.NONE) if T.NONE in d.classes else -1
    lenient = [y[i] == pred[i] or pred[i] in d.alts[idx[i]] for i in range(len(idx))]
    return {
        "n": int(len(idx)),
        "macro_f1_present_classes": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "top1": float(np.mean(pred == y)), "top3": float(np.mean([y[i] in top3[i] for i in range(len(idx))])),
        "top1_matches_primary_or_alternative": float(np.mean(lenient)),
        "per_class": {d.classes[c]: {"precision": float(pr[c]), "recall": float(rc[c]), "support": int(sup[c])}
                      for c in labels if sup[c] or (pred == c).any()},
        "confusion": {"labels": [d.classes[c] for c in present],
                      "matrix": confusion_matrix(y, pred, labels=present).tolist()},
        "none_of_the_above": None if none < 0 else {
            "true_none": int((y == none).sum()), "pred_none": int((pred == none).sum()),
            "recall_none": float(((pred == none) & (y == none)).sum() / max((y == none).sum(), 1)),
            "false_match_rate_on_none": float(((pred != none) & (y == none)).sum() / max((y == none).sum(), 1))},
    }


def run_head(d: HeadData, methods: list[Method]) -> dict:
    out = {}
    for m in methods:
        m.fit(d)
        res = {"model_size_bytes": m.size_bytes()}
        for s in ("val", "test"):
            if len(d.split[s]):
                t0 = time.perf_counter()
                P = m.proba(d, d.split[s])
                res[f"latency_ms_per_image_{s}"] = 1000 * (time.perf_counter() - t0) / len(d.split[s])
                res[s] = evaluate(P, d, d.split[s])
        out[m.name] = res
    return out


METHODS = lambda: [Method("colour_texture", "features"), Method("knn_dinov2", "knn"),  # noqa: E731
                   Method("linear_dinov2", "embedding")]
