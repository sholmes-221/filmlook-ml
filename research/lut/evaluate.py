"""Metric utilities shared by the LUT experiments: top-k, confusion, ECE, rejection AUROC, grouped bootstrap."""
from __future__ import annotations

import numpy as np


def topk_hits(probs: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-probs, axis=1)[:, :k]
    return (order == y[:, None]).any(1)


def topk_hits_sets(probs: np.ndarray, accept: list[set[int]], k: int) -> np.ndarray:
    """Hit if any of the top-k classes is in the per-sample acceptable set (line-level labels)."""
    order = np.argsort(-probs, axis=1)[:, :k]
    return np.array([any(int(c) in acc for c in row) for row, acc in zip(order, accept)])


def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def format_confusion(cm: np.ndarray, names: list[str], width: int = 6) -> str:
    head = " " * 14 + " ".join(f"{n[:width]:>{width}}" for n in names)
    rows = [f"{n[:13]:13} " + " ".join(f"{v:{width}d}" for v in cm[i]) for i, n in enumerate(names)]
    return "\n".join([head] + rows)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) by rank statistic (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def grouped_bootstrap(values: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` with a 95 % CI from resampling groups (scenes / uploaders) with replacement."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        stats.append(values[sel].mean())
    return float(values.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling by grid search on NLL (validation logits)."""
    best, best_nll = 1.0, np.inf
    for T in np.exp(np.linspace(np.log(0.3), np.log(5.0), 60)):
        p = softmax(logits, T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best
