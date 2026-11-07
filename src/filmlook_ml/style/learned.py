"""Learned style models over JS-reproducible features: linear (multinomial logistic), small random forest,
compact MLP. Each exports to a JSON dict that js/filmlook-style/engine.js evaluates (predictLearned), and
``predict_exported`` is the Python mirror used for parity tests.

Classes: profile ids + "none" (no reliable match). Labels must come from independent judgements (human or
authorized Claude teacher), never from the profile engine itself.
"""
from __future__ import annotations

import numpy as np

KINDS = ("linear", "forest", "mlp")


def fit(kind: str, X: np.ndarray, y: list[str], seed: int = 0, sample_weight=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    if kind == "linear":
        m = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
        m.fit(Z, y, sample_weight=sample_weight)
    elif kind == "forest":
        m = RandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=3, random_state=seed)
        m.fit(Z, y, sample_weight=sample_weight)
    elif kind == "mlp":
        m = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=2000, random_state=seed)
        m.fit(Z, y)
    else:
        raise ValueError(kind)
    return sc, m


def export(kind: str, sc, m) -> dict:
    classes = [str(c) for c in m.classes_]
    base = {"type": kind, "classes": classes, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
    if kind == "linear":
        coef, icpt = m.coef_, m.intercept_
        if coef.shape[0] == 1:  # binary -> two-logit form for softmax
            coef, icpt = np.vstack([-coef / 2, coef / 2]), np.array([-icpt[0] / 2, icpt[0] / 2])
        return {**base, "type": "linear", "coef": coef.tolist(), "intercept": icpt.tolist()}
    if kind == "forest":
        trees = []
        for est in m.estimators_:
            t = est.tree_
            val = t.value[:, 0, :]
            val = val / val.sum(1, keepdims=True)
            # sklearn's per-tree classes are the forest's class indices
            trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                          "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                          "value": val.tolist()})
        return {**base, "type": "forest", "trees": trees}
    if kind == "mlp":
        W = [w.tolist() for w in m.coefs_]
        b = [x.tolist() for x in m.intercepts_]
        if len(classes) == 2:  # logistic output -> two-logit softmax
            W[-1] = [[-r[0] / 2, r[0] / 2] for r in W[-1]]
            b[-1] = [-b[-1][0] / 2, b[-1][0] / 2]
        return {**base, "type": "mlp", "weights": W, "biases": b}
    raise ValueError(kind)


def _softmax(a):
    a = np.asarray(a, float)
    e = np.exp(a - a.max())
    return e / e.sum()


def predict_exported(v: np.ndarray, L: dict) -> np.ndarray:
    z = (np.asarray(v, float) - np.array(L["mean"])) / np.array(L["scale"])
    if L["type"] == "linear":
        return _softmax(np.array(L["intercept"]) + np.array(L["coef"]) @ z)
    if L["type"] == "forest":
        acc = np.zeros(len(L["classes"]))
        for T in L["trees"]:
            n = 0
            while T["left"][n] != -1:
                n = T["left"][n] if float(np.float32(z[T["feature"][n]])) <= T["threshold"][n] else T["right"][n]
            acc += np.array(T["value"][n])
        return acc / len(L["trees"])
    if L["type"] == "mlp":
        h = z
        for i, (W, b) in enumerate(zip(L["weights"], L["biases"])):
            h = h @ np.array(W) + np.array(b)
            if i < len(L["weights"]) - 1:
                h = np.maximum(h, 0)
        return _softmax(h)
    raise ValueError(L["type"])


# ------------------------------------------------------------------ evaluation against judgements
def ranking_metrics(ranked: list[list[str]], truth: list[set[str]]) -> dict:
    """ranked: per image, classes ordered best-first (may start with 'none'); truth: judged acceptable set."""
    t1 = np.mean([r[0] in t for r, t in zip(ranked, truth)])
    t3 = np.mean([bool(set(r[:3]) & t) for r, t in zip(ranked, truth)])
    rr = []
    for r, t in zip(ranked, truth):
        ranks = [k for k, c in enumerate(r) if c in t]
        rr.append(1.0 / (ranks[0] + 1) if ranks else 0.0)
    return {"n": len(truth), "top1": float(t1), "top3": float(t3), "mrr": float(np.mean(rr))}


def brier_ece(prob_top: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict:
    b = np.minimum((prob_top * bins).astype(int), bins - 1)
    ece = sum((b == k).mean() * abs(prob_top[b == k].mean() - correct[b == k].mean()) for k in range(bins) if (b == k).any())
    return {"ece": float(ece), "brier": float(((prob_top - correct) ** 2).mean())}
