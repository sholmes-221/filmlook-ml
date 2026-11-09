#!/usr/bin/env python
"""Evaluate photographer matching on the reference database (methods A-D, open set, calibration, subject bias).

  python scripts/refdb_eval.py [--vector dinov2s_cls_224] [--min-known 60] [--seed 0]

Split (pre-registered here, before any result was seen):
- Known photographers: >= --min-known usable works. Unknown photographers (open set): the rest with >= 15 works, plus
  ``FORCED_UNKNOWN`` (one FSA photographer, so the open-set test includes a subject-matched stranger).
- Image-disjoint split of each known photographer's works: 60 / 20 / 20 (train / val / test) by a stable hash of a
  *session key* (photographer + year + first four title words), so frames from one sitting never straddle splits.
- Test is read once per run; val is used for every choice (k, C, temperature, rejection threshold, fusion weights).

Methods: A kNN (cosine, k from val, similarity-weighted vote); B mean prototype per photographer; C k-means
prototypes (k from val in {2,3,5}, max similarity); D multinomial logistic regression (C from val).
Semantic-only baseline: the same logistic regression on SigLIP image embeddings (a content model), and on the
zero-shot tag scores; if subject content alone predicts the photographer about as well as DINOv2, the DINOv2 result
is subject bias, not style.

Metrics: top-1, top-3, macro F1, retrieval precision@5/@10 (test queries against the train bank), confusion matrix,
per-photographer counts, ECE of temperature-scaled scores (temperature fitted on val), open-set AUROC and the
known-accuracy / unknown-rejection trade-off at the val-chosen threshold (95 % of val knowns accepted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.refdb import manifest  # noqa: E402

FORCED_UNKNOWN = ("john_vachon",)
OUT = P.REPORTS / "refdb"


def session_key(r: dict) -> str:
    words = re.findall(r"[a-z]+", (r["title"] or "").lower())[:4]
    return f"{r['photographer_id']}|{r['year'] or ''}|{' '.join(words)}"


def split_of(key: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 80 else "test")


def load(vector: str) -> tuple[list[dict], np.ndarray]:
    with manifest.connect() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT w.work_id, w.photographer_id, w.title, w.year, w.is_monochrome, e.vector FROM works w JOIN embeddings e ON e.work_id = w.work_id "
            "WHERE e.model = ? AND w.rights_status = 'eligible' AND w.duplicate_of IS NULL ORDER BY w.work_id", (vector,))]
    X = np.stack([np.frombuffer(r.pop("vector"), dtype=np.float32) for r in rows]) if rows else np.zeros((0, 384), np.float32)
    return rows, X


def topk_metrics(S: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    order = np.argsort(-S, axis=1)
    top1 = order[:, 0]
    return {"top1": float((top1 == y).mean()), "top3": float(np.mean([y[i] in order[i, :3] for i in range(len(y))])),
            "macro_f1": float(f1_score(y, top1, labels=list(range(len(classes))), average="macro", zero_division=0)), "n": int(len(y))}


def softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = S / T; Z = Z - Z.max(axis=1, keepdims=True); E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_temperature(S: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.003), np.log(10), 200))
    nll = [-np.mean(np.log(softmax(S, t)[np.arange(len(y)), y] + 1e-12)) for t in grid]
    return float(grid[int(np.argmin(nll))])


def ece(P_: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = P_.max(axis=1); pred = P_.argmax(axis=1); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


class Methods:
    """Each method maps embeddings to a score matrix over the known classes (higher = closer)."""

    def __init__(self, Xtr: np.ndarray, ytr: np.ndarray, n_classes: int):
        self.Xtr, self.ytr, self.K = Xtr, ytr, n_classes

    def knn(self, X: np.ndarray, k: int) -> np.ndarray:
        sims = X @ self.Xtr.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        S = np.zeros((len(X), self.K))
        for i in range(len(X)):
            for j in idx[i]:
                S[i, self.ytr[j]] += max(sims[i, j], 0)
        return S / k

    def prototype(self, X: np.ndarray) -> np.ndarray:
        protos = np.stack([self.Xtr[self.ytr == c].mean(0) for c in range(self.K)])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        return X @ protos.T

    def multi_prototype(self, X: np.ndarray, k: int, seed: int) -> np.ndarray:
        S = np.full((len(X), self.K), -1.0)
        for c in range(self.K):
            Xc = self.Xtr[self.ytr == c]
            km = KMeans(n_clusters=min(k, len(Xc)), n_init=10, random_state=seed).fit(Xc)
            P_ = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
            S[:, c] = (X @ P_.T).max(axis=1)
        return S

    def logreg(self, X: np.ndarray, C: float) -> tuple[np.ndarray, LogisticRegression]:
        m = LogisticRegression(C=C, max_iter=3000).fit(self.Xtr, self.ytr)
        return m.decision_function(X), m


def precision_at_k(Xq: np.ndarray, yq: np.ndarray, Xb: np.ndarray, yb: np.ndarray, k: int) -> float:
    idx = np.argsort(-(Xq @ Xb.T), axis=1)[:, :k]
    return float((yb[idx] == yq[:, None]).mean())


def evaluate(rows: list[dict], X: np.ndarray, min_known: int, seed: int) -> dict:
    counts = Counter(r["photographer_id"] for r in rows)
    known = sorted(p for p, n in counts.items() if n >= min_known and p not in FORCED_UNKNOWN)
    unknown = sorted(p for p, n in counts.items() if p not in known and n >= 15)
    cls = {p: i for i, p in enumerate(known)}
    split = np.array([split_of(session_key(r), seed) if r["photographer_id"] in cls else "unknown" for r in rows])
    y = np.array([cls.get(r["photographer_id"], -1) for r in rows])
    unk_mask = np.array([r["photographer_id"] in unknown for r in rows])
    tr, va, te = split == "train", split == "val", split == "test"
    M = Methods(X[tr], y[tr], len(known))

    # Choose hyper-parameters on val.
    k_best = max((3, 5, 10, 20), key=lambda k: topk_metrics(M.knn(X[va], k), y[va], known)["top1"])
    mk_best = max((2, 3, 5), key=lambda k: topk_metrics(M.multi_prototype(X[va], k, seed), y[va], known)["top1"])
    C_best = max((0.1, 0.3, 1.0, 3.0, 10.0), key=lambda C: topk_metrics(M.logreg(X[va], C)[0], y[va], known)["top1"])
    score_fns = {
        "A_knn": lambda Z: M.knn(Z, k_best),
        "B_prototype": M.prototype,
        "C_multi_prototype": lambda Z: M.multi_prototype(Z, mk_best, seed),
        "D_logreg": lambda Z: M.logreg(Z, C_best)[0],
    }
    out = {"known": known, "unknown": unknown, "counts": {p: dict(Counter(split[[r['photographer_id'] == p for r in rows]])) for p in counts},
           "chosen": {"knn_k": k_best, "multi_prototype_k": mk_best, "logreg_C": C_best}, "methods": {}}
    for name, fn in score_fns.items():
        Sva, Ste, Sunk = fn(X[va]), fn(X[te]), fn(X[unk_mask]) if unk_mask.any() else None
        T = fit_temperature(Sva, y[va])
        Pte = softmax(Ste, T)
        res = topk_metrics(Ste, y[te], known)
        res["ece_temperature_scaled"] = ece(Pte, y[te]); res["temperature"] = T
        conf_mat = np.zeros((len(known), len(known)), int)
        for a, b in zip(y[te], Ste.argmax(1)):
            conf_mat[a, b] += 1
        res["confusion"] = conf_mat.tolist()
        res["per_class_recall"] = {p: float(conf_mat[i, i] / max(conf_mat[i].sum(), 1)) for p, i in cls.items()}
        if Sunk is not None:
            Pva = softmax(Sva, T)
            thr = float(np.quantile(Pva.max(1), 0.05))  # accept 95 % of val knowns
            conf_known, conf_unk = Pte.max(1), softmax(Sunk, T).max(1)
            res["open_set"] = {
                "auroc_known_vs_unknown": float(roc_auc_score(np.r_[np.ones(len(conf_known)), np.zeros(len(conf_unk))], np.r_[conf_known, conf_unk])),
                "threshold_from_val": thr, "known_accepted": float((conf_known >= thr).mean()),
                "known_accepted_and_correct": float(((conf_known >= thr) & (Ste.argmax(1) == y[te])).mean()),
                "unknown_rejected": float((conf_unk < thr).mean()), "n_unknown": int(len(conf_unk)),
                "unknown_assigned_to": dict(Counter(known[i] for i in Sunk.argmax(1)[conf_unk >= thr])),
            }
        out["methods"][name] = res
    out["retrieval"] = {f"p@{k}": precision_at_k(X[te], y[te], X[tr], y[tr], k) for k in (5, 10)}
    out["chance_top1"] = float(max(Counter(y[te]).values()) / max(te.sum(), 1))
    out["uniform_chance_top1"] = 1.0 / len(known)
    # Monochrome confound: accuracy of a classifier that only knows whether a print is monochrome.
    mono = np.array([r["is_monochrome"] or 0 for r in rows], dtype=float)[:, None]
    lr = LogisticRegression(max_iter=1000).fit(mono[tr], y[tr])
    out["monochrome_only_top1"] = float((lr.predict(mono[te]) == y[te]).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--vector", default="dinov2s_cls_224"); ap.add_argument("--min-known", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--extra", default="", help="extra vector file (npz with work_id, X) to evaluate, e.g. SigLIP")
    a = ap.parse_args()
    rows, X = load(a.vector)
    if a.extra:
        z = np.load(a.extra, allow_pickle=True); idx = {w: i for i, w in enumerate(z["work_id"])}
        keep = [i for i, r in enumerate(rows) if r["work_id"] in idx]
        rows = [rows[i] for i in keep]; X = np.stack([z["X"][idx[r["work_id"]]] for r in rows]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    res = evaluate(rows, X, a.min_known, a.seed)
    res["vector"] = a.extra or a.vector; res["n_rows"] = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(a.extra).stem if a.extra else a.vector
    (OUT / f"eval_{name}_seed{a.seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {m: {kk: v for kk, v in r.items() if kk in ("top1", "top3", "macro_f1", "ece_temperature_scaled")} | ({"open_auroc": r["open_set"]["auroc_known_vs_unknown"], "unk_rejected": r["open_set"]["unknown_rejected"]} if "open_set" in r else {}) for m, r in res["methods"].items()} for k in ("x",)}["x"], indent=1))
    print("retrieval", res["retrieval"], "chance", res["chance_top1"], "uniform", res["uniform_chance_top1"], "mono-only", res["monochrome_only_top1"])
    print("known", res["known"], "unknown", res["unknown"])


if __name__ == "__main__":
    main()
