"""First reproducible DINOv2-small baseline for the main FilmLook model (frozen backbone, no LLM at inference).

Stage A: retrieval on cached DINOv2-small embeddings (CLS + mean patch, 224 px, the same loader as
inference), mean-centred on the train split and L2-normalised.
Stage B: lightweight logistic heads on those embeddings, with labels that need no model annotation:
  - capture_type: EXIF-derived capture medium (metadata);
  - temperature / saturation / contrast / key / grain / sharpness: classes defined by fixed thresholds on
    deterministic pixel statistics (filmlook_infer.style). These test whether the embedding carries the
    attribute. Applying the formula directly gets them right by definition, so they are not perceptual
    ground truth.
Every head is compared with (a) logistic regression on the 28 measured statistics and (b) the majority class,
on the uploader-grouped test split. Hyper-parameters are chosen on validation only; training-set metrics are
never reported.

Stages C/D (photographer, film aesthetic) are not trained here: no curated reference images and no reliable
labels exist yet (see reports/dinov2_training_strategy.md). The structured output reports them as
``not_available`` rather than guessing.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

BASELINE_VERSION = "dinov2-baseline-v0"
EMB_FILE = "dinov2_small_224_ms1280.npz"
FEAT_FILE = "features_ms1280_c512.json"
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
CAPTURE_CLASSES = ["film_scan", "digital_camera", "smartphone", "digital_filtered", "smartphone_filtered"]


# ----------------------------------------------------------------------------- labels
def attribute_labels(f: dict) -> dict[str, str]:
    """Deterministic attribute classes from measured statistics (same thresholds as filmlook_infer.style)."""
    from filmlook_infer.style import THRESHOLDS as th
    mono = bool(f["is_mono"])
    if mono:
        temp = sat = "monochrome"
    else:
        temp = "warm" if f["warmth"] > th["warm"] else "cool" if f["warmth"] < th["cool"] else "neutral"
        sat = "low" if f["sat_mean"] < th["muted"] else "high" if f["sat_mean"] > th["vivid"] else "medium"
    return {
        "temperature": temp, "saturation": sat,
        "contrast": "high" if f["y_std"] > th["high_contrast"] else "low" if f["y_std"] < th["low_contrast"] else "medium",
        "key": "high_key" if f["y_mean"] > th["high_key"] else "low_key" if f["y_mean"] < th["low_key"] else "balanced",
        "grain": "none" if f["noise_std"] < th["grain_none"] else "coarse" if f["noise_std"] > th["grain_coarse"] else "fine",
        "sharpness": "soft" if f["lap_var"] < th["soft_lap"] else "crisp" if f["lap_var"] > th["crisp_lap"] else "moderate",
    }


# ----------------------------------------------------------------------------- data
def load(records: list[dict], emb_path: Path, feat_path: Path):
    from filmlook_infer.features import feature_vector
    z = np.load(emb_path)
    E = dict(zip(z["sha"].tolist(), z["emb"]))
    F = json.loads(Path(feat_path).read_text())
    rows = [r for r in records if r["sha256"] in E and r["sha256"] in F]
    emb = np.stack([E[r["sha256"]] for r in rows]).astype(np.float32)
    feats = np.stack([feature_vector(F[r["sha256"]]) for r in rows]).astype(np.float32)
    return rows, emb, feats, E, F


def normalise(emb: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = emb[train_mask].mean(0)
    x = emb - mu
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8), mu


# ----------------------------------------------------------------------------- Stage A: retrieval
def retrieval_eval(Z: np.ndarray, rows: list[dict], label_key, bank_mask: np.ndarray, query_mask: np.ndarray,
                   ks=(1, 5, 10)) -> dict:
    """Precision@k of retrieving the same label from the bank (no query is in the bank)."""
    lab = np.array([label_key(r) for r in rows], dtype=object)
    B, Q = np.where(bank_mask)[0], np.where(query_mask)[0]
    S = Z[Q] @ Z[B].T
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hits = (lab[B][order[:, :k]] == lab[Q][:, None]).mean(1)
        out[f"precision@{k}"] = round(float(hits.mean()), 4)
    prior = Counter(lab[B])
    out["chance_precision"] = round(float(np.mean([prior[x] / len(B) for x in lab[Q]])), 4)
    out["n_queries"], out["n_bank"] = int(len(Q)), int(len(B))
    return out


def same_group_retrieval(Z: np.ndarray, rows: list[dict], k: int = 10) -> dict:
    """Leave-one-out: does a query retrieve another photo by the same uploader (authorship proxy)?"""
    g = np.array([r.get("group_id") for r in rows], dtype=object)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    hits, rr, chance, n = [], [], [], 0
    for i in range(len(rows)):
        same = (g == g[i])
        same[i] = False
        if not same.any() or rows[i]["dataset_id"] != "commoncatalog_ccby":
            continue
        n += 1
        order = np.argsort(-S[i])
        ranks = np.where(same[order])[0]
        hits.append(ranks[0] < k)
        rr.append(1.0 / (ranks[0] + 1))
        chance.append(1 - np.prod([(len(rows) - 1 - same.sum() - j) / (len(rows) - 1 - j) for j in range(k)]))
    return {f"hit@{k}": round(float(np.mean(hits)), 4), "mrr": round(float(np.mean(rr)), 4),
            f"chance_hit@{k}": round(float(np.mean(chance)), 4), "n_queries": n}


def edit_invariance(Z_orig: np.ndarray, rows: list[dict], E: dict, mu: np.ndarray, manifest: list[dict],
                    query_mask: np.ndarray) -> dict:
    """Query with an EDITED derivative (film emulation, grain, ...): rank of its own original among all originals.
    Rank 1 almost always = the embedding is dominated by content, not by the rendering edits change."""
    idx = {r["family_id"]: i for i, r in enumerate(rows)}
    ranks, per_edit = [], {}
    for m in manifest:
        if m["edit_type"] == "original" or m["family_id"] not in idx or not query_mask[idx[m["family_id"]]]:
            continue
        if m["sha256"] not in E:
            continue
        q = E[m["sha256"]] - mu
        q = q / (np.linalg.norm(q) + 1e-8)
        s = Z_orig @ q
        rank = int((s > s[idx[m["family_id"]]]).sum()) + 1
        ranks.append(rank)
        per_edit.setdefault(m["edit_type"], []).append(rank == 1)
    return {"queries": len(ranks), "own_original_rank1": round(float(np.mean([r == 1 for r in ranks])), 4),
            "by_edit_rank1": {k: round(float(np.mean(v)), 3) for k, v in sorted(per_edit.items())}}


# ----------------------------------------------------------------------------- Stage B: heads
def _fit_logreg(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


def _metrics(y, P, classes, groups=None, n_boot=300, seed=0) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    pred = np.array(classes)[P.argmax(1)]
    pr, rc, _, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    present = [c for c in classes if (y == c).any()]
    mf1 = f1_score(y, pred, labels=present, average="macro", zero_division=0)
    k = min(3, len(classes) - 1)
    topk = np.argsort(-P, axis=1)[:, :k]
    yi = np.array([classes.index(v) for v in y])
    out = {"n": int(len(y)), "macro_f1": round(float(mf1), 4), "accuracy": round(float((pred == y).mean()), 4),
           f"top{k}": round(float(np.mean([yi[i] in topk[i] for i in range(len(y))])), 4),
           "per_class": {c: {"precision": round(float(pr[j]), 3), "recall": round(float(rc[j]), 3),
                             "support": int(sup[j])} for j, c in enumerate(classes)},
           "confusion": {"labels": classes, "matrix": confusion_matrix(y, pred, labels=classes).tolist()}}
    if groups is not None and n_boot:
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        gi = {g: np.where(groups == g)[0] for g in ug}
        vals = []
        for _ in range(n_boot):
            idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
            vals.append(f1_score(y[idx], pred[idx], labels=present, average="macro", zero_division=0))
        out["macro_f1_ci95"] = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    return out


def run_head(name: str, y_all: np.ndarray, inputs: dict[str, np.ndarray], split: np.ndarray, groups: np.ndarray,
             Z: np.ndarray | None = None) -> dict:
    """Fit each input representation on train, choose C on val (macro-F1), report test."""
    from sklearn.metrics import f1_score
    mask = y_all != None  # noqa: E711
    tr, va, te = (split == "train") & mask, (split == "val") & mask, (split == "test") & mask
    classes = sorted(set(y_all[tr]))
    res = {"classes": classes, "train_counts": dict(Counter(y_all[tr])), "test_counts": dict(Counter(y_all[te]))}
    fitted = {}
    for rep, X in inputs.items():
        best = None
        for C in C_GRID:
            sc, clf = _fit_logreg(X[tr], y_all[tr], C)
            f = f1_score(y_all[va], clf.predict(sc.transform(X[va])), average="macro", zero_division=0)
            if best is None or f > best[0]:
                best = (f, C, sc, clf)
        f, C, sc, clf = best
        P = np.zeros((te.sum(), len(classes)))
        P[:, [classes.index(c) for c in clf.classes_]] = clf.predict_proba(sc.transform(X[te]))
        t0 = time.perf_counter()
        clf.predict_proba(sc.transform(X[te][:1]))
        lat = (time.perf_counter() - t0) * 1000
        res[rep] = {"C": C, "val_macro_f1": round(float(f), 4), "test": _metrics(y_all[te], P, classes, groups[te]),
                    "head_latency_ms": round(lat, 3), "head_params": int(clf.coef_.size + clf.intercept_.size)}
        fitted[rep] = (sc, clf)
    if Z is not None:   # kNN on normalised embeddings (retrieval-based classifier)
        S = Z[te] @ Z[tr].T
        nn = np.argsort(-S, axis=1)[:, :10]
        ytr = y_all[tr]
        P = np.zeros((te.sum(), len(classes)))
        for i in range(len(nn)):
            for j in nn[i]:
                P[i, classes.index(ytr[j])] += 1
        res["knn10_dinov2"] = {"test": _metrics(y_all[te], P, classes, groups[te])}
    maj = Counter(y_all[tr]).most_common(1)[0][0]
    P = np.zeros((te.sum(), len(classes)))
    P[:, classes.index(maj)] = 1
    res["majority"] = {"test": _metrics(y_all[te], P, classes, groups[te], n_boot=0)}
    return res, fitted, classes


# ----------------------------------------------------------------------------- artifacts + inference
def save_heads(out_dir: Path, heads: dict, mu: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"embedding_mean": mu}
    for name, (sc, clf, classes) in heads.items():
        arrays[f"{name}.coef"], arrays[f"{name}.intercept"] = clf.coef_, clf.intercept_
        arrays[f"{name}.scale_mean"], arrays[f"{name}.scale_std"] = sc.mean_, sc.scale_
        meta.setdefault("heads", {})[name] = {"classes": [str(c) for c in clf.classes_]}
    np.savez(out_dir / "heads.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


class BaselinePredictor:
    """Structured predictions from a DINOv2-small embedding. Deterministic; no text generation, no LLM."""

    def __init__(self, ckpt: Path):
        self.meta = json.loads((Path(ckpt) / "meta.json").read_text())
        z = np.load(Path(ckpt) / "heads.npz")
        self.mu = z["embedding_mean"]
        self.heads = {n: (z[f"{n}.coef"], z[f"{n}.intercept"], z[f"{n}.scale_mean"], z[f"{n}.scale_std"], h["classes"])
                      for n, h in self.meta["heads"].items()}

    def predict_embedding(self, emb: np.ndarray) -> dict:
        x = emb - self.mu
        x = x / (np.linalg.norm(x) + 1e-8)
        attrs = {}
        for n, (W, b, m, s, classes) in self.heads.items():
            logits = ((x - m) / s) @ W.T + b
            if W.shape[0] == 1:                            # binary logistic regression
                p1 = 1 / (1 + np.exp(-logits[0]))
                p = np.array([1 - p1, p1])
            else:
                e = np.exp(logits - logits.max())
                p = e / e.sum()
            order = np.argsort(-p)
            attrs[n] = {"label": classes[order[0]], "scores": {classes[i]: round(float(p[i]), 4) for i in order[:3]}}
        na = {"status": "not_available", "reason": "no validated reference set or reliable labels yet"}
        return {"model_version": self.meta["baseline_version"], "taxonomy_version": self.meta.get("taxonomy_version"),
                "visual_attributes": attrs, "photographer_style": na, "film_aesthetic": na,
                "note": "scores are uncalibrated head outputs, not probabilities; attribute heads are trained on "
                        "deterministic measured classes, see meta.json"}
