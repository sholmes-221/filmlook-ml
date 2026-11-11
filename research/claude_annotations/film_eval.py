"""Do Claude's annotations help the film-likeness estimate, judged against independently recorded capture medium?

  python -m research.claude_annotations.film_eval embed      # production-encoder vectors for the film / digital set (local, free)
  python -m research.claude_annotations.film_eval evaluate   # needs the `filmset` annotations

Labels: film scan (1) or digital capture (0) from scanner / camera EXIF and owners' records, the same set and the same
uploader-disjoint split as research/lut/film_likeness_train.py (the 36 measured statistics are its features.npy).
Claude never sees a label and no Claude output is used as one.

Readings on the held-out uploaders (AUROC, bootstrap over uploaders, paired difference to the 36-statistic model):
  measured36            the shipped estimate's features, logistic regression refitted here (the baseline to beat)
  embedding             DINOv2-small (production encoder) alone
  measured36+embedding
  claude_impression     Claude's 0-4 "film-like impression" alone, as a zero-shot judge (how good is the annotator?)
  measured36+claude     statistics + Claude's annotation vector at test time (an upper bound: not deployable)
  measured36+heads      statistics + the annotation fields PREDICTED by local ridge heads from statistics + embedding,
                        heads fitted on the training uploaders only (deployable: no API call per photograph)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.claude_annotations import schema as S  # noqa: E402
from research.claude_annotations.to_vectors import ridge  # noqa: E402

SET = ROOT / "data" / "lut" / "film_likeness"
EMB = ROOT / "data" / "embeddings" / "photographers" / "filmset_prod.npz"
OUT = ROOT / "reports" / "claude" / "film_eval.json"


def rows() -> list[dict]:
    return [json.loads(line) for line in (SET / "manifest.jsonl").read_text().splitlines()]


def embed() -> None:
    from PIL import Image

    from research.photographers.embed import prod_encoder
    enc = prod_encoder("cpu"); rs = rows(); X = []
    for i, r in enumerate(rs):
        X.append(enc([Image.open(r["path"])])[0])
        if i % 400 == 0:
            print(i, len(rs), flush=True)
    np.savez(EMB, X=np.stack(X).astype(np.float32), paths=np.array([r["path"] for r in rs]))
    print("saved", len(X))


def z(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train, *others)]


def auroc_ci(y: np.ndarray, s: np.ndarray, groups: np.ndarray, base: np.ndarray | None = None, n: int = 1000) -> dict:
    rng = np.random.default_rng(0); ug = np.unique(groups); members = {g: np.flatnonzero(groups == g) for g in ug}
    vals, diffs = [], []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[idx].tolist())) < 2:
            continue
        a = roc_auc_score(y[idx], s[idx]); vals.append(a)
        if base is not None:
            diffs.append(a - roc_auc_score(y[idx], base[idx]))
    out = {"auroc": round(float(roc_auc_score(y, s)), 4), "ci": [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]}
    if base is not None:
        out["delta_vs_measured36"] = round(out["auroc"] - float(roc_auc_score(y, base)), 4)
        out["delta_ci"] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    return out


def evaluate() -> None:
    rs = rows(); F = np.load(SET / "features.npy").astype(np.float32); F = np.nan_to_num(F)
    E = np.load(EMB); assert E["paths"].tolist() == [r["path"] for r in rs]
    X = E["X"]
    ann = P.annotated("filmset", P.cfg()["models"]["cheap"])
    ids = [f"filmset:{r['dataset']}:{Path(r['path']).name}" for r in rs]
    have = np.array([i in ann for i in ids])
    y = np.array([r["label"] for r in rs]); g = np.array([r["uploader"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs])
    report: dict = {"n": len(rs), "annotated": int(have.sum()), "train": int(tr.sum()), "test": int((~tr).sum()),
                    "test_film": int(y[~tr].sum()), "test_digital": int((1 - y[~tr]).sum()), "test_uploaders": int(len(set(g[~tr].tolist())))}

    def fit(Atr, Ate, C=0.3):
        a, b = z(Atr, Ate)
        return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(a, y[tr]).decision_function(b)

    base = fit(F[tr], F[~tr])
    res = {"measured36": auroc_ci(y[~tr], base, g[~tr]),
           "embedding": auroc_ci(y[~tr], fit(X[tr], X[~tr], 0.1), g[~tr], base),
           "measured36+embedding": auroc_ci(y[~tr], fit(np.hstack([F[tr], X[tr]]), np.hstack([F[~tr], X[~tr]]), 0.1), g[~tr], base)}
    if have.all():
        A = np.stack([S.vectorise(ann[i]["annotation"]) for i in ids]); names = S.feature_names()
        imp = A[:, names.index("film_like_impression")]
        res["claude_impression"] = auroc_ci(y[~tr], imp[~tr], g[~tr], base)
        res["measured36+claude"] = auroc_ci(y[~tr], fit(np.hstack([F[tr], A[tr]]), np.hstack([F[~tr], A[~tr]])), g[~tr], base)
        # local heads: annotation fields predicted from statistics + embedding, fitted on training uploaders, cross-fitted there
        Z = np.hstack(z(F[tr], F)[1:] + [X]); pred = np.zeros_like(A)
        tri, tei = np.flatnonzero(tr), np.flatnonzero(~tr)
        pred[tei] = ridge(Z[tri], A[tri], Z[tei], lam=10.0)
        ug = np.unique(g[tri]); fold = {u: i % 5 for i, u in enumerate(np.random.default_rng(0).permutation(ug))}; f = np.array([fold[u] for u in g[tri]])
        for k in range(5):
            pred[tri[f == k]] = ridge(Z[tri[f != k]], A[tri[f != k]], Z[tri[f == k]], lam=10.0)
        res["measured36+heads"] = auroc_ci(y[~tr], fit(np.hstack([F[tr], pred[tr]]), np.hstack([F[~tr], pred[~tr]])), g[~tr], base)
        res["measured36+embedding+heads"] = auroc_ci(y[~tr], fit(np.hstack([F[tr], X[tr], pred[tr]]), np.hstack([F[~tr], X[~tr], pred[~tr]]), 0.1), g[~tr], base)
        j = names.index("film_like_impression")
        report["heads"] = {"film_like_impression_pearson_r_on_test": round(float(np.corrcoef(A[tei, j], pred[tei, j])[0, 1]), 4)}
        fr = [i for i, nme in enumerate(names) if nme.startswith("film_rendering:")]
        report["film_rendering_rates"] = {names[i].split(":")[1]: {"film": round(float(A[y == 1, i].mean()), 3), "digital": round(float(A[y == 0, i].mean()), 3)} for i in fr}
        np.savez(ROOT / "dist" / "claude_heads_filmset.npz", note="ridge heads: z-scored 36 statistics + production embedding -> annotation vector; fitted on training uploaders",
                 feature_names=np.array(names)) if (ROOT / "dist").exists() else None
    else:
        report["note"] = "filmset annotations incomplete: only the local readings are reported"
    report["readings"] = res
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"embed": embed, "evaluate": evaluate}[sys.argv[1]]()
