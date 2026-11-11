"""Turn stored annotations into retrieval features, and measure what a local model can recover of them.

  python -m research.claude_annotations.to_vectors [--seed 0]

Writes data/embeddings/photographers/claude_v1.npz with
  X                    the annotation vector of every bank work (schema.vectorise; scan-condition fields left out);
  claude_v1_predicted  the same vector predicted from the production DINOv2 embedding by ridge heads. Heads are
                       fitted on the benchmark's bank split only (bank rows are predicted out of fold), so val and
                       test rows never contribute a label. This is the deployable form: no API call per upload.
and reports/claude/annotation_quality.json: coverage, per-field agreement between the cheap and the check model on
the works both annotated, and how well each field is predicted from the embedding on val + test rows.

Agreement between two Claude models is a consistency reading, not accuracy: neither is ground truth. Whether the
annotations are useful is decided only by research.photographers.evaluate against catalogue attribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.claude_annotations import schema as S  # noqa: E402
from research.photographers import evaluate as E  # noqa: E402


def ridge(Xa: np.ndarray, Ya: np.ndarray, Xb: np.ndarray, lam: float = 1.0) -> np.ndarray:
    mu_x, mu_y = Xa.mean(0), Ya.mean(0)
    A = (Xa - mu_x).T @ (Xa - mu_x) + lam * np.eye(Xa.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, (Xa - mu_x).T @ (Ya - mu_y))
    return (Xb - mu_x) @ W + mu_y


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    c = P.cfg(); cheap, check = c["models"]["cheap"], c["models"]["check"]
    A_cheap, A_check = P.annotated("bank", cheap), P.annotated("bank", check)
    rows = E.load_rows(); ids = [r["work_id"] for r in rows]
    have = [w for w in ids if w in A_cheap or w in A_check]
    report: dict = {"rows": len(ids), "annotated": len(have), "cheap": len(A_cheap), "check": len(A_check)}
    both = sorted(set(A_cheap) & set(A_check))
    if both:
        agree = {}
        for k in list(S.SINGLE) + S.ORDINAL:
            agree[k] = float(np.mean([A_cheap[w]["annotation"][k] == A_check[w]["annotation"][k] for w in both]))
        for k in S.ORDINAL:
            agree[k + "_within_1"] = float(np.mean([abs(int(A_cheap[w]["annotation"][k]) - int(A_check[w]["annotation"][k])) <= 1 for w in both]))
        for k in S.MULTI:
            j = [len(set(A_cheap[w]["annotation"][k]) & set(A_check[w]["annotation"][k])) / max(1, len(set(A_cheap[w]["annotation"][k]) | set(A_check[w]["annotation"][k]))) for w in both]
            agree[k + "_jaccard"] = float(np.mean(j))
        report["cheap_vs_check"] = {"n": len(both), "agreement": agree}
    out = ROOT / "reports" / "claude" / "annotation_quality.json"; out.parent.mkdir(parents=True, exist_ok=True)
    if len(have) < len(ids):
        report["note"] = "not every bank work is annotated yet; claude_v1.npz is written only for full coverage"
        out.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)); return
    # The cheap model's annotation is the feature; the check model replaces it only where the cheap one was invalid.
    X = np.stack([S.vectorise((A_cheap.get(w) or A_check[w])["annotation"]) for w in ids])
    emb = E.load_vectors("prod", ids)
    groups, _ = E.leakage_groups(rows, E.l2(emb[:, :384])); split = E.split_groups(rows, groups, a.seed)
    bank = np.flatnonzero(split == "bank"); rest = np.flatnonzero(split != "bank")
    pred = np.zeros_like(X)
    pred[rest] = ridge(emb[bank], X[bank], emb[rest])
    ug = np.unique(groups[bank]); fold = {g: i % 5 for i, g in enumerate(np.random.default_rng(a.seed).permutation(ug))}
    f = np.array([fold[g] for g in groups[bank]])
    for k in range(5):
        pred[bank[f == k]] = ridge(emb[bank[f != k]], X[bank[f != k]], emb[bank[f == k]])
    names = S.feature_names()
    quality = {}
    start = 0
    for field, values in S.SINGLE.items():
        if field in S.ARTEFACT_FIELDS:
            continue
        sl = slice(start, start + len(values)); start += len(values)
        truth, guess = X[rest][:, sl].argmax(1), pred[rest][:, sl].argmax(1)
        quality[field] = {"accuracy": float((truth == guess).mean()), "majority": float(np.bincount(truth).max() / len(truth))}
    start = len(names) - len(S.ORDINAL)
    for i, field in enumerate(S.ORDINAL):
        t, g = X[rest][:, start + i], pred[rest][:, start + i]
        quality[field] = {"pearson_r": float(np.corrcoef(t, g)[0, 1]) if t.std() > 0 else None}
    report["predicted_from_embedding_on_val_and_test"] = quality
    np.savez(E.EMB / "claude_v1.npz", work_ids=np.array(ids), X=X, claude_v1_predicted=pred.astype(np.float32), feature_names=np.array(names))
    out.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main()
