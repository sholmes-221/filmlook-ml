"""Linear concept heads on the photo-gate embedding: fit on train, choose C and threshold on val, read test once.

  python -m research.concepts.train            -> reports/concepts/heads_results.json, data/concepts/heads_model.npz

Per head: logistic regression (sklearn lbfgs, class_weight balanced) on rows labelled 0/1 for that head (-1 excluded),
C chosen on validation average precision from C_GRID. Threshold: the lowest score t such that every threshold >= t
keeps validation precision >= TARGET_P with at least MIN_FIRES fires (a monotone envelope, so the operating point is
not a lucky spike). Test: precision with Wilson 95 % interval, recall, FPR on the generic random photographs, and the
precision implied at the natural prevalence of the generic set. Ship rule (pre-registered in E60): test precision
>= 0.90, Wilson lower bound >= 0.80, recall >= 0.30, >= 20 test positives.

Near-duplicates: a val/test image whose embedding is within cosine 0.95 (on both halves) of any image in an earlier split
(train before val before test), or whose 64-bit perceptual hash is within Hamming 4, is dropped before anything is fit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from research.concepts.common import DATA, REPORTS
from research.concepts.concept_map import CONCEPTS

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
TARGET_P, MIN_FIRES = 0.95, 20
SHIP_P, SHIP_LB, SHIP_R, SHIP_N = 0.90, 0.80, 0.30, 20


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - hw), float(c + hw)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    sel = pd.read_parquet(DATA / "selection.parquet").set_index("image_id")
    f = np.load(DATA / "feats_dinov2s.npz")
    ids = list(f["ids"])
    X = f["X"].astype(np.float32)
    rows = sel.loc[ids].reset_index()
    return rows, X


def dedup(rows: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """Boolean keep-mask: drop val/test near-duplicates of earlier-split images (embedding and perceptual hash)."""
    import imagehash
    from PIL import Image
    keep = np.ones(len(rows), bool)
    order = {"train": 0, "val": 1, "test": 2}
    s = rows.split.map(order).to_numpy()
    for later in (1, 2):
        ref = np.where(s < later)[0]
        cand = np.where(s == later)[0]
        R = X[ref]
        for b in range(0, len(cand), 2048):
            c = cand[b:b + 2048]
            sim_cls = X[c, :384] @ R[:, :384].T
            sim_pm = X[c, 384:] @ R[:, 384:].T
            dup = ((sim_cls > 0.95) & (sim_pm > 0.95)).any(1)
            keep[c[dup]] = False
    # perceptual hash: exact-ish copies (resized/re-encoded uploads)
    ph_path = DATA / "phash.npz"
    ph = dict(np.load(ph_path, allow_pickle=True)["ph"].item()) if ph_path.exists() else {}
    todo = [i for i in rows.image_id if i not in ph]
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        def one(iid):
            return iid, int(str(imagehash.phash(Image.open(DATA / "images" / iid[:2] / f"{iid}.jpg"))), 16)
        with ThreadPoolExecutor(16) as ex:
            ph.update(ex.map(one, todo))
        np.savez(ph_path, ph=np.array(ph, dtype=object))
    hv = np.array([ph[i] for i in rows.image_id], dtype=np.uint64)
    for later in (1, 2):
        ref = np.where((s < later) & keep)[0]
        cand = np.where((s == later) & keep)[0]
        R = hv[ref]
        for b in range(0, len(cand), 256):
            c = cand[b:b + 256]
            ham = np.bitwise_count(hv[c][:, None] ^ R[None])
            keep[c[(ham <= 4).any(1)]] = False
    return keep


def threshold(p: np.ndarray, y: np.ndarray) -> float | None:
    o = np.argsort(-p)
    ps, ys = p[o], y[o]
    tp = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    prec = tp / n
    if len(ps) < MIN_FIRES:
        return None
    # envelope over cuts with at least MIN_FIRES fires: min precision over all higher thresholds (fewer fires)
    env = np.full(len(ps), -1.0)
    env[MIN_FIRES - 1:] = np.minimum.accumulate(prec[MIN_FIRES - 1:])
    distinct = np.r_[ps[:-1] > ps[1:], True]   # the cut after k separates k from k+1 (no tie straddles the threshold)
    ok = np.where((env >= TARGET_P) & distinct)[0]
    if len(ok) == 0:
        return None
    return float(ps[ok.max()])


def metrics(p, y, generic, t) -> dict:
    fire = p >= t
    tp, fp = int((fire & (y == 1)).sum()), int((fire & (y == 0)).sum())
    npos = int((y == 1).sum())
    lo, hi = wilson(tp, tp + fp)
    g = generic & (y == 0)
    fpr_g = float(fire[g].mean()) if g.any() else float("nan")
    gp = generic & (y >= 0)
    prev = float((y[gp] == 1).mean()) if gp.any() else float("nan")
    rec = tp / npos if npos else 0.0
    nat = prev * rec / (prev * rec + (1 - prev) * fpr_g) if prev * rec + (1 - prev) * fpr_g > 0 else float("nan")
    return {"n_pos": npos, "n_neg": int((y == 0).sum()), "n_generic_neg": int(g.sum()), "fires": int(fire.sum()), "tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "precision_wilson95": [lo, hi], "recall": rec,
            "fpr_generic": fpr_g, "generic_prevalence": prev, "precision_at_generic_prevalence": nat,
            "ap": float(average_precision_score(y == 1, p)) if npos else float("nan")}


def main() -> None:
    no_test = "--no-test" in sys.argv   # development runs: fit and choose on train/val only, never look at test
    t0 = time.time()
    rows, X = load()
    keep = dedup(rows, X)
    print(f"near-duplicates dropped: {int((~keep).sum())} ({rows[~keep].split.value_counts().to_dict()})")
    rows, X = rows[keep].reset_index(drop=True), X[keep]
    split = rows.split.to_numpy()
    generic = rows.generic.to_numpy()
    results, W, B = {}, {}, {}
    for head, (key, _) in CONCEPTS.items():
        y = rows[head].to_numpy().astype(int)
        # Another head feeding the same badge key: its positives are not a mistake for this one's badge; exclude them.
        for other, (k2, _) in CONCEPTS.items():
            if other != head and k2 == key:
                y = np.where((rows[other].to_numpy() == 1) & (y == 0), -1, y)
        tr, va, te = (split == "train") & (y >= 0), (split == "val") & (y >= 0), (split == "test") & (y >= 0)
        if (y[tr] == 1).sum() < 20 or (y[va] == 1).sum() < 5:
            results[head] = {"key": key, "decision": "unsupported", "reason": f"too few positives (train {(y[tr]==1).sum()}, val {(y[va]==1).sum()})"}
            print(head, results[head]["reason"]); continue
        best = None
        for C in C_GRID:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
            m.fit(X[tr], y[tr])
            ap = average_precision_score(y[va], m.decision_function(X[va]))
            if best is None or ap > best[0]:
                best = (ap, C, m)
        ap, C, m = best
        # Platt scaling on val, folded into the weights: a monotone map, so the val ranking and the chosen operating
        # point are unchanged, but scores stay in a range where float32 sigmoid does not saturate at the threshold.
        zv = m.decision_function(X[va])
        pl = LogisticRegression(C=1e6, max_iter=1000).fit(zv[:, None], y[va])
        a_, c_ = float(pl.coef_[0, 0]), float(pl.intercept_[0])
        if a_ <= 0:
            results[head] = {"key": key, "decision": "unsupported", "reason": "validation scores not increasing with the label"}
            continue
        w_cal, b_cal = (a_ * m.coef_[0]).astype(np.float32), float(a_ * m.intercept_[0] + c_)
        score = lambda Z: 1 / (1 + np.exp(-(Z.astype(np.float32) @ w_cal + np.float32(b_cal)).astype(np.float64)))
        sv = score(X[va])
        t = threshold(sv, y[va])
        r = {"key": key, "C": C, "val_ap": float(ap), "platt": [a_, c_], "n_train_pos": int((y[tr] == 1).sum()), "n_train_neg": int((y[tr] == 0).sum())}
        if t is None:
            r.update(decision="unsupported", reason=f"no validation threshold reaches precision {TARGET_P} with >= {MIN_FIRES} fires")
        elif no_test:
            r["threshold"] = t; r["val"] = metrics(sv, y[va], generic[va], t); r["decision"] = "dev"
            print(f"{head:15s} C={C:<4} valAP={ap:.3f} t={t:.3f} val R={r['val']['recall']:.3f} n_val={r['val']['n_pos']}", flush=True)
            results[head] = r; continue
        else:
            st = score(X[te])
            r["threshold"] = t
            r["val"] = metrics(sv, y[va], generic[va], t)
            r["test"] = metrics(st, y[te], generic[te], t)
            tm = r["test"]
            ok = (tm["n_pos"] >= SHIP_N and tm["fires"] > 0 and tm["precision"] >= SHIP_P and tm["precision_wilson95"][0] >= SHIP_LB and tm["recall"] >= SHIP_R)
            r["decision"] = "shipped" if ok else "unsupported"
            if not ok:
                why = []
                if tm["n_pos"] < SHIP_N: why.append(f"test positives {tm['n_pos']} < {SHIP_N}")
                if not tm["fires"] or tm["precision"] < SHIP_P: why.append(f"test precision {tm['precision']:.3f} < {SHIP_P}")
                if tm["precision_wilson95"][0] < SHIP_LB: why.append(f"Wilson lower bound {tm['precision_wilson95'][0]:.3f} < {SHIP_LB}")
                if tm["recall"] < SHIP_R: why.append(f"recall {tm['recall']:.3f} < {SHIP_R}")
                r["reason"] = "; ".join(why)
            W[head], B[head] = w_cal, b_cal
        results[head] = r
        tm = r.get("test", {})
        print(f"{head:15s} C={C:<4} valAP={ap:.3f} t={r.get('threshold', float('nan')):.3f} test P={tm.get('precision', float('nan')):.3f} "
              f"LB={tm.get('precision_wilson95', [float('nan')])[0]:.3f} R={tm.get('recall', float('nan')):.3f} n={tm.get('n_pos', 0)} "
              f"FPRg={tm.get('fpr_generic', float('nan')):.4f} -> {r['decision']}", flush=True)
    if no_test:
        return
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {"n_images": len(rows), "splits": rows.split.value_counts().to_dict(), "seconds": time.time() - t0,
               "rule": {"target_val_precision": TARGET_P, "min_fires": MIN_FIRES, "ship_precision": SHIP_P, "ship_wilson_lb": SHIP_LB,
                        "ship_recall": SHIP_R, "ship_min_test_pos": SHIP_N}, "heads": results}
    (REPORTS / "heads_results.json").write_text(json.dumps(summary, indent=1))
    np.savez(DATA / "heads_model.npz", heads=np.array(list(W)), W=np.stack(list(W.values())), b=np.array(list(B.values()), dtype=np.float32))
    print(f"shipped {sum(r['decision'] == 'shipped' for r in results.values())}/{len(results)}")


if __name__ == "__main__":
    main()
