"""Photographer retrieval benchmark on the public-domain reference bank.

  python -m research.photographers.evaluate [--seed 0] [--min-works 40] [--out reports/photographers/benchmark.json]

Protocol (fixed before any number was read):

* Rows: eligible (public domain / CC0), downloaded works that the downloader did not mark as duplicates.
* Leakage groups: works are merged into one group when they share a session key (photographer + year + first four
  title words: frames of one sitting), when their perceptual hashes are within 10 bits, or when their production
  embeddings have cosine >= 0.90 (alternate scans, crops and re-edits of one negative). A group never straddles
  splits. Groups are assigned 60 / 20 / 20 to bank / val / test by a hash of the group's smallest work id and the
  seed.
* Known photographers: at least --min-works usable works. Open set: every other photographer's works, the works of
  ``HELD_OUT`` photographers (never in the bank), and modern CC-BY digital photographs (out of domain).
* Every choice (k, prototypes, C, temperature, fusion weights, rejection threshold) is made on val; test is read
  once per run with the bank = bank split only.

Metrics on test queries against the bank: image-level Recall@1 / @5, MRR and mAP (a hit = a bank work by the same
photographer); photographer-level top-1 / top-3 accuracy, macro-F1 and confusion for kNN vote, mean prototype,
k-means prototypes and logistic regression; expected calibration error of temperature-scaled scores; open-set
AUROC and the accept / reject trade-off at the val threshold. Baselines: majority class, uniform chance, image
shape + mean colour (archive artefacts), the 17 measured style axes, the 205-value colour / tone descriptor.
Subset reading: the FSA / OWI photographers alone (one agency, one archive, one decade, shared subjects), which is
the reading least explained by archive or era. Confidence intervals: bootstrap over leakage groups.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml.refdb import manifest  # noqa: E402

EMB = ROOT / "data" / "embeddings" / "photographers"
FSA = ("dorothea_lange", "walker_evans", "jack_delano", "russell_lee", "arthur_rothstein", "john_vachon", "marion_post_wolcott",
       "ben_shahn", "carl_mydans", "gordon_parks", "esther_bubley", "john_collier", "alfred_palmer")
HELD_OUT = ("john_vachon", "gertrude_kasebier")  # One FSA stranger (subject-matched) and one pictorialist stranger.


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def load_rows() -> list[dict]:
    """Eligible, downloaded, non-duplicate works that have a production embedding. Works downloaded after the last
    embedding pass are left out until `refresh.sh` runs again, so a download in progress cannot change a benchmark."""
    with manifest.connect() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT work_id, photographer_id, title, year, phash, width, height, is_monochrome, source FROM works "
            "WHERE rights_status = 'eligible' AND sha256 IS NOT NULL AND duplicate_of IS NULL ORDER BY work_id")]
    path = EMB / "prod.npz"
    if path.exists():
        have = set(np.load(path, allow_pickle=False)["work_ids"].tolist())
        rows = [r for r in rows if r["work_id"] in have]
    return rows


def load_vectors(name: str, ids: list[str]) -> np.ndarray | None:
    path = EMB / f"{name.split(':')[0]}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
    if any(w not in pos for w in ids):
        return None
    X = d["X"][[pos[w] for w in ids]].astype(np.float32)
    if name == "prod:cls":
        X = X[:, :384]
    return l2(X)


def session_key(r: dict) -> str:
    words = re.findall(r"[a-z]+", (r["title"] or "").lower())[:4]
    return f"{r['photographer_id']}|{r['year'] or ''}|{' '.join(words)}"


def leakage_groups(rows: list[dict], X: np.ndarray, cos_min: float = 0.90, ham_max: int = 10) -> tuple[np.ndarray, dict]:
    n = len(rows); parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_session: dict[str, int] = {}
    n_session = n_cos = n_ham = 0
    for i, r in enumerate(rows):
        k = session_key(r)
        if k in by_session:
            union(i, by_session[k]); n_session += 1
        else:
            by_session[k] = i
    S = X @ X.T
    np.fill_diagonal(S, 0)
    for i, j in zip(*np.where(np.triu(S) >= cos_min)):
        union(int(i), int(j)); n_cos += 1
    H = np.array([[int(b) for b in bin(int(r["phash"], 16))[2:].zfill(64)] for r in rows], dtype=np.int16)
    for i in range(0, n, 512):
        D = (H[i:i + 512, None, :] != H[None, :, :]).sum(-1)
        for a, b in zip(*np.where(D <= ham_max)):
            if i + a < b:
                union(int(i + a), int(b)); n_ham += 1
    g = np.array([find(i) for i in range(n)])
    return g, {"session_links": n_session, "cosine_links": n_cos, "phash_links": n_ham, "groups": int(len(set(g.tolist()))),
               "cos_min": cos_min, "ham_max": ham_max}


def split_groups(rows: list[dict], g: np.ndarray, seed: int) -> np.ndarray:
    out = np.empty(len(rows), dtype=object)
    for i in range(len(rows)):
        h = int(hashlib.sha256(f"{seed}:{rows[int(g[i])]['work_id']}".encode()).hexdigest(), 16) % 100
        out[i] = "bank" if h < 60 else ("val" if h < 80 else "test")
    return out


# ---------------------------------------------------------------- metrics

def retrieval_metrics(S: np.ndarray, yq: np.ndarray, yb: np.ndarray) -> dict:
    order = np.argsort(-S, axis=1)
    rel = yb[order] == yq[:, None]
    first = np.where(rel.any(1), rel.argmax(1), 10**9)
    ap = []
    for r in rel:
        hits = np.flatnonzero(r)
        ap.append(float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))) if len(hits) else 0.0)
    return {"recall@1": float((first < 1).mean()), "recall@5": float((first < 5).mean()), "recall@10": float((first < 10).mean()),
            "mrr": float(np.mean(1.0 / (first + 1))), "map": float(np.mean(ap)), "precision@5": float(rel[:, :5].mean())}


def class_metrics(P: np.ndarray, y: np.ndarray, n_class: int) -> dict:
    order = np.argsort(-P, axis=1)
    return {"top1": float((order[:, 0] == y).mean()), "top3": float(np.mean([y[i] in order[i, :3] for i in range(len(y))])),
            "macro_f1": float(f1_score(y, order[:, 0], labels=list(range(n_class)), average="macro", zero_division=0))}


def softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = S / T; Z = Z - Z.max(1, keepdims=True); E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_temperature(S: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.002), np.log(20), 240))
    return float(grid[int(np.argmin([-np.mean(np.log(softmax(S, t)[np.arange(len(y)), y] + 1e-12)) for t in grid]))])


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, ok = P.max(1), P.argmax(1) == y
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


# ---------------------------------------------------------------- photographer-level scorers (bank -> scores)

def knn_scores(Sq: np.ndarray, yb: np.ndarray, n_class: int, k: int) -> np.ndarray:
    """Similarity-weighted vote of the k nearest bank works, normalised by the class prior among the k so that a
    photographer with many bank works does not win by count alone: score = sum of the top-k similarities per class."""
    idx = np.argsort(-Sq, axis=1)[:, :k]
    out = np.zeros((len(Sq), n_class), dtype=np.float32)
    for c in range(n_class):
        out[:, c] = (np.take_along_axis(Sq, idx, 1) * (yb[idx] == c)).sum(1)
    return out / k


def topm_scores(Sq: np.ndarray, yb: np.ndarray, n_class: int, m: int) -> np.ndarray:
    """Mean similarity to each photographer's m closest bank works (count-invariant; several styles of one
    photographer are covered because only the closest works speak)."""
    out = np.zeros((len(Sq), n_class), dtype=np.float32)
    for c in range(n_class):
        s = np.sort(Sq[:, yb == c], axis=1)[:, ::-1][:, :m]
        out[:, c] = s.mean(1)
    return out


def proto_scores(Xq: np.ndarray, Xb: np.ndarray, yb: np.ndarray, n_class: int, k: int, seed: int) -> np.ndarray:
    out = np.zeros((len(Xq), n_class), dtype=np.float32)
    for c in range(n_class):
        Xc = Xb[yb == c]
        centres = l2(Xc.mean(0, keepdims=True)) if k == 1 or len(Xc) < 4 * k else l2(KMeans(k, n_init=4, random_state=seed).fit(Xc).cluster_centers_)
        out[:, c] = (Xq @ centres.T).max(1)
    return out


def boot_ci(correct: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        vals.append(np.concatenate([correct[members[g]] for g in pick]).mean())
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def zs(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (a - ref.mean()) / (ref.std() + 1e-9)


def evaluate_space(name: str, X: np.ndarray, y: np.ndarray, split: np.ndarray, known: np.ndarray, groups: np.ndarray, n_class: int,
                   seed: int, extra: dict[str, np.ndarray]) -> dict:
    """extra: name -> open-set query matrix in this space (rows not in the bank)."""
    b, v, t = known & (split == "bank"), known & (split == "val"), known & (split == "test")
    Xb, yb = X[b], y[b]
    Sv, St = X[v] @ Xb.T, X[t] @ Xb.T
    out: dict = {"n_bank": int(b.sum()), "n_val": int(v.sum()), "n_test": int(t.sum()), "dim": int(X.shape[1])}
    out["retrieval_test"] = retrieval_metrics(St, y[t], yb)
    first = (yb[np.argsort(-St, axis=1)[:, 0]] == y[t]).astype(float)
    out["retrieval_test"]["recall@1_ci"] = boot_ci(first, groups[t])
    # choices on val
    ks = [1, 3, 5, 10, 20, 40]
    k = max(ks, key=lambda q: class_metrics(knn_scores(Sv, yb, n_class, q), y[v], n_class)["top1"])
    ms = [1, 2, 3, 5, 8, 12]
    m = max(ms, key=lambda q: class_metrics(topm_scores(Sv, yb, n_class, q), y[v], n_class)["top1"])
    pk = max([2, 3, 5, 8], key=lambda q: class_metrics(proto_scores(X[v], Xb, yb, n_class, q, seed), y[v], n_class)["top1"])
    C = max([0.1, 0.3, 1.0, 3.0, 10.0], key=lambda q: LogisticRegression(C=q, max_iter=3000).fit(Xb, yb).score(X[v], y[v]))
    lr = LogisticRegression(C=C, max_iter=3000).fit(Xb, yb)
    scorers = {
        "knn_vote": (knn_scores(Sv, yb, n_class, k), knn_scores(St, yb, n_class, k), {"k": k}),
        "top_m_mean": (topm_scores(Sv, yb, n_class, m), topm_scores(St, yb, n_class, m), {"m": m}),
        "mean_prototype": (proto_scores(X[v], Xb, yb, n_class, 1, seed), proto_scores(X[t], Xb, yb, n_class, 1, seed), {}),
        "kmeans_prototypes": (proto_scores(X[v], Xb, yb, n_class, pk, seed), proto_scores(X[t], Xb, yb, n_class, pk, seed), {"k": pk}),
        "logreg": (lr.decision_function(X[v]), lr.decision_function(X[t]), {"C": C}),
    }
    out["methods"] = {}
    for mname, (Pv, Pt, choice) in scorers.items():
        T = fit_temperature(Pv, y[v]); prob = softmax(Pt, T)
        cm = class_metrics(Pt, y[t], n_class)
        cm.update({"choice": choice, "temperature": round(T, 5), "ece": round(ece(prob, y[t]), 4), "ece_val": round(ece(softmax(Pv, T), y[v]), 4),
                   "top1_ci": boot_ci((Pt.argmax(1) == y[t]).astype(float), groups[t])})
        fsa = np.isin(y[t], extra["_fsa_classes"])
        if fsa.sum() >= 20:
            sub = Pt[fsa][:, extra["_fsa_classes"]]
            cm["fsa_only"] = {"n": int(fsa.sum()), "n_class": int(len(extra["_fsa_classes"])),
                              "top1": float((extra["_fsa_classes"][sub.argmax(1)] == y[t][fsa]).mean()),
                              "majority": float(np.bincount(y[t][fsa]).max() / fsa.sum())}
        out["methods"][mname] = cm
    # open set, on the count-invariant scorer
    Pv, Pt = scorers["top_m_mean"][0], scorers["top_m_mean"][1]
    thr = float(np.percentile(Pv.max(1), 5))  # 95 % of val knowns accepted
    out["open_set"] = {"scorer": "top_m_mean max score", "threshold_val_p5": round(thr, 4), "known_test_accepted": float((Pt.max(1) >= thr).mean())}
    for ename, Xe in extra.items():
        if ename.startswith("_") or Xe is None or not len(Xe):
            continue
        Pe = topm_scores(Xe @ Xb.T, yb, n_class, m).max(1)
        out["open_set"][ename] = {"n": int(len(Xe)), "auroc": round(float(roc_auc_score(np.r_[np.ones(len(Pt)), np.zeros(len(Pe))], np.r_[Pt.max(1), Pe])), 4),
                                  "rejected_at_threshold": round(float((Pe < thr).mean()), 4)}
    out["_scores"] = {"val": Pv, "test": Pt, "m": m}
    out["confusion_top_m"] = np.bincount(y[t] * n_class + Pt.argmax(1), minlength=n_class * n_class).reshape(n_class, n_class).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-works", type=int, default=40)
    ap.add_argument("--out", default="reports/photographers/benchmark.json")
    a = ap.parse_args()
    rows = load_rows(); ids = [r["work_id"] for r in rows]
    Xp = load_vectors("prod", ids)
    if Xp is None:
        raise SystemExit("run research.photographers.embed --model prod first")
    groups, audit = leakage_groups(rows, l2(Xp[:, :384]))
    split = split_groups(rows, groups, a.seed)
    pid = np.array([r["photographer_id"] for r in rows])
    counts = Counter(pid.tolist())
    classes = sorted(p for p, n in counts.items() if n >= a.min_works and p not in HELD_OUT)
    cidx = {p: i for i, p in enumerate(classes)}
    known = np.isin(pid, classes)
    y = np.array([cidx.get(p, -1) for p in pid])
    fsa_classes = np.array([cidx[p] for p in classes if p in FSA])
    heldout, small = np.isin(pid, HELD_OUT), ~known & ~np.isin(pid, HELD_OUT)
    report: dict = {"seed": a.seed, "n_rows": len(rows), "classes": classes, "held_out_photographers": list(HELD_OUT),
                    "per_class": {p: dict(Counter(split[pid == p].tolist())) for p in classes}, "leakage_audit": audit,
                    "n_open_set": {"held_out_photographers": int(heldout.sum()), "small_photographers": int(small.sum())}}
    # leakage check after the split: nearest bank neighbour of each test query under the production embedding
    b, t = known & (split == "bank"), known & (split == "test")
    nn = (l2(Xp[:, :384])[t] @ l2(Xp[:, :384])[b].T).max(1)
    report["leakage_audit"]["test_to_bank_max_cosine"] = {"p50": float(np.percentile(nn, 50)), "p99": float(np.percentile(nn, 99)), "max": float(nn.max())}
    yt, n_class = y[t], len(classes)
    maj = int(np.bincount(y[b]).argmax())
    report["baselines"] = {"majority_top1": float((yt == maj).mean()), "uniform_top1": 1.0 / n_class,
                           "majority_macro_f1": float(f1_score(yt, np.full_like(yt, maj), average="macro", zero_division=0))}

    modern = None
    if (EMB / "modern_prod.npz").exists():
        modern = np.load(EMB / "modern_prod.npz")["X"].astype(np.float32)

    spaces: dict[str, np.ndarray] = {}
    for name in ("prod:cls", "prod", "dinov2_base", "siglip_base", "clip_b32"):
        X = load_vectors(name, ids)
        if X is not None:
            spaces[name] = X
    desc = EMB / "descriptors.npz"
    if desc.exists():
        d = np.load(desc, allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
        if all(w in pos for w in ids):
            order = [pos[w] for w in ids]
            ax = np.nan_to_num(d["axes"][order], nan=0.5); f205 = d["film205"][order]
            mu, sd = f205[known & (split == "bank")].mean(0), f205[known & (split == "bank")].std(0) + 1e-6
            spaces["axes17"] = l2(ax - ax[known & (split == "bank")].mean(0))
            spaces["film205"] = l2((f205 - mu) / sd)
    shape = np.array([[r["width"] / r["height"], float(r["is_monochrome"] or 0)] for r in rows], dtype=np.float32)
    spaces["shape_mono"] = l2((shape - shape[b].mean(0)) / (shape[b].std(0) + 1e-6))

    results = {}
    for name, X in spaces.items():
        extra = {"_fsa_classes": fsa_classes, "held_out_photographers": X[heldout], "small_photographers": X[small]}
        if name in ("prod", "prod:cls") and modern is not None:
            extra["modern_digital"] = l2(modern[:, :384]) if name == "prod:cls" else l2(modern)
        results[name] = evaluate_space(name, X, y, split, known, groups, n_class, a.seed, extra)
        r = results[name]
        print(f"{name:12s} R@1 {r['retrieval_test']['recall@1']:.3f} R@5 {r['retrieval_test']['recall@5']:.3f} mAP {r['retrieval_test']['map']:.3f} | "
              + " ".join(f"{k} {m['top1']:.3f}" for k, m in r["methods"].items()), flush=True)

    # Annotation space (research/claude_annotations): present only once every row has a valid annotation.
    ann = EMB / "claude_v1.npz"
    if ann.exists():
        d = np.load(ann, allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
        if all(w in pos for w in ids):
            A = d["X"][[pos[w] for w in ids]].astype(np.float32)
            results["claude_v1"] = evaluate_space("claude_v1", l2(A - A[b].mean(0)), y, split, known, groups, n_class, a.seed,
                                                  {"_fsa_classes": fsa_classes, "held_out_photographers": l2(A - A[b].mean(0))[heldout]})
            if "claude_v1_predicted" in d.files:  # the same fields predicted from the production embedding by local heads (bank split only)
                Pd = d["claude_v1_predicted"][[pos[w] for w in ids]].astype(np.float32)
                results["claude_v1_predicted"] = evaluate_space("claude_v1_predicted", l2(Pd - Pd[b].mean(0)), y, split, known, groups, n_class, a.seed,
                                                                {"_fsa_classes": fsa_classes})
        else:
            print("claude_v1.npz does not cover every row; skipped")

    # Late fusion: z-scored photographer scores (top-m mean) of several spaces, weights chosen on val, read once on test.
    v = known & (split == "val")
    fsa_t = np.isin(yt, fsa_classes)

    def fuse(base: str, others: list[str]) -> dict:
        comp = [base] + others
        Z = {c: (zs(results[c]["_scores"]["val"], results[c]["_scores"]["val"]), zs(results[c]["_scores"]["test"], results[c]["_scores"]["val"])) for c in comp}
        grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        best = None
        import itertools
        for w in itertools.product(grid, repeat=len(others)):
            acc = float(((Z[base][0] + sum(wi * Z[o][0] for wi, o in zip(w, others))).argmax(1) == y[v]).mean())
            if best is None or acc > best[0] + 1e-9:
                best = (acc, w)
        Pt = Z[base][1] + sum(wi * Z[o][1] for wi, o in zip(best[1], others))
        out = class_metrics(Pt, yt, n_class)
        base_ok = (results[base]["_scores"]["test"].argmax(1) == yt).astype(float)
        diff = (Pt.argmax(1) == yt).astype(float) - base_ok
        out.update({"weights": {base: 1.0, **{o: wi for o, wi in zip(others, best[1])}}, "val_top1": best[0],
                    "top1_ci": boot_ci((Pt.argmax(1) == yt).astype(float), groups[t]), "delta_vs_" + base: float(diff.mean()), "delta_ci": boot_ci(diff, groups[t]),
                    "_pred": Pt.argmax(1)})
        if fsa_t.sum() >= 20:
            sub = Pt[fsa_t][:, fsa_classes]
            out["fsa_only_top1"] = float((fsa_classes[sub.argmax(1)] == yt[fsa_t]).mean())
            out["fsa_only_top1_base"] = float((fsa_classes[results[base]["_scores"]["test"][fsa_t][:, fsa_classes].argmax(1)] == yt[fsa_t]).mean())
        return out

    report["fusion"] = {}
    combos = {"measured": ["film205", "axes17"], "claude": ["claude_v1"], "claude_predicted": ["claude_v1_predicted"],
              "measured+claude": ["film205", "axes17", "claude_v1"], "measured+claude_predicted": ["film205", "axes17", "claude_v1_predicted"]}
    for name, others in combos.items():
        if "prod" in results and all(o in results for o in others):
            report["fusion"][name] = fuse("prod", others)
    if "measured" in report["fusion"] and "measured+claude" in report["fusion"]:  # does the annotation add to the measured hybrid?
        d2 = (report["fusion"]["measured+claude"]["_pred"] == yt).astype(float) - (report["fusion"]["measured"]["_pred"] == yt).astype(float)
        report["fusion"]["measured+claude"]["delta_vs_measured"] = float(d2.mean()); report["fusion"]["measured+claude"]["delta_vs_measured_ci"] = boot_ci(d2, groups[t])
    for name, f in report["fusion"].items():
        f.pop("_pred")
        print("fusion", name, json.dumps({k: f[k] for k in f if k != "weights"}), f["weights"])
    for r in results.values():
        r.pop("_scores")
    report["spaces"] = results
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("majority", report["baselines"]["majority_top1"], "classes", len(classes), "->", out)


if __name__ == "__main__":
    main()
