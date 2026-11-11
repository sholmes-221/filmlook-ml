"""C18: does photographer matching depend on subject matter rather than style?

    python -m research.photographers.subject_audit [--seed 0]   # -> reports/photographers/subject_audit.json

Subjects: `subject_primary` from the Haiku annotations of the bank (style-annotation-v2), used only to stratify this
audit, never as a training target. Same leakage-grouped split and classes as evaluate.py; scorer = mean of the 3
closest bank works per photographer (the deployed scorer). Measurements:
  1. subject agreement: how often the top retrieved bank work has the query's subject, against the base rate;
  2. subject-only baseline: photographer predicted from the subject label alone (how entangled the task is);
  3. cross-subject accuracy: photographer top-1 when every bank work sharing the query's subject is removed;
  4. a subject-debiased embedding (directions that predict subject on the bank projected out, iteratively) fused with
     the measured style descriptors (axes17, film205), weights chosen on validation under the cross-subject protocol.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from research.photographers.evaluate import EMB, HELD_OUT, boot_ci, l2, leakage_groups, load_rows, load_vectors, split_groups, topm_scores

ANN = "data/annotations/claude_research/bank/claude-haiku-5-5.jsonl"


def subjects(ids):
    lab = {}
    for line in open(ANN):
        r = json.loads(line)
        if r.get("valid") and r.get("annotation"):
            lab[r["item_id"]] = r["annotation"].get("subject_primary")
    return np.array([lab.get(i) for i in ids], dtype=object)


def debias(X, s, train, k=12, rounds=3):
    """Iterative nullspace projection: fit subject classifiers (LDA directions) on the bank and remove them."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    Z = X.copy()
    for _ in range(rounds):
        lda = LinearDiscriminantAnalysis(n_components=min(k, len(set(s[train])) - 1)).fit(Z[train], s[train])
        W = np.linalg.qr(lda.scalings_[:, :lda.n_components if hasattr(lda, "n_components") and lda.n_components else k])[0]
        Z = Z - (Z @ W) @ W.T
    return l2(Z)


def evaluate(S_parts, weights, y, s, q, b, n_class, cross):
    """Photographer top-1 for queries q against bank b, optionally excluding bank works with the query's subject."""
    S = sum(w * P[np.ix_(q, b)] for w, P in zip(weights, S_parts))
    if cross:
        S = np.where(s[q][:, None] == s[b][None, :], -1e9, S)
    sc = topm_scores(S, y[b], n_class, 3)
    return sc.argmax(1) == y[q], S


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    rows = load_rows(); ids = [r["work_id"] for r in rows]
    Xp = load_vectors("prod", ids)
    groups, _ = leakage_groups(rows, l2(Xp[:, :384])); split = split_groups(rows, groups, a.seed)
    pid = np.array([r["photographer_id"] for r in rows]); counts = Counter(pid.tolist())
    classes = sorted(p for p, n in counts.items() if n >= 40 and p not in HELD_OUT); cidx = {p: i for i, p in enumerate(classes)}
    y = np.array([cidx.get(p, -1) for p in pid]); s = subjects(ids)
    ok = (y >= 0) & (s != None)  # noqa: E711
    b, v, t = np.where(ok & (split == "bank"))[0], np.where(ok & (split == "val"))[0], np.where(ok & (split == "test"))[0]
    n_class = len(classes)
    rep = {"n": {"bank": len(b), "val": len(v), "test": len(t)}, "classes": n_class, "subjects": dict(Counter(s[b].tolist()).most_common())}
    # descriptors
    d = np.load(EMB / "descriptors.npz", allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}; order = [pos[w] for w in ids]
    ax = np.nan_to_num(d["axes"][order], nan=0.5); f205 = d["film205"][order]
    axes = l2(ax - ax[b].mean(0)); film = l2((f205 - f205[b].mean(0)) / (f205[b].std(0) + 1e-6))
    spaces = {"prod": l2(Xp), "axes17": axes, "film205": film}
    for name in ("siglip_base", "clip_b32", "dinov2_base"):
        X = load_vectors(name, ids)
        if X is not None:
            spaces[name] = X
    spaces["prod_debiased"] = debias(l2(Xp), s, b)
    # Offline reference only (production cannot compute it): Claude's annotation vector without any subject field
    # (subject defines the protocol), i.e. motifs, composition, light, tone, mood.
    ann = EMB / "claude_v1.npz"
    if ann.exists():
        from research.claude_annotations import schema as S
        dd = np.load(ann, allow_pickle=False); apos = {w: i for i, w in enumerate(dd["work_ids"].tolist())}
        if all(w in apos for w in ids):
            names = S.feature_names()
            keep = [i for i, n in enumerate(names) if not n.startswith(("subject_primary", "subjects_secondary", "people_count", "human_presence", "setting", "urban_setting"))]
            A = dd["X"][[apos[w] for w in ids]][:, keep].astype(np.float32)
            spaces["claude_style_no_subject"] = l2(A - A[b].mean(0))
            motif = [k for k, i in enumerate(keep) if names[i].startswith("motifs")]
            spaces["claude_motifs_only"] = l2(A[:, motif] - A[b][:, motif].mean(0))
    sims = {k: X @ X.T for k, X in spaces.items()}
    # 2. subject-only baseline: photographer frequency given subject, from the bank
    tab = {}
    for c, sub in zip(y[b], s[b]):
        tab.setdefault(sub, Counter())[c] += 1
    pred = np.array([tab.get(sub, Counter({0: 1})).most_common(1)[0][0] for sub in s[t]])
    rep["subject_only_top1"] = float((pred == y[t]).mean()); rep["uniform_top1"] = 1 / n_class
    base_same = float(np.mean([np.mean(s[b] == sub) for sub in s[t]]))
    rep["base_rate_same_subject"] = base_same
    rep["spaces"] = {}
    for name, S in sims.items():
        full, Sfull = evaluate([S], [1.0], y, s, t, b, n_class, cross=False)
        cross, _ = evaluate([S], [1.0], y, s, t, b, n_class, cross=True)
        nn = Sfull.argmax(1)
        same = float(np.mean(s[b][nn] == s[t]))
        # subject predictability: 1-NN subject accuracy within the bank -> test
        rep["spaces"][name] = {"top1_full": float(full.mean()), "top1_full_ci": boot_ci(full.astype(float), groups[t]),
                               "top1_cross_subject": float(cross.mean()), "top1_cross_ci": boot_ci(cross.astype(float), groups[t]),
                               "top1_retrieval_same_subject_rate": same, "subject_1nn_accuracy": same}
        print(f"{name:14s} full {full.mean():.3f}  cross-subject {cross.mean():.3f}  top-1 retrieval same subject {same:.3f} (base {base_same:.3f})", flush=True)
    # 4. hybrid: weights on validation under the cross-subject protocol
    parts = ["prod_debiased", "axes17", "film205"] + (["siglip_base"] if "siglip_base" in sims else [])
    best = None
    for w0 in (0.0, 0.5, 1.0):
        for w1 in (0.0, 0.25, 0.5, 1.0):
            for w2 in (0.0, 0.25, 0.5, 1.0):
                ws = [w0, w1, w2] + ([0.0] if len(parts) == 4 else [])
                if sum(ws) == 0:
                    continue
                acc = evaluate([sims[p] for p in parts], ws, y, s, v, b, n_class, cross=True)[0].mean()
                if best is None or acc > best[0]:
                    best = (acc, ws)
    ws = best[1]
    for cross in (False, True):
        r, _ = evaluate([sims[p] for p in parts], ws, y, s, t, b, n_class, cross=cross)
        rep.setdefault("hybrid", {"parts": parts, "weights": ws, "val_cross": float(best[0])})[("cross" if cross else "full")] = {"top1": float(r.mean()), "ci": boot_ci(r.astype(float), groups[t])}
    # deployed-style hybrid for comparison: production embedding + axes + film205 with the shipped fusion idea (equal z weights)
    for cross in (False, True):
        r, _ = evaluate([sims["prod"], sims["axes17"], sims["film205"]], [1.0, 0.5, 0.5], y, s, t, b, n_class, cross=cross)
        rep.setdefault("prod_hybrid_reference", {})[("cross" if cross else "full")] = {"top1": float(r.mean()), "ci": boot_ci(r.astype(float), groups[t])}
    if "claude_style_no_subject" in sims:
        for cross in (False, True):
            r, _ = evaluate([sims["prod"], sims["axes17"], sims["film205"], sims["claude_style_no_subject"]], [1.0, 0.5, 0.5, 0.5], y, s, t, b, n_class, cross=cross)
            rep.setdefault("prod_hybrid_plus_claude_style_offline", {})[("cross" if cross else "full")] = {"top1": float(r.mean()), "ci": boot_ci(r.astype(float), groups[t])}
        print("prod hybrid + claude style (offline)", rep["prod_hybrid_plus_claude_style_offline"])
    print("subject-only top1", round(rep["subject_only_top1"], 3), "uniform", round(1 / n_class, 3))
    print("hybrid (debiased)", rep["hybrid"]); print("prod hybrid ref", rep["prod_hybrid_reference"])
    json.dump(rep, open(f"reports/photographers/subject_audit{'' if a.seed == 0 else '_seed' + str(a.seed)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
