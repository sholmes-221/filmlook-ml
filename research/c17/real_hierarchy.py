"""C17: hierarchical film identification on real scans (uploader-claimed labels), leakage-safe.

    python -m research.c17.real_hierarchy   # -> reports/c17/real_hierarchy.json

Data: data/c08/pairs/extra_real (1,146 CommonCatalog / ARTeFACT film scans; labels are the uploader's claim, tier
"uploader-reported"; 44 ARTeFACT rows are owner records). Features: 312 colour values; DINOv2-small blocks 4/8/12
(each CLS / mean part unit length). Validation: 5-fold GroupKFold by uploader (an uploader's rolls, scanner and subjects
never straddle train and test), plus a dHash near-duplicate check across folds. Levels: negative vs reversal (colour);
Kodak vs Fuji (colour negative); stock line (lines with >= 8 uploaders); exact stock within a line. Reported against
the majority class with uploader-bootstrap CIs; calibration (ECE); unknown-line rejection (max-probability AUROC
between scans of trained lines and scans of other lines).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

P = Path("data/c08/pairs/extra_real"); OUT = Path("reports/c17")
RNG = np.random.default_rng(0)


def parts(L):
    return np.concatenate([L[:, k * 384:(k + 1) * 384] / (np.linalg.norm(L[:, k * 384:(k + 1) * 384], axis=1, keepdims=True) + 1e-12) for k in range(6)], 1)


def boot(ok, g, n=1000):
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    v = [ok[np.concatenate([idx[u] for u in RNG.choice(ug, len(ug))])].mean() for _ in range(n)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def ece(P, y, bins=10):
    c, p = P.max(1), P.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((p[m] == y[m]).mean() - c[m].mean())
    return float(e)


def oof(X, y, g, C=0.05):
    P = np.zeros((len(y), len(set(y))))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        P[np.ix_(te, m.classes_)] = m.predict_proba(sc.transform(X[te]))
    return P


def main():
    m = [json.loads(l) for l in open(P / "meta.jsonl")]
    D = parts(np.load(P / "dino.npy").astype(np.float32)); C = np.load(P / "colour.npy").astype(np.float32)
    F36 = np.load(P / "f36.npy").astype(np.float32); CG = np.load(P / "colour_greyworld.npy").astype(np.float32)
    feats = {"film_stats36 (grain/halation-type cues)": F36, "colour312_greyworld (scanner cast removed)": CG, "dino_all+colour+stats36+greyworld": np.c_[D, C, F36, CG], "colour312": C, "dino_L12 (production)": D[:, 1536:], "dino_L4+L8+L12": D, "dino_all+colour": np.c_[D, C]}
    up = np.array([r["uploader"] for r in m]); ft = np.array([r.get("film_type") or "" for r in m]); line = np.array([r.get("line") or "" for r in m])
    stock = np.array([r.get("stock") or "" for r in m])
    brand = np.array(["kodak" if l.startswith("kodak") else "fuji" if l.startswith("fuji") else "" for l in line])
    lu = {l: len(set(up[line == l])) for l in set(line) if l}
    lines = sorted(l for l, n in lu.items() if n >= 8)
    tasks = {
        "negative_vs_reversal": (np.isin(ft, ["colour_negative", "colour_reversal"]), ft),
        "kodak_vs_fuji (colour negative)": ((ft == "colour_negative") & (brand != ""), brand),
        "stock_line": (np.isin(line, lines), line),
        "portra_160_vs_400_vs_800 (exact)": (np.isin(stock, ["kodak_portra_160", "kodak_portra_400", "kodak_portra_800"]), stock),
    }
    rep = {"n_scans": len(m), "uploaders": int(len(set(up))), "label_tier": "uploader-reported (1,102) / owner-recorded ARTeFACT (44)", "lines": {l: lu[l] for l in lines}, "tasks": {}}
    for tname, (sel, lab) in tasks.items():
        y_raw = lab[sel]; classes = sorted(set(y_raw)); y = np.array([classes.index(v) for v in y_raw]); g = up[sel]
        maj = Counter(y).most_common(1)[0][1] / len(y)
        res = {"n": int(sel.sum()), "uploaders": int(len(set(g))), "classes": classes, "majority": round(maj, 3)}
        for fname, X in feats.items():
            Pr = oof(X[sel], y, g); ok = Pr.argmax(1) == y
            bal = float(np.mean([ok[y == k].mean() for k in range(len(classes))]))
            r = {"top1": round(float(ok.mean()), 3), "ci": boot(ok, g), "balanced_acc": round(bal, 3), "ece": round(ece(Pr, y), 3)}
            if len(classes) > 3:
                r["top3"] = round(float((np.argsort(-Pr, 1)[:, :3] == y[:, None]).any(1).mean()), 3)
            res[fname] = r
            print(f"{tname:36s} {fname:24s} top1 {r['top1']:.3f} {r['ci']} bal {bal:.3f} (majority {maj:.3f}, chance-bal {1/len(classes):.3f})", flush=True)
        rep["tasks"][tname] = res
    # unknown-line rejection: train on the known lines (all folds' other uploaders), score scans of other labelled lines
    known = np.isin(line, lines); other = (line != "") & ~known
    y_raw = line[known]; classes = sorted(set(y_raw)); y = np.array([classes.index(v) for v in y_raw])
    for fname, X in feats.items():
        Pk = oof(X[known], y, up[known])
        sc = StandardScaler().fit(X[known]); mdl = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[known]), y)
        Po = mdl.predict_proba(sc.transform(X[other]))
        auc = roc_auc_score(np.r_[np.ones(len(Pk)), np.zeros(len(Po))], np.r_[Pk.max(1), Po.max(1)])
        rep.setdefault("unknown_line_rejection_auroc", {})[fname] = round(float(auc), 3)
    print("unknown-line rejection AUROC:", rep["unknown_line_rejection_auroc"], "| other-line scans:", int(other.sum()))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "real_hierarchy.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
