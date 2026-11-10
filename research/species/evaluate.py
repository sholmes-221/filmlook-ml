"""Hierarchical evaluation, confidence-tier policy, per-class decisions and unknown-species rejection.

  python -m research.species.evaluate --run mobilenet_v3_large_last [--decision]   -> data/species/eval/<run>.json
                                                                                       (+ research/species/decision.json)

Species / genus / family top-1 and top-5 on the recorder-disjoint test split (genus and family probabilities are the
summed softmax of their species), recorder-cluster bootstrap CIs, per-class precision, the tier policy
("species if p >= tau_s and the class's held-out species precision >= 0.90, else genus if >= 0.90, else family, else
uncertain"; taus and tiers are chosen on the validation split, the policy is then scored on test), and rejection of
unknown species / non-wildlife with max-softmax and (when available) kNN distance: AUROC and FPR at 95 % TPR.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from research.species import common as C
from research.species import data as D

MIN_PREC = 0.90
MIN_N = 3


def softmax(lg):
    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_preds(run: str, split: str):
    p = C.DATA / "preds" / f"{run}__{split}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def group_probs(probs, class_groups, groups):
    out = np.zeros((len(probs), len(groups)), np.float32)
    for ci, gi in enumerate(class_groups):
        out[:, gi] += probs[:, ci]
    return out


def topk_acc(scores, y, k):
    top = np.argsort(-scores, 1)[:, :k]
    return (top == y[:, None]).any(1)


def boot_ci(hits, clusters, n=1000, seed=0):
    """Cluster bootstrap over recorders of a mean of 0/1 hits."""
    rng = np.random.default_rng(seed)
    cl = defaultdict(list)
    for h, c in zip(hits, clusters):
        cl[c].append(h)
    keys = list(cl)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        s = np.concatenate([cl[keys[i]] for i in pick])
        vals.append(s.mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def auroc(pos, neg):
    """pos: scores of in-distribution (should be high), neg: unknown."""
    s = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s[np.r_[True, np.diff(np.sort(s)) == 0]]):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = len(pos), len(neg)
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(pos, neg, tpr=0.95):
    thr = np.percentile(pos, 100 * (1 - tpr))
    return float((neg >= thr).mean()), float(thr)


def tier_tables(probs, y, groups_of, n_classes, taus):
    """For each tau: coverage and precision of species predictions, per-class precision counts."""
    pred = probs.argmax(1); conf = probs.max(1)
    out = {}
    for tau in taus:
        acc = conf >= tau
        out[tau] = {"coverage": float(acc.mean()), "precision": float((pred[acc] == y[acc]).mean()) if acc.any() else float("nan"),
                    "per_class": {c: (int(((pred == c) & acc & (y == c)).sum()), int(((pred == c) & acc).sum())) for c in range(n_classes)}}
    return out


def choose_tau(probs, y, taus, min_prec=MIN_PREC):
    pred, conf = probs.argmax(1), probs.max(1)
    best = None
    for tau in taus:
        acc = conf >= tau
        if acc.sum() == 0:
            continue
        prec = (pred[acc] == y[acc]).mean()
        if prec >= min_prec and (best is None or acc.mean() > best[1]):
            best = (tau, float(acc.mean()), float(prec))
    return best or (1.01, 0.0, float("nan"))


def apply_policy(probs, cls_tier, gen_tier, fam_tier, cg, cf, taus):
    """-> list of (tier, index) decisions per row."""
    sp_p, sp = probs.max(1), probs.argmax(1)
    gp = group_probs(probs, cg, range(max(cg) + 1)); fp = group_probs(probs, cf, range(max(cf) + 1))
    g, gpm = gp.argmax(1), gp.max(1); f, fpm = fp.argmax(1), fp.max(1)
    out = []
    for i in range(len(probs)):
        if sp_p[i] >= taus["species"] and cls_tier[sp[i]]:
            out.append(("species", int(sp[i])))
        elif gpm[i] >= taus["genus"] and gen_tier[g[i]]:
            out.append(("genus", int(g[i])))
        elif fpm[i] >= taus["family"] and fam_tier[f[i]]:
            out.append(("family", int(f[i])))
        else:
            out.append(("uncertain", -1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--decision", action="store_true", help="also write research/species/decision.json from this run")
    ap.add_argument("--reject-score", default="msp", choices=["msp", "knn_dist"], help="score used for the rejection metrics")
    a = ap.parse_args()
    rows = D.load_rows()
    space = D.label_space(rows)
    classes, index = space["classes"], space["index"]
    n = len(classes)
    genera = sorted({c["genus"] for c in classes}); families = sorted({c["family"] for c in classes})
    gi, fi = {g: i for i, g in enumerate(genera)}, {f: i for i, f in enumerate(families)}
    cg = [gi[c["genus"]] for c in classes]; cf = [fi[c["family"]] for c in classes]
    res = {"run": a.run, "n_classes": n, "n_genera": len(genera), "n_families": len(families)}

    def prep(split):
        p = load_preds(a.run, split)
        if p is None:
            return None
        rr = [rows[i] for i in p["idx"]]
        keep = np.array([r.get("species_id") in index for r in rr]) if not split.startswith("unknown") else np.ones(len(rr), bool)
        y = np.array([index.get(r.get("species_id"), -1) for r in rr])[keep]
        return {"probs": softmax(p["logits"][keep]), "y": y, "rows": [r for r, k in zip(rr, keep) if k],
                "knn_dist": p["knn_dist"][keep] if "knn_dist" in p else None}

    val, test = prep("val"), prep("test")
    unk = {s: prep(s) for s in ("unknown_taxa", "unknown_nonwild")}
    res["n_val"], res["n_test"] = len(val["y"]), len(test["y"])
    # ---- hierarchical accuracy on test
    clusters = [r["recorder"] for r in test["rows"]]
    yg = np.array([cg[c] for c in test["y"]]); yf = np.array([cf[c] for c in test["y"]])
    gp = group_probs(test["probs"], cg, genera); fp = group_probs(test["probs"], cf, families)
    acc = {}
    for name, sc, yy in (("species", test["probs"], test["y"]), ("genus", gp, yg), ("family", fp, yf)):
        for k in (1, 5):
            h = topk_acc(sc, yy, k)
            acc[f"{name}_top{k}"] = {"value": float(h.mean()), "ci95": boot_ci(h, clusters)}
    res["test_accuracy"] = acc
    res["val_species_top1"] = float(topk_acc(val["probs"], val["y"], 1).mean())
    # per kind
    kinds = np.array([classes[c]["kind"] for c in test["y"]])
    res["test_species_top1_by_kind"] = {k: float(topk_acc(test["probs"][kinds == k], test["y"][kinds == k], 1).mean()) for k in ("bird", "flower")}
    # ---- per-class precision / recall on test (argmax, no threshold)
    pred = test["probs"].argmax(1)
    per_class = {}
    for c, cl in enumerate(classes):
        tp = int(((pred == c) & (test["y"] == c)).sum()); fpn = int(((pred == c) & (test["y"] != c)).sum()); fn = int(((pred != c) & (test["y"] == c)).sum())
        per_class[cl["species_id"]] = {"n_test": int((test["y"] == c).sum()), "tp": tp, "fp": fpn, "fn": fn,
                                       "precision": tp / (tp + fpn) if tp + fpn else None, "recall": tp / (tp + fn) if tp + fn else None}
    res["per_class_test"] = per_class
    res["macro_recall_test"] = float(np.mean([v["recall"] for v in per_class.values() if v["recall"] is not None]))
    # ---- tier policy chosen on val
    taus_grid = [round(t, 2) for t in np.arange(0.30, 1.0, 0.01)]
    vgp = group_probs(val["probs"], cg, genera); vfp = group_probs(val["probs"], cf, families)
    vyg = np.array([cg[c] for c in val["y"]]); vyf = np.array([cf[c] for c in val["y"]])
    tau_s = choose_tau(val["probs"], val["y"], taus_grid); tau_g = choose_tau(vgp, vyg, taus_grid); tau_f = choose_tau(vfp, vyf, taus_grid)
    taus = {"species": tau_s[0], "genus": tau_g[0], "family": tau_f[0]}
    res["tau_selection_val"] = {"species": tau_s, "genus": tau_g, "family": tau_f, "rule": f"max coverage with accepted precision >= {MIN_PREC} on val"}

    def tiers(probs, y, tau, n_groups):
        pr, cf_ = probs.argmax(1), probs.max(1)
        acc_ = cf_ >= tau
        ok, stat = [], []
        for c in range(n_groups):
            m = (pr == c) & acc_
            k, tot = int((m & (y == c)).sum()), int(m.sum())
            prec = k / tot if tot else None
            ok.append(bool(tot >= MIN_N and prec >= MIN_PREC)); stat.append((k, tot, prec))
        return ok, stat

    cls_tier, cls_stat = tiers(val["probs"], val["y"], taus["species"], n)
    gen_tier, gen_stat = tiers(vgp, vyg, taus["genus"], len(genera))
    fam_tier, fam_stat = tiers(vfp, vyf, taus["family"], len(families))
    res["tiers_enabled"] = {"species": int(sum(cls_tier)), "genus": int(sum(gen_tier)), "family": int(sum(fam_tier))}

    def score_policy(d, y_s, y_g, y_f):
        dec = apply_policy(d["probs"], cls_tier, gen_tier, fam_tier, cg, cf, taus)
        out = {}
        for tier, truth in (("species", y_s), ("genus", y_g), ("family", y_f)):
            m = [i for i, (t, _) in enumerate(dec) if t == tier]
            out[tier] = {"n": len(m), "share": len(m) / len(dec), "precision": float(np.mean([dec[i][1] == truth[i] for i in m])) if m else None}
        out["uncertain"] = {"n": sum(t == "uncertain" for t, _ in dec), "share": sum(t == "uncertain" for t, _ in dec) / len(dec)}
        return out, dec

    res["policy_val"], _ = score_policy(val, val["y"], vyg, vyf)
    res["policy_test"], test_dec = score_policy(test, test["y"], yg, yf)
    # test-side per-class precision at the species tier (held-out evidence for decision.json)
    tier_prec_test = {}
    for c, cl in enumerate(classes):
        m = [i for i, (t, k) in enumerate(test_dec) if t == "species" and k == c]
        tier_prec_test[cl["species_id"]] = {"n_claimed": len(m), "precision": float(np.mean([test["y"][i] == c for i in m])) if m else None}
    # ---- unknown rejection
    rej = {}
    pos_msp = test["probs"].max(1)
    for s, d in unk.items():
        if d is None:
            continue
        neg_msp = d["probs"].max(1)
        fpr, thr = fpr_at_tpr(pos_msp, neg_msp)
        rej[s] = {"n": len(neg_msp), "msp": {"auroc": auroc(pos_msp, neg_msp), "fpr_at_95tpr": fpr, "threshold": thr}}
        if test["knn_dist"] is not None and d["knn_dist"] is not None:
            fpr2, thr2 = fpr_at_tpr(-test["knn_dist"], -d["knn_dist"])
            rej[s]["knn_dist"] = {"auroc": auroc(-test["knn_dist"], -d["knn_dist"]), "fpr_at_95tpr": fpr2, "threshold_dist": -thr2}
        udec = apply_policy(d["probs"], cls_tier, gen_tier, fam_tier, cg, cf, taus)
        rej[s]["policy"] = dict(Counter(t for t, _ in udec))
        rej[s]["false_species_claim_rate"] = sum(t == "species" for t, _ in udec) / len(udec)
    res["rejection"] = rej
    (C.DATA / "eval").mkdir(exist_ok=True)
    json.dump(res, open(C.DATA / "eval" / f"{a.run}.json", "w"), indent=1, ensure_ascii=False)
    summary = {k: res[k] for k in ("run", "n_classes", "n_test", "test_accuracy", "test_species_top1_by_kind", "tau_selection_val", "tiers_enabled", "policy_test")}
    summary["rejection"] = {s: {k: v for k, v in d.items() if k != "policy"} for s, d in rej.items()}
    print(json.dumps(summary, indent=1))
    if a.decision:
        taxa = {t["id"]: t for t in json.load(open(C.DATA / "taxa.json"))}
        dec = {}
        for c, cl in enumerate(classes):
            sid = cl["species_id"]
            g_ok, f_ok = gen_tier[cg[c]], fam_tier[cf[c]]
            tier = "species" if cls_tier[c] else "genus" if g_ok else "family" if f_ok else "uncertain"
            dec[sid] = {"tier_enabled": tier, "gbif_key": cl["gbif_key"], "canonical": cl["canonical"], "genus": cl["genus"], "family": cl["family"],
                        "threshold": taus["species"] if tier == "species" else taus["genus"] if tier == "genus" else taus["family"] if tier == "family" else None,
                        "val_species_precision": cls_stat[c][2], "val_species_claims": cls_stat[c][1],
                        "heldout_test_species_precision": tier_prec_test[sid]["precision"], "heldout_test_species_claims": tier_prec_test[sid]["n_claimed"],
                        "test_precision_argmax": per_class[sid]["precision"], "test_recall_argmax": per_class[sid]["recall"], "n_test": per_class[sid]["n_test"],
                        "n_train": sum(1 for r in rows if r["split"] == "train" and r["species_id"] == sid)}
        for sid, t in taxa.items():
            if sid not in dec:
                dec[sid] = {"tier_enabled": "unsupported", "reason": t["note"] or f"status {t['status']}" if t["status"] != "ok" else "no training images collected",
                            "gbif_key": t.get("gbif_key"), "threshold": None, "heldout_test_species_precision": None, "n_test": 0}
        meta = {"run": a.run, "taus": taus, "rule": f"tier enabled when held-out (val) precision >= {MIN_PREC} with >= {MIN_N} accepted predictions",
                "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"), "counts": dict(Counter(v["tier_enabled"] for v in dec.values()))}
        json.dump({"meta": meta, "species": dec}, open(C.OUT / "decision.json", "w"), indent=1, ensure_ascii=False)
        print("decision.json:", meta["counts"])


if __name__ == "__main__":
    main()
