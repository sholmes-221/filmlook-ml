"""C09 pilot: how reliable is the style annotation, and does keeping only reliable fields help matching?

  python -m research.c09.reliability      -> reports/c09/reliability.json

1. Agreement per field on bank works annotated twice:
   - Haiku vs Haiku (task `bankretest`: same works, prompt, model and settings, a second independent request),
   - Haiku vs Sonnet (works Sonnet annotated in the earlier quality check).
   Single-choice fields: Cohen's kappa. List fields: kappa per option (options chosen in >= 2 % of rows), averaged,
   plus mean Jaccard of the chosen sets. 0-4 axes: quadratic-weighted kappa.
2. Rule fixed before reading any matching result: a field is "reliable" when Haiku-vs-Haiku kappa >= 0.6
   (0.4 reported as a sensitivity reading). Subject fields stay out, as in subject_audit.py.
3. Matching with only reliable fields (oracle annotations, same split, scorer, cross-subject protocol and weight grid
   as local_predictors.py; w chosen on VAL cross-subject top-1, TEST read once per seed), paired against the shipped
   hybrid and against the oracle with every field.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score

from research.c09 import local_predictors as LP
from research.claude_annotations import pipeline as P
from research.claude_annotations import schema as S

OUT = LP.REP / "reliability.json"
EXCLUDE = set(LP.SUBJECT_FIELDS) | set(S.ARTEFACT_FIELDS) | {"widely_reproduced"}


def ann(task: str, model: str, strip: str = "") -> dict[str, dict]:
    return {k[len(strip):] if k.startswith(strip) else k: r["annotation"] for k, r in P.annotated(task, model).items()}


def kappa(a, b, weights=None) -> float | None:
    if len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b, weights=weights))


def agreement(x: dict[str, dict], y: dict[str, dict]) -> dict:
    ids = sorted(set(x) & set(y)); out = {"n": len(ids), "fields": {}}
    for f, opts in S.SINGLE.items():
        out["fields"][f] = {"kind": "single", "kappa": kappa([x[i][f] for i in ids], [y[i][f] for i in ids]),
                            "agree": float(np.mean([x[i][f] == y[i][f] for i in ids]))}
    for f, opts in S.MULTI.items():
        ks = []
        for o in opts:
            a = [o in x[i][f] for i in ids]; b = [o in y[i][f] for i in ids]
            if (np.mean(a) + np.mean(b)) / 2 >= 0.02:
                k = kappa(a, b)
                if k is not None:
                    ks.append(k)
        jac = [len(set(x[i][f]) & set(y[i][f])) / max(1, len(set(x[i][f]) | set(y[i][f]))) if (x[i][f] or y[i][f]) else 1.0 for i in ids]
        out["fields"][f] = {"kind": "multi", "kappa": float(np.mean(ks)) if ks else None, "options_scored": len(ks), "jaccard": float(np.mean(jac))}
    for f in S.ORDINAL:
        out["fields"][f] = {"kind": "ordinal", "kappa": kappa([int(x[i][f]) for i in ids], [int(y[i][f]) for i in ids], "quadratic"),
                            "agree": float(np.mean([x[i][f] == y[i][f] for i in ids]))}
    return out


def matching(reliable: set[str], seed: int) -> dict:
    D = LP.load(seed); split, y, s, A, names = D["split"], D["y"], D["s"], D["A"], D["names"]
    cols = np.array([LP.field_of(n) in reliable for n in names])
    ok = (y >= 0) & (s != None)  # noqa: E711
    b, v, t = (np.flatnonzero(ok & (split == k)) for k in ("bank", "val", "test"))
    axes = LP.l2(D["ax"] - D["ax"][b].mean(0)); film = LP.l2((D["f205"] - D["f205"][b].mean(0)) / (D["f205"][b].std(0) + 1e-6))
    Xp = LP.l2(D["Xp"]); base = [Xp @ Xp.T, axes @ axes.T, film @ film.T]

    def sim(M):
        Z = LP.l2(M - M[b].mean(0)); return Z @ Z.T
    variants = {"oracle_all_fields": sim(A), "oracle_reliable_fields": sim(A[:, cols])}
    rep = {"seed": seed, "columns": {"all": int(len(cols)), "reliable": int(cols.sum())}, "n": {"bank": len(b), "val": len(v), "test": len(t)}}
    per = {}
    for cross in (False, True):
        proto = "cross" if cross else "full"
        ref = LP.run_match(base, [1.0, 0.5, 0.5], y, s, t, b, D["n_class"], cross)
        rep[proto] = {"reference": {k: float(x.mean()) for k, x in ref.items()}}
        per[proto] = {"reference": ref}
        for name, Sm in variants.items():
            accs = {w: float(LP.run_match(base + [Sm], [1.0, 0.5, 0.5, w], y, s, v, b, D["n_class"], True)["top1"].mean()) for w in LP.W_GRID}
            w = max(LP.W_GRID, key=lambda k: (accs[k], -k))
            r = LP.run_match(base + [Sm], [1.0, 0.5, 0.5, w], y, s, t, b, D["n_class"], cross)
            per[proto][name] = r
            rep[proto][name] = {"w": w, **{k: float(x.mean()) for k, x in r.items()},
                                "d_vs_reference": {k: {"diff": round(float((r[k] - ref[k]).mean()), 4), "ci95": LP.paired_ci(r[k] - ref[k], D["groups"][t])} for k in r}}
        d = {k: per[proto]["oracle_reliable_fields"][k] - per[proto]["oracle_all_fields"][k] for k in ref}
        rep[proto]["reliable_minus_all"] = {k: {"diff": round(float(x.mean()), 4), "ci95": LP.paired_ci(x, D["groups"][t])} for k, x in d.items()}
    rep["_per_query"] = {"groups": D["groups"][t].tolist(), **{f"{p}:{n}:{k}": per[p][n][k].tolist() for p in per for n in per[p] for k in per[p][n]}}
    return rep


def pooled(reps: list[dict]) -> dict:
    out = {}
    for proto in ("full", "cross"):
        for comp, (a, bname) in {"reliable_vs_reference": ("oracle_reliable_fields", "reference"), "all_vs_reference": ("oracle_all_fields", "reference"),
                                 "reliable_vs_all": ("oracle_reliable_fields", "oracle_all_fields")}.items():
            for k in ("top1", "top3", "mrr"):
                diff = np.concatenate([np.array(r["_per_query"][f"{proto}:{a}:{k}"]) - np.array(r["_per_query"][f"{proto}:{bname}:{k}"]) for r in reps])
                groups = np.concatenate([np.array([f"{r['seed']}:{g}" for g in r["_per_query"]["groups"]]) for r in reps])
                out.setdefault(proto, {}).setdefault(comp, {})[k] = {"diff": round(float(diff.mean()), 4), "ci95": LP.paired_ci(diff, groups)}
    return out


def main() -> None:
    cheap, check = P.cfg()["models"]["cheap"], P.cfg()["models"]["check"]
    h1 = ann("bank", cheap); h2 = ann("bankretest", cheap, "retest:"); so = ann("bank", check)
    rep = {"schema": S.VERSION, "haiku_vs_haiku": agreement(h1, h2), "haiku_vs_sonnet": agreement(h1, so)}
    rule = {}
    for thr in (0.6, 0.4):
        rule[str(thr)] = sorted(f for f, r in rep["haiku_vs_haiku"]["fields"].items() if f not in EXCLUDE and r["kappa"] is not None and r["kappa"] >= thr)
    rep["reliable_fields"] = rule
    rep["style_fields_considered"] = sorted(f for f in rep["haiku_vs_haiku"]["fields"] if f not in EXCLUDE)
    for thr in ("0.6", "0.4"):
        reps = [matching(set(rule[thr]), seed) for seed in (0, 1)]
        rep[f"matching_kappa_{thr}"] = {"per_seed": [{k: v for k, v in r.items() if k != "_per_query"} for r in reps], "pooled": pooled(reps)}
    OUT.write_text(json.dumps(rep, indent=1))
    hh, hs = rep["haiku_vs_haiku"], rep["haiku_vs_sonnet"]
    print(f"n retest {hh['n']}, n sonnet {hs['n']}")
    for f in rep["style_fields_considered"]:
        a, b = hh["fields"][f]["kappa"], hs["fields"][f]["kappa"]
        print(f"{f:24s} H-H {a if a is None else round(a, 2)!s:>6}  H-S {b if b is None else round(b, 2)!s:>6}")
    print("reliable (0.6):", rule["0.6"])
    for thr in ("0.6", "0.4"):
        print(thr, json.dumps(rep[f"matching_kappa_{thr}"]["pooled"]))


if __name__ == "__main__":
    main()
