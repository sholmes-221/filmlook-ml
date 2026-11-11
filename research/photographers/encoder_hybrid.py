"""C19: does a stronger image encoder (swapped in or added) improve the photographer-style hybrid?

    python -m research.photographers.encoder_hybrid                       # -> reports/photographers/encoder_hybrid.json
    python -m research.photographers.encoder_hybrid --encoders clip_b32 clip_b32_onnx_int8 --out reports/photographers/encoder_hybrid_int8.json

Protocol and gate: research/photographers/encoder_hybrid.md (pre-registered). Same rows, leakage groups, split,
classes, scorer and cross-subject protocol as subject_audit.py; matching and paired bootstrap from
research/c09/local_predictors.py (run_match, paired_ci; pooled over seeds with clusters (seed, group)).
Weights chosen on VAL cross-subject top-1; TEST read once per seed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from research.c09.local_predictors import paired_ci, run_match
from research.photographers.evaluate import EMB, HELD_OUT, l2, leakage_groups, load_rows, load_vectors, split_groups
from research.photographers.subject_audit import subjects

REF_W = (1.0, 0.5, 0.5)
M_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)
W_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
METRICS = ("top1", "top3", "mrr")


def prepare(seed: int, encoders: list[str]) -> dict:
    rows = load_rows(); ids = [r["work_id"] for r in rows]
    Xp = load_vectors("prod", ids)
    groups, _ = leakage_groups(rows, l2(Xp[:, :384])); split = split_groups(rows, groups, seed)
    pid = np.array([r["photographer_id"] for r in rows]); counts = Counter(pid.tolist())
    classes = sorted(p for p, n in counts.items() if n >= 40 and p not in HELD_OUT); cidx = {p: i for i, p in enumerate(classes)}
    y = np.array([cidx.get(p, -1) for p in pid]); s = subjects(ids)
    ok = (y >= 0) & (s != None)  # noqa: E711
    b, v, t = (np.flatnonzero(ok & (split == k)) for k in ("bank", "val", "test"))
    d = np.load(EMB / "descriptors.npz", allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
    order = [pos[w] for w in ids]
    ax = np.nan_to_num(d["axes"][order], nan=0.5); f205 = d["film205"][order]
    axes = l2(ax - ax[b].mean(0)); film = l2((f205 - f205[b].mean(0)) / (f205[b].std(0) + 1e-6))
    sims = {"prod": l2(Xp) @ l2(Xp).T, "axes17": axes @ axes.T, "film205": film @ film.T}
    for e in encoders:
        X = load_vectors(e, ids)
        if X is None:
            raise SystemExit(f"{e}.npz missing or incomplete")
        sims[e] = X @ X.T
    return dict(y=y, s=s, b=b, v=v, t=t, groups=groups, n_class=len(classes), sims=sims)


def variants(encoders: list[str]) -> dict:
    """name -> (parts, list of weight vectors, index of the tuned weight vector closest to the shipped choice)."""
    out = {"retuned_ref": (["prod", "axes17", "film205"], [(1.0, m, m) for m in M_GRID], 1)}
    for e in encoders:
        out[f"swap_{e}"] = ([e, "axes17", "film205"], [(1.0, m, m) for m in M_GRID], 1)
        out[f"add_{e}"] = (["prod", e, "axes17", "film205"], [(1.0, w, 0.5, 0.5) for w in W_GRID], 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["clip_b32", "siglip_base", "dinov2_base"])
    ap.add_argument("--out", default="reports/photographers/encoder_hybrid.json")
    a = ap.parse_args()
    V = variants(a.encoders)
    rep: dict = {"encoders": a.encoders, "reference": {"parts": ["prod", "axes17", "film205"], "weights": REF_W},
                 "grids": {"m": M_GRID, "w": W_GRID}, "per_seed": {}}
    per_query: dict = {}
    for seed in (0, 1):
        D = prepare(seed, a.encoders); y, s, b, v, t, nc, S = D["y"], D["s"], D["b"], D["v"], D["t"], D["n_class"], D["sims"]
        R = {"n": {"bank": len(b), "val": len(v), "test": len(t), "classes": nc}, "val": {}, "test": {}}
        ref_val = run_match([S[p] for p in ("prod", "axes17", "film205")], REF_W, y, s, v, b, nc, True)["top1"].mean()
        R["val"]["reference_cross_top1"] = float(ref_val)
        chosen = {}
        for name, (parts, grid, pref) in V.items():
            accs = [float(run_match([S[p] for p in parts], w, y, s, v, b, nc, True)["top1"].mean()) for w in grid]
            k = max(range(len(grid)), key=lambda i: (round(accs[i], 9), -abs(i - pref)))
            chosen[name] = grid[k]
            R["val"][name] = {"weights": dict(zip(parts, grid[k])), "cross_top1": accs[k], "grid_cross_top1": accs}
        print(f"seed {seed} VAL cross top-1: ref {ref_val:.3f} " + " ".join(f"{n} {R['val'][n]['cross_top1']:.3f}" for n in V), flush=True)
        # TEST, read once
        pq = {"groups": [f"{seed}:{g}" for g in D["groups"][t]]}
        for cross in (False, True):
            proto = "cross" if cross else "full"
            ref = run_match([S[p] for p in ("prod", "axes17", "film205")], REF_W, y, s, t, b, nc, cross)
            R["test"].setdefault(proto, {})["reference"] = {k: float(x.mean()) for k, x in ref.items()}
            pq[f"{proto}:reference"] = ref
            for name, (parts, _, _) in V.items():
                r = run_match([S[p] for p in parts], chosen[name], y, s, t, b, nc, cross)
                pq[f"{proto}:{name}"] = r
                R["test"][proto][name] = {**{k: float(x.mean()) for k, x in r.items()},
                                          "diff_vs_ref": {k: {"diff": round(float((r[k] - ref[k]).mean()), 4), "ci95": paired_ci(r[k] - ref[k], D["groups"][t])} for k in METRICS}}
        rep["per_seed"][str(seed)] = R; per_query[seed] = pq
    # primary candidate (pre-registered): highest mean VAL cross top-1 over seeds, control excluded
    val_mean = {n: float(np.mean([rep["per_seed"][str(sd)]["val"][n]["cross_top1"] for sd in (0, 1)])) for n in V}
    rep["val_mean_cross_top1"] = val_mean
    rep["primary_candidate"] = max((n for n in V if n != "retuned_ref"), key=lambda n: val_mean[n])
    pooled: dict = {}
    for proto in ("full", "cross"):
        g = np.concatenate([np.array(per_query[sd]["groups"]) for sd in (0, 1)])
        pooled[proto] = {"reference": {k: round(float(np.concatenate([per_query[sd][f"{proto}:reference"][k] for sd in (0, 1)]).mean()), 4) for k in METRICS}}
        for name in V:
            row = {}
            for k in METRICS:
                x = np.concatenate([per_query[sd][f"{proto}:{name}"][k] for sd in (0, 1)])
                d = x - np.concatenate([per_query[sd][f"{proto}:reference"][k] for sd in (0, 1)])
                row[k] = round(float(x.mean()), 4)
                row[f"d_{k}"] = {"diff": round(float(d.mean()), 4), "ci95": paired_ci(d, g)}
            pooled[proto][name] = row
    rep["pooled"] = pooled
    json.dump(rep, open(a.out, "w"), indent=1)
    print("primary candidate:", rep["primary_candidate"])
    for proto in ("full", "cross"):
        print(proto, "reference", pooled[proto]["reference"])
        for name in V:
            r = pooled[proto][name]
            print(f"  {proto:5s} {name:22s} top1 {r['top1']:.3f} d {r['d_top1']['diff']:+.3f} {r['d_top1']['ci95']}  "
                  f"top3 {r['top3']:.3f} d {r['d_top3']['diff']:+.3f} {r['d_top3']['ci95']}  mrr {r['mrr']:.3f} d {r['d_mrr']['diff']:+.3f} {r['d_mrr']['ci95']}")


if __name__ == "__main__":
    main()
