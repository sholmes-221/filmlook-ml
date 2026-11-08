#!/usr/bin/env python
"""E50: does "similar look" mean similar rendering, similar scene, or the same provenance?

  python scripts/e50_aesthetic_vs_content.py   # CPU only (cached features) -> reports/e50_aesthetic_vs_content.json

On the commoncatalog_film analysis half (real film scans, uploader excluded everywhere) and the aic_photographs analysis
half (museum prints, artist excluded), for each representation:
  neighbour coherence   mean |difference| of the standardised explicit statistics between a query and its five nearest
                        neighbours, divided by the same quantity for five random candidates (20 draws); reported for the
                        16 colour-tone statistics, the 12 texture statistics, and for the 28 together. Below 1 means the
                        neighbours share rendering; 1 means the embedding ignores it.
  explained variance    uploader-grouped ridge R^2 of the embedding from (a) the 28 explicit rendering statistics,
                        (b) a dimension-matched 28-d PCA of the frozen DINOv2-Small CLS++mean embedding (scene content),
                        (c) provenance one-hots (scanner category x colour type; decade x process on the museum set).
  monochrome agreement  share of the five neighbours with the same monochrome flag as the query (prior reported).
Representations: the three E43 students' pooled PCA-64, the ImageNet-initialised pooled PCA-64, DINOv2 CLS++mean PCA-64,
DINOv2 L4+L8 PCA-64, the explicit statistics themselves (a ceiling for coherence). Reductions fitted on CC-BY train originals.
Pre-registered rule (research/backlog.yaml E50): supported = 1 iff for every seed, on the film scans, the colour-tone
coherence ratio is <= 0.8, the texture ratio is <= 0.9, and the rendering R^2 is at least the content R^2.
Analysis halves only; no test half, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from e29_audit import grouped_r2  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import INIT, REF, fit_embed  # noqa: E402
from style_benchmark import unit  # noqa: E402

from filmlook_infer.features import FEATURE_NAMES  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
MONO = FEATURE_NAMES.index("is_mono")


def coherence(E, Xs, group, n_draw=20, seed=0):
    """Neighbour / random ratio of mean |standardised attribute difference| (query's group excluded)."""
    rng = np.random.default_rng(seed); S = E @ E.T; nb, rd = [], []
    for i in range(len(E)):
        ok = group != group[i]
        if ok.sum() < 10:
            continue
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:5]; nb.append(np.abs(Xs[order] - Xs[i]).mean(0))
        cand = np.where(ok)[0]; rd.append(np.mean([np.abs(Xs[rng.choice(cand, 5, replace=False)] - Xs[i]).mean(0) for _ in range(n_draw)], 0))
    nb, rd = np.array(nb), np.array(rd)
    out = {}
    for name, idx in (("colour_tone", IDX_C), ("texture", IDX_T), ("all28", np.arange(Xs.shape[1]))):
        out[name] = round(float(nb[:, idx].mean() / rd[:, idx].mean()), 4)
    return out


def mono_agreement(E, mono, group):
    S = E @ E.T; agree, prior = [], []
    for i in range(len(E)):
        ok = group != group[i]
        if ok.sum() < 10:
            continue
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:5]; agree.append((mono[order] == mono[i]).mean()); prior.append((mono[ok] == mono[i]).mean())
    return {"agreement": round(float(np.mean(agree)), 4), "prior": round(float(np.mean(prior)), 4)}


def onehots(cols):
    out = []
    for c in cols:
        u = sorted(set(c)); out.append(np.array([[v == x for x in u] for v in c], np.float32))
    return np.hstack(out)


def main():
    t0 = time.time()
    fit_paths = [r["path"] for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby" and r.get("split") == "train" and r.get("edit_type") == "original"]
    ra, rg, rv, rb, L = M.labels(); A = dict(np.load(M.AIC.with_suffix(".npz"))); G = dict(np.load(M.CCF.with_suffix(".npz"))); Fit = dict(np.load(FIT.with_suffix(".npz")))
    mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6
    sets = {"film_scans": {"explicit": (G["explicit"][L["ig"]] - mu) / sd, "content": PCA64(Fit["dino_global"], 28)(G["dino_global"][L["ig"]]), "group": L["up"], "prov": onehots([L["scanner"], np.array([str(r.get("film_type")) for r in rg])]),
                           "mono": G["explicit"][L["ig"]][:, MONO] > 0.5},
            "museum_prints": {"explicit": (A["explicit"][L["ia"]] - mu) / sd, "content": PCA64(Fit["dino_global"], 28)(A["dino_global"][L["ia"]]), "group": L["artist"],
                              "prov": onehots([L["proc"], np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in ra])]), "mono": A["explicit"][L["ia"]][:, MONO] > 0.5}}
    reps = {}
    for tag in (*REF, INIT):
        za, zg, _, _ = M.student_features(f"{tag}:features_576", ra, rg, rv, rb, L["ia"], L["ig"]); p = PCA64(fit_embed(tag, fit_paths)); reps[f"{tag}:pca64"] = {"film_scans": p(zg), "museum_prints": p(za)}
    for key in ("dino_global", "dino_style"):
        p = PCA64(Fit[key]); reps[f"{key}:pca64"] = {"film_scans": p(G[key][L["ig"]]), "museum_prints": p(A[key][L["ia"]])}
    reps["explicit28_std"] = {"film_scans": sets["film_scans"]["explicit"], "museum_prints": sets["museum_prints"]["explicit"]}
    out = {"experiment": "E50_aesthetic_vs_content", "analysis_data_only": True, "n": {k: int(len(v["group"])) for k, v in sets.items()}, "representations": {}}
    for name, Z in reps.items():
        r = {}
        for sname, s in sets.items():
            E = unit(np.nan_to_num(Z[sname].astype(np.float32))); Xs = np.nan_to_num(s["explicit"])
            r[sname] = {"coherence_ratio": coherence(E, Xs, s["group"]), "mono": mono_agreement(E, s["mono"], s["group"]),
                        "r2": {"rendering_explicit28": grouped_r2(Xs, E, s["group"]), "content_dino_pca28": grouped_r2(s["content"], E, s["group"]), "provenance_onehots": grouped_r2(s["prov"], E, s["group"])}}
        out["representations"][name] = r; f = r["film_scans"]
        print(f"{name:34s} scans: coherence colour {f['coherence_ratio']['colour_tone']:.3f} texture {f['coherence_ratio']['texture']:.3f} | R2 rendering {f['r2']['rendering_explicit28']:.3f} content {f['r2']['content_dino_pca28']:.3f} provenance {f['r2']['provenance_onehots']:.3f} | mono {f['mono']['agreement']:.3f}/{f['mono']['prior']:.3f}"
              f" || prints: colour {r['museum_prints']['coherence_ratio']['colour_tone']:.3f} texture {r['museum_prints']['coherence_ratio']['texture']:.3f} R2 rend {r['museum_prints']['r2']['rendering_explicit28']:.3f} cont {r['museum_prints']['r2']['content_dino_pca28']:.3f}", flush=True)
    checks = {}
    for tag in REF:
        f = out["representations"][f"{tag}:pca64"]["film_scans"]
        checks[tag] = {"colour_ratio_le_0.8": f["coherence_ratio"]["colour_tone"] <= 0.8, "texture_ratio_le_0.9": f["coherence_ratio"]["texture"] <= 0.9, "rendering_r2_ge_content_r2": f["r2"]["rendering_explicit28"] >= f["r2"]["content_dino_pca28"]}
    ok = all(all(c.values()) for c in checks.values()); mean = lambda g: round(float(np.mean([g(out["representations"][f"{t}:pca64"]["film_scans"]) for t in REF])), 4)  # noqa: E731
    out["checks"] = checks
    out["summary"] = {"supported": int(ok), "seeds_passing": int(sum(all(c.values()) for c in checks.values())), "student_colour_ratio_mean": mean(lambda f: f["coherence_ratio"]["colour_tone"]), "student_texture_ratio_mean": mean(lambda f: f["coherence_ratio"]["texture"]),
                      "student_rendering_r2_mean": mean(lambda f: f["r2"]["rendering_explicit28"]), "student_content_r2_mean": mean(lambda f: f["r2"]["content_dino_pca28"]), "student_provenance_r2_mean": mean(lambda f: f["r2"]["provenance_onehots"]),
                      "student_mono_agreement_mean": mean(lambda f: f["mono"]["agreement"]), "mono_prior": out["representations"][f"{REF[0]}:pca64"]["film_scans"]["mono"]["prior"],
                      **{f"{n}_{k}": out["representations"][n]["film_scans"]["coherence_ratio"]["colour_tone"] if k == "colour_ratio" else out["representations"][n]["film_scans"]["r2"]["content_dino_pca28"] for n in (f"{INIT}:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std") for k in ("colour_ratio", "content_r2")},
                      "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "e50_aesthetic_vs_content.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
