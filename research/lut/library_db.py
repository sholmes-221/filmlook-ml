"""Ingest the discovered LUT library (research/lut_library/manifest.json) into separate style databases and compare LUTs.

    python -m research.lut.library_db build --which library   # -> research/lut/db_library/  (all sRGB film-look LUTs; CC BY-SA caveat)
    python -m research.lut.library_db build --which free      # -> research/lut/db_free/     (owner's 13 presets + CC BY / CC0 LUTs)
    python -m research.lut.library_db compare                 # -> reports/lut/library_clusters.json (dE between lattices, clustering, per-stock table)

Ingestion rules (every decision is recorded in the database json under `ingestion`):
  * role must be `film_look` (creative looks, the identity LUT and the negative inversion are skipped);
  * format haldclut_png / haldclut_tif (LUT3D.from_hald) or .cube (LUT3D.from_cube); RawTherapee .pp3 profiles are not LUTs;
  * the input colour space must be sRGB display-referred. The 8 filmlab Vision3 LUTs expect DaVinci Intermediate /
    DaVinci Wide Gamut input and a density-wire print stage; no verified DI/DWG transform is implemented here, so they are
    skipped rather than applied to sRGB (which would be wrong);
  * lattices are resampled to the 33^3 working grid; LUT3D.validate must pass (finite, in range, grey-monotone).
The vendored 13-preset database (research/lut/db) is never touched. db_library stores lattices at 17^3 (size) and no
inverse lattices; db_free stores the full 33^3 lattices (inverses optional, `--no-inverse` keeps it at ~12 MB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.lut.engine.lut3d import LUT3D
from research.lut.engine.lut_db import LATTICE, build_from_styles, preset_styles

MANIFEST = Path("research/lut_library/manifest.json")
FREE_LICENSES = {"CC-BY-4.0", "CC0-1.0", "CC-BY-2.0", "PDM-1.0"}
COLOUR_TYPES = {"colour_negative", "colour_negative_cine", "colour_reversal", "instant", "instant_colour", "colour_negative_special", "camera_simulation"}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def ingest(records: list[dict], only_free: bool = False) -> tuple[list[dict], list[LUT3D], list[dict]]:
    styles, luts, skipped = [], [], []
    seen = set()
    for r in records:
        reason = None
        if r.get("role") != "film_look":
            reason = f"role {r.get('role')}"
        elif r.get("format") == "rawtherapee_pp3":
            reason = "RawTherapee .pp3 profile, not a LUT (the matching HaldCLUT TIFF is ingested)"
        elif not str(r.get("input_color_space", "")).lower().startswith("srgb"):
            reason = f"input colour space not sRGB ({r.get('input_color_space')[:60]}...): no verified transform, skipped"
        elif only_free and r.get("license") not in FREE_LICENSES:
            reason = f"licence {r.get('license')} excluded from the free database"
        elif not r.get("validation", {}).get("ok", False):
            reason = "library validation not ok"
        if reason:
            skipped.append({"id": r["id"], "reason": reason})
            continue
        sid = r["id"].replace("/", "__")
        if sid in seen:
            skipped.append({"id": r["id"], "reason": "duplicate id"})
            continue
        try:
            lut = LUT3D.from_hald(r["path"]) if r["format"].startswith("haldclut") else LUT3D.from_cube(r["path"])
        except Exception as e:  # noqa: BLE001
            skipped.append({"id": r["id"], "reason": f"parse error: {e}"})
            continue
        if lut.size != LATTICE:
            lut = lut.resample(LATTICE)
        lut.name = r["name"]
        seen.add(sid)
        luts.append(lut)
        variant = " ".join(r.get("variant_tokens") or [])
        styles.append({"id": sid, "name": r["name"], "library_id": r["id"], "source": r["source_slug"], "format": r["format"], "path": r["path"],
                       "license": r["license"], "license_url": r.get("license_url"), "attribution": r.get("attribution_text"),
                       "film_stock": r.get("film_stock"), "film_brand": r.get("film_brand"), "film_type": r.get("film_type"),
                       "variant_tokens": r.get("variant_tokens") or [], "push_pull_stops": r.get("push_pull_stops"), "variant": variant or None,
                       "original_lut_size": r.get("lut_size"), "label_basis": "LUT-derived aesthetic",
                       "approximation": ("Exact lattice from the published LUT (resampled to 33^3, trilinear). The LUT itself is its author's "
                                         "subjective approximation of the film look; see the source technical notes."),
                       "technical_notes": r.get("technical_notes"), "explanation_phrases": []})
    return styles, luts, skipped


def cmd_build(a):
    m = load_manifest()
    if a.which == "library":
        styles, luts, skipped = ingest(m["luts"], only_free=False)
        extra = {"database": "db_library", "ingestion": {"n_records": len(m["luts"]), "n_ingested": len(styles), "skipped": skipped},
                 "licence_decision_point": ("Most ingested LUTs are CC BY-SA 4.0 (RawTherapee collection). Whether a descriptor database or a model "
                                            "derived from them is an 'adaptation' that must be released under CC BY-SA 4.0 is unresolved; the owner "
                                            "must decide before anything derived from this database ships in the proprietary website. db_free "
                                            "contains only the owner's presets and CC BY / CC0 LUTs (attribution required)."),
                 "sources": m["sources"]}
        build_from_styles(styles, luts, "research/lut/db_library", n_ref=a.n_ref, with_inverse=False, lattice_store=17, extra_meta=extra)
    else:
        p_styles, p_luts, presets_by_id = preset_styles(a.presets)
        f_styles, f_luts, skipped = ingest(m["luts"], only_free=True)
        extra = {"database": "db_free", "ingestion": {"n_records": len(m["luts"]), "n_free_luts": len(f_styles), "n_presets": len(p_styles),
                                                      "skipped": [s for s in skipped if "licence" not in s["reason"]],
                                                      "attributions": sorted({s["attribution"] for s in f_styles if s["attribution"]})},
                 "licence_note": "Owner's presets + CC BY 4.0 (GlassMeter) + CC0 (sguyader) LUTs: shippable with the attribution texts listed."}
        build_from_styles(p_styles + f_styles, p_luts + f_luts, "research/lut/db_free", n_ref=a.n_ref, with_inverse=not a.no_inverse, presets_by_id=presets_by_id, extra_meta=extra)


def cmd_compare(a):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from research.lut.forward_model import srgb_to_lab
    m = load_manifest()
    styles, luts, _ = ingest(m["luts"])
    p_styles, p_luts, _ = preset_styles(a.presets)
    nat = np.load("data/lut/natural_pixels.npy")[:4000]
    colour = [i for i, s in enumerate(styles) if s["film_type"] in COLOUR_TYPES]
    print(f"{len(styles)} library LUTs, {len(colour)} colour film looks, {len(p_styles)} presets")
    all_luts = [luts[i] for i in colour] + p_luts
    all_meta = [styles[i] for i in colour] + [{"id": s["id"], "name": s["name"], "film_stock": s["id"], "source": "owner_preset", "film_type": "preset"} for s in p_styles]
    labs = np.stack([srgb_to_lab(l.apply(nat).astype(np.float64)) for l in all_luts])      # (N, P, 3)
    N = len(all_luts)
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = np.linalg.norm(labs - labs[i][None], axis=2).mean(1)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0)
    L = len(colour)
    Dl = D[:L, :L]
    # clustering of the library colour LUTs on the dE matrix
    best = None
    for k in range(4, min(40, L - 1)):
        lab_k = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(Dl)
        sc = silhouette_score(Dl, lab_k, metric="precomputed")
        if best is None or sc > best[1]:
            best = (k, sc, lab_k)
    k, sil, clusters = best
    stocks = np.array([all_meta[i]["film_stock"] for i in range(L)])
    per_stock = {}
    for st in sorted(set(stocks)):
        idx = np.where(stocks == st)[0]
        row = {"n_luts": int(len(idx)), "sources": sorted({all_meta[i]["source"] for i in idx}), "film_type": all_meta[idx[0]]["film_type"]}
        others = np.where(stocks != st)[0]
        if len(idx) > 1:
            within = Dl[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
            row["within_stock_mean_de"] = float(within.mean())
            row["within_stock_max_de"] = float(within.max())
        between = Dl[np.ix_(idx, others)]
        row["nearest_other_stock_mean_de"] = float(between.min(1).mean())
        nn = Dl[idx][:, :].copy()
        nn[np.arange(len(idx)), idx] = np.inf
        nn_idx = nn.argmin(1)
        row["nn_same_stock_rate"] = float((stocks[nn_idx] == st).mean()) if len(idx) > 1 else None
        row["nearest_other_stock"] = str(stocks[others][between.min(0).argsort()[0]]) if len(others) else None
        cl = clusters[idx]
        row["clusters"] = sorted(set(int(c) for c in cl))
        row["cluster_majority_share"] = float(np.bincount(cl).max() / len(cl))
        # the owner's preset for this stock, if any
        pm = [j for j in range(L, N) if all_meta[j]["film_stock"] == st or all_meta[j]["film_stock"].replace("kodak_color_plus_200", "kodak_colorplus_200") == st]
        if pm:
            j = pm[0]
            d_same = D[j, idx]
            rank = int((D[j, :L] < d_same.min()).sum()) + 1
            row["owner_preset"] = {"id": all_meta[j]["id"], "mean_de_to_same_stock_luts": float(d_same.mean()), "min_de": float(d_same.min()),
                                   "rank_of_nearest_same_stock_lut_among_library": rank, "nearest_library_lut": all_meta[int(D[j, :L].argmin())]["library_id"],
                                   "nearest_library_stock": str(stocks[int(D[j, :L].argmin())])}
        per_stock[st] = row
    # presets with no library counterpart: nearest library stock
    preset_nn = {}
    for j in range(L, N):
        order = D[j, :L].argsort()[:3]
        preset_nn[all_meta[j]["id"]] = [(all_meta[i]["library_id"], str(stocks[i]), round(float(D[j, i]), 2)) for i in order]
    offdiag = Dl[np.triu_indices(L, 1)]
    out = {"n_library_colour_luts": L, "n_presets": len(p_styles), "n_pixels": int(len(nat)), "de_offdiag_mean": float(offdiag.mean()), "de_offdiag_min": float(offdiag.min()),
           "de_offdiag_p10": float(np.percentile(offdiag, 10)), "clustering": {"method": "agglomerative average linkage on mean dE76 (natural pixels)", "n_clusters": int(k), "silhouette": float(sil),
                                                                             "cluster_sizes": np.bincount(clusters).tolist()},
           "within_stock_mean_de_overall": float(np.mean([v["within_stock_mean_de"] for v in per_stock.values() if "within_stock_mean_de" in v])),
           "nn_same_stock_rate_overall": float(np.mean([v["nn_same_stock_rate"] for v in per_stock.values() if v["nn_same_stock_rate"] is not None])),
           "per_stock": per_stock, "preset_nearest_library": preset_nn,
           "lut_ids": [all_meta[i]["library_id"] for i in range(L)], "clusters": clusters.tolist(), "de_matrix_colour_luts": Dl.round(2).tolist()}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"colour LUTs {L}: mean pairwise dE {out['de_offdiag_mean']:.2f} (min {out['de_offdiag_min']:.2f}); clusters {k} (silhouette {sil:.3f}); "
          f"within-stock mean dE {out['within_stock_mean_de_overall']:.2f}; NN same-stock rate {out['nn_same_stock_rate_overall']:.2f}")
    print(f"{'stock':28} {'n':>3} {'within dE':>9} {'nearest other':>13} {'NN same':>7} {'clusters':>9} {'preset dE/rank':>15}")
    for st, v in sorted(per_stock.items(), key=lambda kv: -kv[1]["n_luts"]):
        op = v.get("owner_preset")
        op_txt = f"{op['min_de']:.1f} / {op['rank_of_nearest_same_stock_lut_among_library']}" if op else ""
        nn_rate = v["nn_same_stock_rate"] if v["nn_same_stock_rate"] is not None else float("nan")
        print(f"{st:28} {v['n_luts']:3d} {v.get('within_stock_mean_de', float('nan')):9.2f} {v['nearest_other_stock_mean_de']:13.2f} "
              f"{nn_rate:7.2f} {str(v['clusters'])[:9]:>9} {op_txt:>15}")
    print("presets -> nearest library LUTs:", json.dumps(preset_nn, indent=0)[:1500])
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--which", choices=["library", "free"], required=True)
    b.add_argument("--n-ref", type=int, default=32)
    b.add_argument("--presets", default="data/lut/presets.json")
    b.add_argument("--no-inverse", action="store_true", help="skip the inverse lattices (approach B) to keep the database small")
    c = sub.add_parser("compare")
    c.add_argument("--presets", default="data/lut/presets.json")
    c.add_argument("--out", default="reports/lut/library_clusters.json")
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_compare(a)


if __name__ == "__main__":
    main()
