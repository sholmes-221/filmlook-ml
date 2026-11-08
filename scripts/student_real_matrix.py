#!/usr/bin/env python
"""Real-photograph evaluation matrix for lightweight students (analysis halves only; no test half, no BlueNeg).

  python scripts/student_real_matrix.py --tags style_fs_v2_real_vic3,style_fs_v2_real_pcaw --reference style_fs_v2_real
      -> reports/student_real_matrix.json

For each student (64-d `style_embedding` from its ONNX export, computed on demand and cached under
data/embeddings/student_matrix_<tag>.npz) the matrix separates the signals the project must not confuse:
  museum_p5            process retrieval on aic_photographs, artist excluded, stereo cards removed, decade matched
  artist_share         photographer clustering on the museum set (top-5 share, prior ~0.017)
  stock_line_p5        stock-line retrieval on commoncatalog_film, uploader excluded (chance = class prior)
  uploader_share       photographer clustering on the film scans (prior ~0.03)
  scanner_probe        scanner-category linear probe, uploader-grouped (leakage guard; majority 0.44)
  photographer_p5      same-uploader retrieval among candidates with disjoint subject tags and aspect ratio within 10 %
  within_uploader_auc  same-stock-line vs other-line ranking inside an uploader, other days only (chance 0.5)
  colour_only_auc      the same restricted to colour scans (removes the monochrome cue)
  synth_family_p5      synthetic family retrieval on the filmsynth validation split, source excluded (cost of the trade)
  own_source_rank1     content leakage: a treated validation image retrieving its own untreated source among the 15 validation
                       sources (chance 0.067; above 0.5 = scene content dominates; E40)
  eff_rank             effective rank of the embedding on the film scans (collapse diagnostic)
Group bootstrap CIs everywhere. The script never reads a test half.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment_v3 as V3  # noqa: E402
from e29_audit import eff_rank, retrieval  # noqa: E402
from eval_commons_film import boot_groups, grouped_probe  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from frozen_head_choice import share  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402
from within_photographer_stock import within_auc  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
CACHE_KEY = {"style_fs_v2": "student64"}


def embed(tag, paths, real, output="style_embedding"):
    """Embeddings of checkpoints/student_<tag> for the given image paths (CPU ONNX): the 64-d `style_embedding` or, for
    students exported after E42, the 576-d pooled `features_576`."""
    import onnxruntime as ort
    from PIL import Image
    from filmlook_infer.compact import preprocess
    ck = P.CHECKPOINTS / f"student_{tag}"; sess = ort.InferenceSession(str(ck / "model.onnx"), providers=["CPUExecutionProvider"])
    oi = json.loads((ck / "meta.json").read_text())["outputs"].index(output)
    with Pool(16) as pool:
        arrs = pool.map(V3._load, [(p, str(i), real, real) for i, p in enumerate(paths)], chunksize=8)
    acc = []
    for i in range(0, len(arrs), 64):
        acc.append(sess.run(None, {"pixel_values": np.concatenate([preprocess(Image.fromarray(x)) for x in arrs[i:i + 64]])})[oi])
    return np.concatenate(acc).astype(np.float32)


def student_features(tag, rows_a, rows_g, rows_v, rows_b, ia, ig):
    """Embeddings on the three evaluation sets; existing caches (full sets, subset by the analysis indices) are reused
    for the older students. A tag suffixed `:features_576` selects the pooled features of an E42-style export."""
    tag, output = (tag.split(":", 1) + ["style_embedding"])[:2]
    A = dict(np.load(AIC.with_suffix(".npz"))); G = dict(np.load(CCF.with_suffix(".npz"))); key = CACHE_KEY.get(tag, "student_" + tag)
    cache = P.EMBEDDINGS / f"student_matrix_{tag}{'' if output == 'style_embedding' else '_' + output}.npz"
    if cache.exists() and "bank" in np.load(cache).files:
        z = np.load(cache); return z["aic"], z["ccf"], z["val"], z["bank"]
    if key in A and key in G and output == "style_embedding":
        za, zg = A[key][ia], G[key][ig]
    else:
        za, zg = embed(tag, [r["path"] for r in rows_a], True, output), embed(tag, [r["path"] for r in rows_g], True, output)
    zv = embed(tag, [r["path"] for r in rows_v], False, output); zb = embed(tag, [r["path"] for r in rows_b], True, output)
    np.savez(cache, aic=za, ccf=zg, val=zv, bank=zb); return za, zg, zv, zb


def matrix(za, zg, zv, L, zb=None):
    Ea, Eg, Ev = (unit(np.nan_to_num(z.astype(np.float32))) for z in (za, zg, zv))
    rank1 = None
    if zb is not None:
        Eb = unit(np.nan_to_num(zb.astype(np.float32))); S = Ev @ Eb.T; rank1 = round(float(np.mean([L["src_b"][S[i].argmax()] == s for i, s in enumerate(L["src_v"])])), 4)
    a_all, g_all = within_auc(Eg[L["kmulti"]], L["line_m"], L["up_m"], L["day_m"], True)
    a_col, g_col = within_auc(Eg[L["kmulti"]], L["line_m"], L["up_m"], L["day_m"], True, L["colour_m"])
    return {"museum_p5": retrieval(Ea, L["proc"], L["artist"], L["allowed"]), "artist_share": share(Ea, L["artist"]), "stock_line_p5": retrieval(Eg, L["line"], L["up"]), "uploader_share": share(Eg, L["up"]),
            "scanner_probe": grouped_probe(Eg[L["has"]], L["scanner"][L["has"]], L["up"][L["has"]]), "photographer_p5": retrieval(Eg, L["up"], np.arange(len(Eg)), L["photog_ok"]),
            "within_uploader_auc": {"auc": round(float(a_all.mean()), 4), "ci95": boot_groups(a_all, g_all), "n_queries": int(len(a_all))} if len(a_all) else None,
            "colour_only_auc": {"auc": round(float(a_col.mean()), 4), "ci95": boot_groups(a_col, g_col), "n_queries": int(len(a_col))} if len(a_col) else None,
            "synth_family_p5": retrieval(Ev, L["fam_v"], L["src_v"]), "own_source_rank1": rank1, "eff_rank": eff_rank(Eg)}


def labels():
    from collections import Counter
    rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]; rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    fs_val = (st == "filmsynth") & (sp == "val") & treated; rows_v = [r for r, k in zip(rows_c, fs_val) if k]
    bank = (st == "source") & (sp == "val"); rows_b = [r for r, k in zip(rows_c, bank) if k]
    ia = np.array([i for i, r in enumerate(rows_a) if r["split"] == "analysis"]); ig = np.array([i for i, r in enumerate(rows_g) if r["split"] == "analysis" and r.get("film_stock")])
    ra = [rows_a[i] for i in ia]; rg = [rows_g[i] for i in ig]
    artist = np.array([r["uploader"] for r in ra]); proc = np.array([process_family(r.get("process")) for r in ra]); stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in ra])
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in ra]); allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    up = np.array([r["uploader"] for r in rg]); line = np.array([r.get("film_stock_line") or r["film_stock_claim"] for r in rg]); scanner = np.array([r.get("scanner_category") or "" for r in rg])
    day = np.array([str(r.get("datetaken") or "?")[:10] for r in rg]); colour = np.array([r.get("film_type") in ("colour_negative", "colour_reversal") for r in rg])
    subj = [set(r.get("subjects") or []) for r in rg]; lar = np.array([np.log(r["image_width"] / r["image_height"]) for r in rg])
    photog_ok = np.array([[not (a & b) for b in subj] for a in subj]) & (np.abs(lar[:, None] - lar[None]) < 0.1)
    per = {u: Counter(line[up == u]) for u in set(up)}; kmulti = np.isin(up, [u for u, c in per.items() if len(c) >= 2 and sum(c.values()) >= 4])
    return ra, rg, rows_v, rows_b, {"ia": ia, "ig": ig, "src_b": np.array([str(r["source"]) for r in rows_b]), "artist": artist, "proc": proc, "allowed": allowed, "up": up, "line": line, "scanner": scanner, "has": scanner != "", "photog_ok": photog_ok, "kmulti": kmulti,
                            "line_m": line[kmulti], "up_m": up[kmulti], "day_m": day[kmulti], "colour_m": colour[kmulti], "fam_v": np.array([r["family"] for r in rows_v]), "src_v": np.array([str(r["source"]) for r in rows_v])}


def fmt(name, r):
    m, s, w, c, v = r["museum_p5"], r["stock_line_p5"], r["within_uploader_auc"], r["colour_only_auc"], r["synth_family_p5"]
    return (f"{name:28s} museum {m['precision@5']:.3f} {m['ci95']} ch {m['chance']:.3f} art {r['artist_share']:.3f} | stock {s['precision@5']:.3f} {s['ci95']} ch {s['chance']:.3f} up {r['uploader_share']:.3f} "
            f"scan {r['scanner_probe']['accuracy'] or 0:.3f} photog {r['photographer_p5']['precision@5'] or 0:.3f} | within {w['auc']:.3f} {w['ci95']} colour {c['auc']:.3f} {c['ci95']} | synth {v['precision@5']:.3f} rank1 {r['own_source_rank1'] if r['own_source_rank1'] is not None else -1:.3f} | erank {r['eff_rank']}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--reference", default="style_fs_v2_real"); ap.add_argument("--out", default="student_real_matrix.json")
    a = ap.parse_args(); t0 = time.time()
    ra, rg, rv, rb, L = labels(); out = {"analysis_data_only": True, "n_aic": len(ra), "n_ccf": len(rg), "n_fs_val": len(rv), "n_bank": len(rb), "reference": a.reference, "students": {}}
    for tag in [a.reference, *[t for t in a.tags.split(",") if t]]:
        za, zg, zv, zb = student_features(tag, ra, rg, rv, rb, L["ia"], L["ig"]); r = matrix(za, zg, zv, L, zb); out["students"][tag] = r; print(fmt(tag, r), flush=True)
    out["seconds"] = round(time.time() - t0, 1); P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / a.out).write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
