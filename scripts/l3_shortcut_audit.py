#!/usr/bin/env python
"""E28: is the layer-3 teacher's synthetic gain a shortcut? Audit on existing data only (no Commons needed).

  python scripts/l3_shortcut_audit.py        # -> reports/l3_shortcut_audit.json

Background. E26/E27 found that explicit28 ++ DINOv2-Small layer-3 patch mean/std (family SupCon on filmsynth TRAIN)
beats the L4+L8 teacher on cross-scene treatment retrieval (val B filmsynth 0.572 vs 0.467) with less content
leakage, but the gain was absent on the held-out generators (sdxl 0.324 vs 0.347, zimage 0.364 vs 0.342) and the
generator share rose (0.594 vs 0.541). The real-film check planned for it (E27, Commons) is blocked by rate limiting.
Three pre-registered checks that need no new data (features come from the E26 per-layer cache and the benchmark cache):

 C1 generator-swap (synthetic-shortcut check). Train the SupCon projection on the sdxl TRAIN families instead of
    filmsynth and evaluate B on the val split, where filmsynth and zimage are now held-out generators. If L3's
    advantage is a filmsynth-pipeline fingerprint it vanishes under the swap. Rule: `swap_status` = "persists" when
    L3's filmsynth-val lower CI > baseline's point estimate (both sdxl-trained); "vanishes" when L3's point estimate
    <= baseline's upper CI; else "inconclusive".
 C2 scanner leakage (real-film check without stock labels). CommonCatalog film_scan originals, train+val only (E24
    protocol: uploader-grouped linear probe of the scanner family). Rule: L3 probe accuracy <= baseline + 0.05.
 C3 photographer leakage. Same scans: same-uploader share of the top-5 (nothing excluded). Rule: L3 share <=
    baseline + 0.02.
`summary.passes` = 1 iff swap_status != "vanishes" and C2 and C3 hold, else 0 (falsified). Passing is NOT evidence
of real-film transfer (that remains E27); it only means the L3 candidate survives the shortcut audit. The students
student_style_l3_real (E27a) and student_style_fs_v2_real (E23b) are reported on C2/C3 as observations. No test
split, no BlueNeg blue-intact, no Commons data is touched.
"""
import json
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from eval_commons_film import grouped_probe  # noqa: E402
from scanner_effects import family, share  # noqa: E402
from style_benchmark import evaluate, load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

CACHE = P.EMBEDDINGS / "bench_layers_v1.npz"
STUDENTS = {"student_style_fs_v2_real_64 (E23b)": "style_fs_v2_real", "student_style_l3_real_64 (E27a)": "style_l3_real"}


def b_of(r, s):
    b = r["B_same_treatment_across_scenes"][s]; return {"precision@5": b["precision@5"], "ci95": b["ci95"], "chance": b.get("chance")}


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load_cache(); z = np.load(CACHE); assert (z["ids"] == np.array([r["id"] for r in rows])).all(), "per-layer cache mismatch"
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    feats = {"L4L8_ms (baseline)": F["dino_style"].astype(np.float32), "L3_ms": z["ms_L3"].astype(np.float32)}
    hyb = {n: np.hstack([F["explicit"], Z]) for n, Z in feats.items()}
    out = {"experiment": "E28_l3_shortcut_audit", "analysis_data_only": True, "generator_swap": {}, "real_scans": {}}
    # C1: projection trained on filmsynth (reference, as E27) and on sdxl (swap); evaluated on the val split
    for train_set in ("filmsynth", "sdxl"):
        tr = (st == train_set) & (sp == "train"); res = {}
        for name, H in hyb.items():
            proj = X.supcon(H[tr], fam[tr], dev); E = proj(H); r = evaluate(E, F, rows, "val")
            res[name] = {"B": {s: b_of(r, s) for s in ("filmsynth", "sdxl", "zimage")}, "C_generator_share": r["C_generator_fingerprint"]["same_generator_share_top5"],
                         "E_rank1": r["E_semantic_leakage"]["own_source_rank1"], "E_subject_probe": r["E_semantic_leakage"]["subject_probe_accuracy"]}
            print(f"[proj on {train_set:9s}] {name:20s} B fs {res[name]['B']['filmsynth']['precision@5']:.3f} {res[name]['B']['filmsynth']['ci95']} sdxl {res[name]['B']['sdxl']['precision@5']:.3f} "
                  f"zimage {res[name]['B']['zimage']['precision@5']:.3f}  C {res[name]['C_generator_share']:.3f}  E r1 {res[name]['E_rank1']:.3f}", flush=True)
        out["generator_swap"][f"proj_trained_on_{train_set}"] = {"n_train": int(tr.sum()), "n_families": int(len(set(fam[tr]))), "results": res}
    sw = out["generator_swap"]["proj_trained_on_sdxl"]["results"]; l3, bs = sw["L3_ms"]["B"]["filmsynth"], sw["L4L8_ms (baseline)"]["B"]["filmsynth"]
    swap_status = "persists" if l3["ci95"][0] > bs["precision@5"] else ("vanishes" if l3["precision@5"] <= bs["ci95"][1] else "inconclusive")
    # C2/C3: real film scans (CommonCatalog film_scan originals, train+val), filmsynth-trained projections as deployed
    man = {r["image_id"]: r for r in read_jsonl(P.MANIFESTS / "manifest.jsonl")}
    m = np.array([r["set"] == "cc_original" and r.get("medium") == "film_scan" and r["split"] in ("train", "val") for r in rows]); idx = np.where(m)[0]
    ids = [rows[i]["id"] for i in idx]; scanner = np.array([family(man[i].get("camera")) for i in ids]); up = np.array([man[i]["group_id"] for i in ids])
    trf = (st == "filmsynth") & (sp == "train"); cands = {}
    for name, H in hyb.items():
        cands[name] = X.supcon(H[trf], fam[trf], dev)(H)[idx]
    cands["explicit28_supcon (no DINO)"] = X.supcon(F["explicit"][trf], fam[trf], dev)(F["explicit"])[idx]
    cands["L3_ms_only_supcon (no explicit)"] = X.supcon(feats["L3_ms"][trf], fam[trf], dev)(feats["L3_ms"])[idx]
    for name, tag in STUDENTS.items():
        p = P.EMBEDDINGS / f"student_{tag}_bench.npz"
        if p.exists():
            s = np.load(p); assert (s["ids"] == np.array([r["id"] for r in rows])).all(); cands[name] = s["student64"][idx]
    rs = {}
    for name, Z in cands.items():
        E = unit(Z.astype(np.float32))
        rs[name] = {"scanner_probe": grouped_probe(Z, scanner, up), "same_scanner_share_top5_uploader_excluded": share(E, scanner, up, True), "same_uploader_share_top5": share(E, up, up, False)}
        p, s, u = rs[name]["scanner_probe"], rs[name]["same_scanner_share_top5_uploader_excluded"], rs[name]["same_uploader_share_top5"]
        print(f"[real scans] {name:34s} scanner acc {p['accuracy']:.3f} (maj {p['majority']:.3f}) scan share {s['share']:.3f} (prior {s['prior']:.3f}) uploader share {u['share']:.4f} (prior {u['prior']:.4f})", flush=True)
    out["real_scans"] = {"n": int(len(idx)), "n_uploaders": int(len(set(up))), "scanner_families": dict(Counter(scanner)), "candidates": rs}
    base, cand = rs["L4L8_ms (baseline)"], rs["L3_ms"]
    c2 = cand["scanner_probe"]["accuracy"] <= base["scanner_probe"]["accuracy"] + 0.05
    c3 = cand["same_uploader_share_top5"]["share"] <= base["same_uploader_share_top5"]["share"] + 0.02
    stu = {n: rs[n] for n in STUDENTS if n in rs}
    summary = {"swap_status": swap_status, "swap_L3_minus_baseline_filmsynth_val": round(l3["precision@5"] - bs["precision@5"], 4),
               "scanner_ok": bool(c2), "scanner_acc_L3_minus_baseline": round(cand["scanner_probe"]["accuracy"] - base["scanner_probe"]["accuracy"], 4),
               "uploader_ok": bool(c3), "uploader_share_L3_minus_baseline": round(cand["same_uploader_share_top5"]["share"] - base["same_uploader_share_top5"]["share"], 4),
               "passes": int(swap_status != "vanishes" and c2 and c3), "students_observed": list(stu), "seconds": round(time.time() - t0, 1)}
    out["summary"] = summary
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "l3_shortcut_audit.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
