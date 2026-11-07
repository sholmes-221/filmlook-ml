#!/usr/bin/env python
"""E24: scanner and photographer effects in real film scans (CommonCatalog film_scan originals, train+val only).

  STUDENT_TAGS=style_fs_v2_real python scripts/scanner_effects.py   # -> reports/scanner_effects.json

Scanner family from the EXIF `camera` field (Noritsu / Fuji Frontier / flatbed-dedicated / other). For each
embedding (benchmark cache): uploader-grouped CV linear probe of the scanner family (vs majority), the same-scanner
share of the top-5 neighbours with the query's uploader excluded (vs prior), and the same-uploader share with
nothing excluded (photographer bias, vs prior). High scanner recoverability in a style embedding means that
"same film look" retrieval on real scans can be driven by the lab's scanner rather than the film.
Test-split scans are left out so this analysis never touches the final split.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from eval_commons_film import grouped_probe  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402


def family(camera):
    c = (camera or "").lower()
    if "noritsu" in c:
        return "noritsu"
    if "fuji" in c and ("sp-" in c or "slp" in c or "frontier" in c):
        return "fuji_frontier"
    if any(k in c for k in ("epson", "plustek", "canon", "coolscan", "nikon", "minolta", "imacon", "hasselblad", "pakon", "copibook")):
        return "flatbed_dedicated"
    return "other_unknown"


def share(E, lab, group, exclude_group):
    S = E @ E.T; out, prior = [], []
    for i in range(len(E)):
        ok = (group != group[i]) if exclude_group else (np.arange(len(E)) != i)
        order = np.argsort(-np.where(ok, S[i], -np.inf))[:5]
        out.append((lab[order] == lab[i]).mean()); prior.append((lab[ok] == lab[i]).mean())
    return {"share": round(float(np.mean(out)), 4), "prior": round(float(np.mean(prior)), 4)}


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load_cache(); man = {r["image_id"]: r for r in read_jsonl(P.MANIFESTS / "manifest.jsonl")}
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(F["hybrid"][tr], fam[tr], dev)
    m = np.array([r["set"] == "cc_original" and r.get("medium") == "film_scan" and r["split"] in ("train", "val") for r in rows])
    idx = np.where(m)[0]; ids = [rows[i]["id"] for i in idx]
    scanner = np.array([family(man[i].get("camera")) for i in ids]); up = np.array([man[i]["group_id"] for i in ids])
    cands = {"raw_dino_global": F["dino_global"][idx], "raw_hybrid": F["hybrid"][idx], "hybrid_supcon_family": proj(F["hybrid"][idx]), "student_fs_v2_64": F["student64"][idx],
             "colour_tone16": F["explicit"][idx][:, IDX_C], "texture12": F["explicit"][idx][:, IDX_T]}
    for tag in [t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]:
        cands[f"student_{tag}_64"] = np.load(P.EMBEDDINGS / f"student_{tag}_bench.npz")["student64"][idx]
    out = {"n": int(len(idx)), "scanner_families": dict(Counter(scanner)), "n_uploaders": int(len(set(up))), "candidates": {}}
    for name, Z in cands.items():
        E = unit(Z.astype(np.float32))
        out["candidates"][name] = {"scanner_probe": grouped_probe(Z, scanner, up), "same_scanner_share_top5_uploader_excluded": share(E, scanner, up, True),
                                   "same_uploader_share_top5": share(E, up, up, False)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "scanner_effects.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "scanner_families", "n_uploaders")}))
    print(f"{'candidate':26s} {'scanner acc':>12s} {'maj':>6s} {'scan share':>11s} {'prior':>6s} {'upl share':>10s} {'prior':>6s}")
    for name, r in out["candidates"].items():
        p, s, u = r["scanner_probe"], r["same_scanner_share_top5_uploader_excluded"], r["same_uploader_share_top5"]
        print(f"{name:26s} {p['accuracy'] or 0:12.3f} {p.get('majority', 0):6.3f} {s['share']:11.3f} {s['prior']:6.3f} {u['share']:10.3f} {u['prior']:6.3f}")


if __name__ == "__main__":
    main()
