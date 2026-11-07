#!/usr/bin/env python
"""E29: cross-source robustness on curated CC0 museum photographs (aic_photographs), analysis half only.

  python scripts/eval_curated_photos.py --features          # reuses eval_commons_film.do_features on the aic manifest (GPU, one pass)
  python scripts/eval_curated_photos.py --split analysis    # -> reports/aic_photographs_analysis.json

Labels are museum catalogue records of the physical print process (medium_display, coarsened to a few families), never
aesthetic judgements. For every candidate embedding: process precision@5 with the artist excluded (chance = class prior,
artist-bootstrap CI), artist share of the top-5 (photographer leakage; prior = expected share), and the agreement between
the real-distil student and its teacher (Spearman of pairwise cosines on a fixed random pair sample). Pre-registered rule
(research/backlog.yaml E29): falsified for a candidate when its artist-share excess over hybrid_supcon_family exceeds 0.02
or its process p@5 lower CI is below chance. The test half is not read.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import eval_commons_film as EC  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from eval_commons_film import boot_groups, retrieval  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402
import synth_experiment as X  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

DS = "aic_photographs"
PROCESS = [(r"gelatin silver|silver gelatin", "gelatin_silver"), (r"albumen", "albumen"), (r"salted paper|salt print", "salted_paper"), (r"platinum|palladium", "platinum"),
           (r"chromogenic|dye coupler|c-print|type c", "chromogenic"), (r"cyanotype", "cyanotype"), (r"photogravure|gravure", "photogravure"), (r"daguerreotype", "daguerreotype"),
           (r"collodion|ambrotype|tintype", "collodion"), (r"carbon print", "carbon"), (r"gum bichromate", "gum"), (r"dye transfer|dye imbibition", "dye_transfer"), (r"inkjet|pigment print", "inkjet")]


def process_family(medium: str | None) -> str:
    m = (medium or "").lower()
    for rx, fam in PROCESS:
        if re.search(rx, m):
            return fam
    return "other"


def agreement(A: np.ndarray, B: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed); n = len(A); i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs); keep = i != j
    a, b = unit(A), unit(B); return round(float(spearmanr((a[i] * a[j]).sum(1)[keep], (b[i] * b[j]).sum(1)[keep]).correlation), 4)


def do_eval(split: str = "analysis"):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = dict(np.load(EC.OUT.with_suffix(".npz"))); rows = json.loads(EC.OUT.with_suffix(".json").read_text())["rows"]
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam_c = np.array([r["family"] for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(Fc["hybrid"][tr], fam_c[tr], dev)
    hyb = np.hstack([F["explicit"], F["dino_style"]])
    cands = {"raw_dino_global": F["dino_global"], "raw_hybrid": hyb, "hybrid_supcon_family": proj(hyb), "student_fs_v2_64": F["student64"],
             "colour_tone16": F["explicit"][:, IDX_C], "texture12": F["explicit"][:, IDX_T]}
    for t in [t for t in os.environ.get("STUDENT_TAGS", "").split(",") if t]:
        if "student_" + t in F:
            cands[f"student_{t}_64"] = F["student_" + t]
    artist = np.array([r["uploader"] for r in rows]); half = np.array([r["split"] for r in rows]); proc = np.array([process_family(r.get("process")) for r in rows])
    m = half == split
    out = {"dataset": DS, "split": split, "n": int(m.sum()), "n_artists": int(len(set(artist[m]))), "processes": dict(Counter(proc[m])), "candidates": {}}
    for name, Z in cands.items():
        E = unit(Z[m].astype(np.float32)); r = {}
        r["process_p5_artist_excluded"] = retrieval(E, proc[m], artist[m])
        S = E @ E.T; np.fill_diagonal(S, -np.inf); own = []
        for i in range(len(E)):
            order = np.argsort(-S[i])[:5]; own.append((artist[m][order] == artist[m][i]).mean())
        r["artist_share_top5"] = {"share": round(float(np.mean(own)), 4), "prior": round(float(np.mean([(artist[m] == a).mean() for a in artist[m]])), 4)}
        out["candidates"][name] = r
    teacher = cands["hybrid_supcon_family"][m]
    out["student_teacher_agreement"] = {n: agreement(cands[n][m], teacher) for n in cands if n.startswith("student_")}
    base = out["candidates"]["hybrid_supcon_family"]; rules = {}
    for name, r in out["candidates"].items():
        if name == "hybrid_supcon_family":
            continue
        p = r["process_p5_artist_excluded"]; leak = r["artist_share_top5"]["share"] - base["artist_share_top5"]["share"]
        rules[name] = {"artist_share_excess": round(float(leak), 4), "process_ci_low_above_chance": bool(p.get("ci95") and p["ci95"][0] > p.get("chance", 0)),
                       "falsified": bool(leak > 0.02 or not (p.get("ci95") and p["ci95"][0] > p.get("chance", 0)))}
    out["rules"] = rules
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / f"{DS}_{split}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("n", "n_artists", "processes")}))
    print(f"{'candidate':26s} {'proc p@5':>9s} {'CI':>16s} {'chance':>7s} {'artist share':>13s} {'prior':>6s} falsified")
    for name, r in out["candidates"].items():
        p, a = r["process_p5_artist_excluded"], r["artist_share_top5"]
        print(f"{name:26s} {p.get('precision@5') or 0:9.3f} {str(p.get('ci95')):>16s} {p.get('chance', 0) or 0:7.3f} {a['share']:13.3f} {a['prior']:6.3f} {rules.get(name, {}).get('falsified', '-')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--features", action="store_true"); ap.add_argument("--split", choices=["analysis", "test"], default=None)
    a = ap.parse_args(); EC.configure(DS)
    if "STUDENT_TAGS" not in os.environ:                                      # same eligible students as the evaluation pipeline
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        from research.eval_pipeline import eligible_students
        os.environ["STUDENT_TAGS"] = ",".join(eligible_students())
    if a.features:
        EC.do_features()
    if a.split:
        if a.split == "test":
            raise SystemExit("the test half of aic_photographs is not evaluated by this script (analysis only)")
        do_eval(a.split)
