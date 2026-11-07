#!/usr/bin/env python
"""E22: does the student reproduce its teacher outside the synthetic domain? (synthetic-to-real gap, hypothesis H-domain)

  python scripts/student_teacher_agreement.py   # -> reports/student_teacher_agreement.json

student_fs_v2 was distilled from the hybrid + family SupCon teacher on filmsynth v2 images only. If the student
fails on real scans because it never saw real inputs (input-domain shift of the distillation), its agreement with
the teacher should be high on filmsynth and low on real photographs. Agreement, per image set, from the benchmark
cache (no labels): top-10 neighbour Jaccard between teacher and student, Spearman correlation of the two pairwise
cosine-similarity matrices, and the same for a within-teacher control (teacher vs teacher on the nuisance copies,
filmsynth only). Sets: filmsynth val (treated), CommonCatalog originals (val+train, real digital + film),
BlueNeg blue-corrupted (real film; the analysis partition); BlueNeg blue-intact is NOT used (final-only). CommonCatalog
train originals are reported separately because E23b distils on them.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402


def agreement(A, B, k=10):
    A, B = unit(A), unit(B); SA, SB = A @ A.T, B @ B.T; np.fill_diagonal(SA, -np.inf); np.fill_diagonal(SB, -np.inf)
    na, nb = np.argsort(-SA, 1)[:, :k], np.argsort(-SB, 1)[:, :k]
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nb)])
    iu = np.triu_indices(len(A), 1)
    return {"neighbour_jaccard_top10": round(float(jac), 4), "similarity_spearman": round(float(spearmanr(SA[iu], SB[iu]).correlation), 4), "n": int(len(A))}


def main():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load_cache()
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    treated = np.array([bool(r.get("treated", False)) for r in rows]); part = np.array([r.get("partition", "") for r in rows])
    tr = (st == "filmsynth") & (sp == "train"); proj = X.supcon(F["hybrid"][tr], fam[tr], dev); T = proj(F["hybrid"])
    import os
    tag = os.environ.get("STUDENT_TAG")                      # a student embedded with scripts/student_bench_embed.py
    S = np.load(P.EMBEDDINGS / f"student_{tag}_bench.npz")["student64"] if tag else F["student64"]
    sets = {"filmsynth_val_treated": (st == "filmsynth") & (sp == "val") & treated, "filmsynth_test_treated": (st == "filmsynth") & (sp == "test") & treated,
            "sdxl_val": (st == "sdxl") & (sp == "val"), "zimage_val": (st == "zimage") & (sp == "val"),
            "cc_originals_real_train": (st == "cc_original") & (sp == "train"),        # E23b distils on these: in-sample for that student
            "cc_originals_real": (st == "cc_original") & np.isin(sp, ["val", "test"]),  # held-out real photographs
            "blueneg_corrupted_real_film": (st == "blueneg") & (part == "blue-corrupted")}
    out = {"teacher": "hybrid + family SupCon (filmsynth train)", "student": f"student_{tag} 64-d" if tag else "student_fs_v2 64-d", "sets": {}}
    for name, m in sets.items():
        out["sets"][name] = agreement(T[m], S[m])
    # control: teacher self-agreement under nuisance (how much agreement a faithful student could reach)
    fv = (st == "filmsynth") & (sp == "val") & treated; fn = (st == "filmsynth_nuis") & (sp == "val") & treated
    out["control_teacher_vs_teacher_nuisance_filmsynth_val"] = agreement(T[fv], T[fn])
    out["control_student_vs_student_nuisance_filmsynth_val"] = agreement(S[fv], S[fn])
    # does the student's real-scan geometry follow content instead? student vs DINOv2 global on real sets
    for name in ("cc_originals_real", "blueneg_corrupted_real_film"):
        m = sets[name]; out["sets"][name]["student_vs_dino_global"] = agreement(S[m], F["dino_global"][m]); out["sets"][name]["teacher_vs_dino_global"] = agreement(T[m], F["dino_global"][m])
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / (f"student_teacher_agreement_{tag}.json" if tag else "student_teacher_agreement.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
