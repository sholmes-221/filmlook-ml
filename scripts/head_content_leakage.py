#!/usr/bin/env python
"""E40: do the heads that gain on real photographs (E33-E36, E38) do so by re-encoding scene content?

  python scripts/head_content_leakage.py     # -> reports/head_content_leakage.json (cached features, CPU)

The trainer's content-invariance test (own-source rank-1: a treated filmsynth validation image's nearest neighbour among
the 15 untreated validation sources is its own source; chance 1/15) read 0.16-0.18 for the baseline students, 0.25-0.29
for the VICReg-target students and 0.76 for the label-free-whitened-target student (E38/E39 reports). This run applies
the same test to the frozen-feature heads of E33/E34 and to the independent whitening of E36 on the benchmark cache,
next to each head's commoncatalog_film stock-line p@5 (uploader excluded) and museum process p@5 (stereo removed,
decade matched), so that every "real-photograph gain" is read together with its content leakage.
Pre-registered rule (research/backlog.yaml E40): summary.supported = 1 iff every head whose stock-line p@5 exceeds the
SupCon projection's by >= 0.02 has own-source rank-1 >= 0.5, and the projection's own-source rank-1 is <= 0.25.
Analysis halves and the filmsynth validation split only; no test split, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
from colour_vs_texture import IDX_C, IDX_T  # noqa: E402
from e29_audit import retrieval  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from frozen_head_choice import PCAW  # noqa: E402
from head_anticollapse import train_head  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"


def own_source_rank1(Eq, Eb, src_q, src_b):
    S = unit(Eq) @ unit(Eb).T
    return round(float(np.mean([src_b[S[i].argmax()] == s for i, s in enumerate(src_q)])), 4)


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    src = np.array([str(r.get("source")) for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); q = (st == "filmsynth") & (sp == "val") & treated; bank = (st == "source") & (sp == "val"); cc_tr = (st == "cc_original") & (sp == "train")
    A = dict(np.load(AIC.with_suffix(".npz"))); rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]; Fit = dict(np.load(FIT.with_suffix(".npz")))
    ma = np.array([r["split"] == "analysis" for r in rows_a]); mg = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows_g])
    artist = np.array([r["uploader"] for r in rows_a])[ma]; proc = np.array([process_family(r.get("process")) for r in rows_a])[ma]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows_a])[ma]; decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows_a])[ma]
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    up = np.array([r["uploader"] for r in rows_g])[mg]; line = np.array([r.get("film_stock_line") or r.get("film_stock_claim") or "?" for r in rows_g])[mg]
    sets = {"ccf": (G["dino_style"][mg], G["explicit"][mg], G["student64"][mg]), "aic": (A["dino_style"][ma], A["explicit"][ma], A["student64"][ma]),
            "q": (Fc["dino_style"][q], Fc["explicit"][q], Fc["student64"][q]), "bank": (Fc["dino_style"][bank], Fc["explicit"][bank], Fc["student64"][bank])}
    fit_cc = Fc["dino_style"][cc_tr]; proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    heads = {"supcon_family_proj": lambda d, e, s: proj(np.hstack([e, d])), "L4L8_raw": lambda d, e, s: d}
    heads["L4L8_std_cc"] = (lambda mu, sd: (lambda d, e, s: (d - mu) / sd))(fit_cc.mean(0), fit_cc.std(0) + 1e-6)
    for k in (16, 64):
        heads[f"L4L8_pcaw_cc_{k}"] = (lambda p: (lambda d, e, s: p(d)))(PCAW(k).fit(fit_cc))
    hyb_fit = np.hstack([Fc["explicit"][cc_tr], fit_cc]); heads["hybrid_pcaw_cc_64"] = (lambda p: (lambda d, e, s: p(np.hstack([e, d]))))(PCAW(64).fit(hyb_fit))
    pw = PCAW(64).fit(proj(hyb_fit)); heads["supcon_proj_pcaw_cc"] = lambda d, e, s: pw(proj(np.hstack([e, d])))
    for lam in (1.0, 3.0):
        heads[f"supcon_t0.1_vic{lam}"] = (lambda h: (lambda d, e, s: h(np.hstack([e, d]))))(train_head(Fc["hybrid"][tr], fam[tr], dev, tau=0.1, lam_vic=lam, seed=0))
    ex_fit = Fc["explicit"][cc_tr]
    heads["explicit28_std_cc"] = (lambda mu, sd: (lambda d, e, s: (e - mu) / sd))(ex_fit.mean(0), ex_fit.std(0) + 1e-6)
    heads["colour16_std_cc"] = (lambda mu, sd: (lambda d, e, s: (e[:, IDX_C] - mu) / sd))(ex_fit[:, IDX_C].mean(0), ex_fit[:, IDX_C].std(0) + 1e-6)
    heads["texture12_std_cc"] = (lambda mu, sd: (lambda d, e, s: (e[:, IDX_T] - mu) / sd))(ex_fit[:, IDX_T].mean(0), ex_fit[:, IDX_T].std(0) + 1e-6)
    heads["student_fs_v2_raw"] = lambda d, e, s: s
    heads["student_fs_v2_whiten_indep"] = (lambda p: (lambda d, e, s: p(s)))(PCAW(64).fit(Fit["student_style_fs_v2"]))
    out = {"experiment": "E40_head_content_leakage", "analysis_data_only": True, "n_queries": int(q.sum()), "n_bank_sources": int(bank.sum()), "chance_rank1": round(1 / int(bank.sum()), 4), "heads": {}}
    print(f"{'head':24s} {'own-source rank-1':>17s} {'stock-line p@5':>14s} {'museum p@5':>10s}", flush=True)
    for name, h in heads.items():
        f = lambda k: np.nan_to_num(h(*sets[k]).astype(np.float32))  # noqa: E731
        r = {"own_source_rank1": own_source_rank1(f("q"), f("bank"), src[q], src[bank]), "stock_line_p5": retrieval(unit(f("ccf")), line, up), "museum_p5": retrieval(unit(f("aic")), proc, artist, allowed)}
        out["heads"][name] = r; print(f"{name:24s} {r['own_source_rank1']:17.3f} {r['stock_line_p5']['precision@5']:14.3f} {r['museum_p5']['precision@5']:10.3f}", flush=True)
    b = out["heads"]["supcon_family_proj"]; gain = {n: r for n, r in out["heads"].items() if n != "supcon_family_proj" and r["stock_line_p5"]["precision@5"] - b["stock_line_p5"]["precision@5"] >= 0.02}
    ok = all(r["own_source_rank1"] >= 0.5 for r in gain.values()) and b["own_source_rank1"] <= 0.25
    out["summary"] = {"supported": int(ok), "projection_rank1": b["own_source_rank1"], "gaining_heads": {n: r["own_source_rank1"] for n, r in gain.items()}, "n_gaining": len(gain),
                      "n_gaining_content_dominated": int(sum(r["own_source_rank1"] >= 0.5 for r in gain.values())), "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "head_content_leakage.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
