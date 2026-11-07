#!/usr/bin/env python
"""E26: which DINOv2-Small layers and which patch statistics carry transferable film-treatment signal?

  python scripts/layer_stat_ablation.py                 # -> reports/layer_stat_ablation.json (+ cached per-layer features)

The hybrid representation uses mean++std of patch tokens from layers 4 and 8, an untested guess (backlog E9). Classic
style-transfer evidence says style lives in second-order statistics (Gram matrices) of early layers while content lives
in late-layer activations. Hypothesis: a (layer, statistic) choice exists that keeps the synthetic treatment
sensitivity of L4+L8 mean/std, retrieves real same-stock scans at least as well, and leaks less content.

One GPU pass over the benchmark image list (data/embeddings/bench_features_v1.json, identical preprocessing) caches,
per layer 1..12: mean++std of the 256 patch tokens (768-d) and the upper triangle of the covariance of the tokens
after a fixed random projection to 32 dims (528-d, "gram32"). Candidates = explicit28 ++ feature -> family SupCon
projection trained on filmsynth TRAIN only (as benchmark v1). Evaluation uses analysis data only: benchmark val split
(tests A, B, C, E via scripts/style_benchmark.evaluate) and BlueNeg blue-corrupted same-stock p@5 with the query's
roll excluded and a roll bootstrap CI. The blue-intact partition and the test split are not touched.

Pre-registered rule (research/backlog.yaml E26): a candidate "beats" the L4+L8 mean/std baseline when
  (i)  its BlueNeg-corrupted same-stock p@5 lower CI bound exceeds the baseline's point estimate,
  (ii) its val B filmsynth p@5 is not below the baseline's lower CI bound, and
  (iii) its own-source rank-1 (test E) does not exceed the baseline's by more than 0.05.
`summary.beats` = 1 if any candidate satisfies all three, else 0 (falsified).
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import build_bench_features as BB  # noqa: E402
import synth_experiment as X  # noqa: E402
from style_benchmark import evaluate, load_cache  # noqa: E402
from synth_experiment_v3 import same_stock  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

CACHE = P.EMBEDDINGS / "bench_layers_v1.npz"
PROJ_DIM = 32


def extract(rows, dev):
    import torch
    from transformers import AutoModel
    from PIL import Image
    m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    g = torch.Generator().manual_seed(0); W = (torch.randn(384, PROJ_DIM, generator=g) / np.sqrt(384)).to(dev)
    with Pool(16) as pool:
        arrs = pool.map(BB._load, [(r["path"], r["id"], r["real"], r["nuis"]) for r in rows], chunksize=8)
    ms = {l: [] for l in range(1, 13)}; gr = {l: [] for l in range(1, 13)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        for i in range(0, len(arrs), 64):
            px = torch.from_numpy(np.stack([X.dinov2_pixels(Image.fromarray(a), 224) for a in arrs[i:i + 64]])).to(dev)
            hs = m(pixel_values=px, output_hidden_states=True).hidden_states
            for l in range(1, 13):
                t = hs[l][:, 1:].float()                                            # patch tokens (B, 256, 384)
                ms[l].append(torch.cat([t.mean(1), t.std(1)], 1).cpu().numpy())
                q = t @ W; q = q - q.mean(1, keepdim=True); c = q.transpose(1, 2) @ q / q.shape[1]      # (B, 32, 32) covariance
                iu = torch.triu_indices(PROJ_DIM, PROJ_DIM)
                cu = c[:, iu[0], iu[1]]
                gr[l].append((torch.sign(cu) * torch.sqrt(cu.abs())).cpu().numpy())      # signed sqrt: tames heavy tails
    out = {f"ms_L{l}": np.concatenate(ms[l]).astype(np.float16) for l in ms}
    out.update({f"gram_L{l}": np.concatenate(gr[l]).astype(np.float16) for l in gr})
    np.savez(CACHE, ids=np.array([r["id"] for r in rows]), **out)
    return out


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F, rows = load_cache()
    if CACHE.exists() and (np.load(CACHE)["ids"] == np.array([r["id"] for r in rows])).all():
        z = np.load(CACHE); L = {k: z[k] for k in z.files if k != "ids"}; seconds_extract = 0.0
    else:
        L = extract(rows, dev); seconds_extract = round(time.time() - t0, 1)
    st = np.array([r["set"] for r in rows]); sp = np.array([r["split"] for r in rows]); fam = np.array([r["family"] for r in rows])
    part = np.array([r.get("partition", "") for r in rows]); roll = np.array([r.get("roll", "") for r in rows])
    tr = (st == "filmsynth") & (sp == "train"); bc = (st == "blueneg") & (part == "blue-corrupted")
    feats = {"L4L8_ms (baseline)": F["dino_style"]}
    for l in range(1, 13):
        feats[f"L{l}_ms"] = L[f"ms_L{l}"].astype(np.float32); feats[f"L{l}_gram32"] = L[f"gram_L{l}"].astype(np.float32)
    feats["L1-4_gram32"] = np.hstack([L[f"gram_L{l}"] for l in (1, 2, 3, 4)]).astype(np.float32)
    feats["L4L8_gram32"] = np.hstack([L["gram_L4"], L["gram_L8"]]).astype(np.float32)
    feats["L4L8_ms+gram32"] = np.hstack([F["dino_style"], L["gram_L4"], L["gram_L8"]]).astype(np.float32)
    res = {}
    for name, Z in feats.items():
        H = np.hstack([F["explicit"], Z]); proj = X.supcon(H[tr], fam[tr], dev); E = proj(H)
        r = evaluate(E, F, rows, "val")
        res[name] = {"dim_in": int(H.shape[1]), "val_B_filmsynth": r["B_same_treatment_across_scenes"]["filmsynth"], "val_A_p": r["A_treatment_gt_nuisance"]["p"],
                     "val_C_share": r["C_generator_fingerprint"]["same_generator_share_top5"], "val_E_rank1": r["E_semantic_leakage"]["own_source_rank1"],
                     "val_E_subject_probe": r["E_semantic_leakage"]["subject_probe_accuracy"],
                     "blueneg_corrupted_same_stock": same_stock(E[bc], fam[bc], roll[bc])}
        print(f"{name:22s} B fs {res[name]['val_B_filmsynth']['precision@5']:.3f} {res[name]['val_B_filmsynth']['ci95']}  E r1 {res[name]['val_E_rank1']:.3f}  subj {res[name]['val_E_subject_probe']:.3f}  "
              f"BN-corr p@5 {res[name]['blueneg_corrupted_same_stock']['precision@5']:.3f} {res[name]['blueneg_corrupted_same_stock']['ci95']}", flush=True)
    base = res["L4L8_ms (baseline)"]; beats = {}
    for name, r in res.items():
        if name == "L4L8_ms (baseline)":
            continue
        c1 = r["blueneg_corrupted_same_stock"]["ci95"][0] > base["blueneg_corrupted_same_stock"]["precision@5"]
        c2 = r["val_B_filmsynth"]["precision@5"] >= base["val_B_filmsynth"]["ci95"][0]
        c3 = r["val_E_rank1"] <= base["val_E_rank1"] + 0.05
        beats[name] = {"real_ci_above_baseline": bool(c1), "synthetic_not_below": bool(c2), "content_not_worse": bool(c3), "beats": bool(c1 and c2 and c3)}
    winners = [n for n, b in beats.items() if b["beats"]]
    best_real = max(res, key=lambda n: res[n]["blueneg_corrupted_same_stock"]["precision@5"])
    summary = {"beats": int(bool(winners)), "winners": winners, "best_real_candidate": best_real, "baseline": base, "best_real": res[best_real],
               "n_blueneg_corrupted": int(bc.sum()), "seconds_extract": seconds_extract, "seconds": round(time.time() - t0, 1)}
    rep = {"experiment": "E26_layer_statistic_ablation", "analysis_data_only": True, "results": res, "rules": beats, "summary": summary}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "layer_stat_ablation.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("beats", "winners", "best_real_candidate", "seconds")}))


if __name__ == "__main__":
    main()
