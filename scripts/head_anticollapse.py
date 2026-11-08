#!/usr/bin/env python
"""E34: is the SupCon projection's rank collapse what loses real-photograph information, and is its low photographer
leakage a by-product of that collapse?

  python scripts/head_anticollapse.py          # -> reports/head_anticollapse.json (cached features; tiny heads on GPU if free)

Heads are trained on the same frozen hybrid features (explicit 28 + DINOv2-Small L4+L8 statistics 1536) and the same
synthetic family labels (filmsynth train sources) as the current projection, with two collapse controls:
  tau       : the SupCon temperature (0.1 = current; 0.5 = softer);
  vicreg    : VICReg variance + covariance regularisation (weight lambda) on the pre-normalisation output, which forbids
              the 64 dimensions from collapsing onto a few family directions.
Evaluation (analysis halves only, group bootstrap CIs, uploader/artist excluded): commoncatalog_film stock-line p@5,
uploader share, scanner-category probe (leakage guard); aic_photographs process p@5 with stereo cards removed and the
decade matched, artist share; synthetic family p@5 on the filmsynth validation split (cost); effective rank.
Pre-registered rule (research/backlog.yaml E34): primary candidate `supcon_t0.1_vic1.0` (seed 0): supported = 1 iff its
effective rank on commoncatalog_film >= 16, its museum p@5 lower CI is above chance, its stock-line p@5 lower CI is above
chance, and its uploader-share, artist-share (<= +0.02) and scanner-probe (<= +0.05) excesses over the current projection
stay within the guards. Seeds 1 and 2 of the primary candidate are reported for stability. No test half, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from e29_audit import eff_rank, retrieval  # noqa: E402
from eval_commons_film import grouped_probe  # noqa: E402
from eval_curated_photos import process_family  # noqa: E402
from frozen_head_choice import share  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
PRIMARY = "supcon_t0.1_vic1.0"
CONFIGS = [("supcon_t0.1", 0.1, 0.0), ("supcon_t0.5", 0.5, 0.0), ("supcon_t0.1_vic0.3", 0.1, 0.3), (PRIMARY, 0.1, 1.0), ("supcon_t0.1_vic3.0", 0.1, 3.0), ("vicreg_only", None, 1.0)]


def train_head(Xtr, ftr, dev, dim=64, epochs=400, tau=0.1, lam_vic=0.0, seed=0):
    """SupCon on synthetic family labels with optional VICReg variance/covariance regularisation (tau=None: VICReg only,
    positives = the two noisy views of each row)."""
    import torch
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); X = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    lab = torch.tensor(np.unique(ftr, return_inverse=True)[1], device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dim)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = (lab[:, None] == lab[None]).float(); pos.fill_diagonal_(0); eye = torch.eye(len(X), device=dev, dtype=torch.bool)

    def vic(z):
        zc = z - z.mean(0); std = torch.sqrt(zc.var(0) + 1e-4); v = torch.relu(1 - std).mean()
        C = (zc.T @ zc) / (len(z) - 1); c = (C - torch.diag(torch.diag(C))).pow(2).sum() / z.shape[1]
        return v + c
    for _ in range(epochs):
        h = net(X + 0.05 * torch.randn_like(X)); z = torch.nn.functional.normalize(h, dim=1); loss = 0.0
        if tau is not None:
            logits = (z @ z.T / tau).masked_fill(eye, -1e9); logp = logits - torch.logsumexp(logits, 1, keepdim=True)
            loss = (-(pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()
        else:
            h2 = net(X + 0.05 * torch.randn_like(X)); loss = ((h - h2) ** 2).mean() + vic(h2)
        if lam_vic:
            loss = loss + lam_vic * vic(h)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor(sc.transform(Xn), dtype=torch.float32, device=dev)).detach().cpu().numpy()


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c])
    src = np.array([str(r.get("source")) for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); fs_val = (st == "filmsynth") & (sp == "val") & treated
    A = dict(np.load(AIC.with_suffix(".npz"))); rows_a = json.loads(AIC.with_suffix(".json").read_text())["rows"]
    G = dict(np.load(CCF.with_suffix(".npz"))); rows_g = json.loads(CCF.with_suffix(".json").read_text())["rows"]
    ma = np.array([r["split"] == "analysis" for r in rows_a]); mg = np.array([r["split"] == "analysis" and bool(r.get("film_stock")) for r in rows_g])
    artist = np.array([r["uploader"] for r in rows_a])[ma]; proc = np.array([process_family(r.get("process")) for r in rows_a])[ma]
    stereo = np.array(["stereo" in (r.get("process") or "").lower() for r in rows_a])[ma]
    decade = np.array([str((r.get("date_start") // 10) * 10) if r.get("date_start") else "nd" for r in rows_a])[ma]
    allowed = (decade[:, None] == decade[None]) & ~stereo[None] & ~stereo[:, None]
    up = np.array([r["uploader"] for r in rows_g])[mg]; line = np.array([r.get("film_stock_line") or r.get("film_stock_claim") or "?" for r in rows_g])[mg]
    scanner = np.array([r.get("scanner_category") or "" for r in rows_g])[mg]; has = scanner != ""
    hyb_c = np.hstack([G["explicit"], G["dino_style"]])[mg]; hyb_a = np.hstack([A["explicit"], A["dino_style"]])[ma]; hyb_v = Fc["hybrid"][fs_val]
    out = {"experiment": "E34_head_anticollapse", "analysis_data_only": True, "n_ccf": int(mg.sum()), "n_aic": int(ma.sum()), "n_fs_val": int(fs_val.sum()), "heads": {}}

    def evaluate(name, f):
        Ec, Ea, Ev = (unit(np.nan_to_num(f(Z).astype(np.float32))) for Z in (hyb_c, hyb_a, hyb_v))
        r = {"stock_line_p5": retrieval(Ec, line, up), "uploader_share": share(Ec, up), "scanner_probe": grouped_probe(Ec[has], scanner[has], up[has]),
             "museum_p5": retrieval(Ea, proc, artist, allowed), "artist_share": share(Ea, artist), "synth_family_p5": retrieval(Ev, fam[fs_val], src[fs_val]), "eff_rank_ccf": eff_rank(Ec)}
        out["heads"][name] = r; s, m_, v = r["stock_line_p5"], r["museum_p5"], r["synth_family_p5"]
        print(f"{name:24s} stock {s['precision@5']:.3f} {str(s['ci95']):>15s} ch {s['chance']:.3f} up {r['uploader_share']:.3f} scan {r['scanner_probe']['accuracy']:.3f} | museum {m_['precision@5']:.3f} {str(m_['ci95']):>15s} ch {m_['chance']:.3f} art {r['artist_share']:.3f} | synth {v['precision@5']:.3f} | erank {r['eff_rank_ccf']}", flush=True)
        return r
    for name, tau, lam in CONFIGS:
        evaluate(name, train_head(Fc["hybrid"][tr], fam[tr], dev, tau=tau, lam_vic=lam, seed=0))
    for seed in (1, 2):
        evaluate(f"{PRIMARY}_seed{seed}", train_head(Fc["hybrid"][tr], fam[tr], dev, tau=0.1, lam_vic=1.0, seed=seed))
    b, c = out["heads"]["supcon_t0.1"], out["heads"][PRIMARY]
    checks = {"eff_rank_ok": c["eff_rank_ccf"] >= 16, "museum_ci_low_above_chance": c["museum_p5"]["ci95"][0] > c["museum_p5"]["chance"], "stock_line_ci_low_above_chance": c["stock_line_p5"]["ci95"][0] > c["stock_line_p5"]["chance"],
              "uploader_leak_ok": c["uploader_share"] - b["uploader_share"] <= 0.02, "artist_leak_ok": c["artist_share"] - b["artist_share"] <= 0.02,
              "scanner_leak_ok": (c["scanner_probe"]["accuracy"] or 0) - (b["scanner_probe"]["accuracy"] or 0) <= 0.05}
    out["summary"] = {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY, "primary_eff_rank": c["eff_rank_ccf"], "baseline_eff_rank": b["eff_rank_ccf"],
                      "primary_stock_line_p5": c["stock_line_p5"]["precision@5"], "baseline_stock_line_p5": b["stock_line_p5"]["precision@5"], "primary_museum_p5": c["museum_p5"]["precision@5"], "baseline_museum_p5": b["museum_p5"]["precision@5"],
                      "primary_uploader_share": c["uploader_share"], "baseline_uploader_share": b["uploader_share"], "primary_synth_family_p5": c["synth_family_p5"]["precision@5"], "baseline_synth_family_p5": b["synth_family_p5"]["precision@5"],
                      "seed_stock_line_p5": [out["heads"][f"{PRIMARY}_seed{s}"]["stock_line_p5"]["precision@5"] for s in (1, 2)], "seed_museum_p5": [out["heads"][f"{PRIMARY}_seed{s}"]["museum_p5"]["precision@5"] for s in (1, 2)],
                      "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "head_anticollapse.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
