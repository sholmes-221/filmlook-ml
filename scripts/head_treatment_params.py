#!/usr/bin/env python
"""E41: can a head raise rank with treatment information instead of scene content?

  python scripts/head_treatment_params.py     # -> reports/head_treatment_params.json (cached features; tiny heads on GPU if free)

E40: every head that gained on real photographs let scene content back in (own-source rank-1 0.7-1.0), while the
collapsed family projection (rank ~4, own-source 0.30) kept it out. The synthetic study also records six continuous
treatment parameters per image (temperature shift, saturation, contrast, grain strength, shadow lift, highlight
compression; temperature and saturation masked for monochrome treatments). Heads on the frozen hybrid features:
  params_ridge6        ridge regression hybrid -> 6 standardised parameters (fit on filmsynth train)
  params_mlp6          2-layer MLP regression -> 6 parameters
  supcon_params64      SupCon on families + parameter regression from the same 64-d output (multi-task), lambda_reg 1
  supcon_params_vic64  the same plus VICReg (lambda 1)
  proj_plus_params70   the family projection concatenated with the ridge parameter estimate (unit-scaled blocks)
Evaluated on the real-photograph matrix (scripts/student_real_matrix.py: museum, stock-line, within-uploader AUC,
photographer, scanner, synthetic family p@5, own-source rank-1, effective rank), analysis halves only.
Pre-registered rule (research/backlog.yaml E41), primary `supcon_params_vic64`: supported = 1 iff effective rank >= 8,
own-source rank-1 <= 0.35, museum p@5 >= the projection's, within-uploader different-day AUC >= the projection's - 0.02,
and synthetic family p@5 >= the projection's - 0.10. No test half, no BlueNeg.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import student_real_matrix as M  # noqa: E402
import synth_experiment as X  # noqa: E402
from style_benchmark import load_cache, unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

AIC = P.EMBEDDINGS / "aic_photographs_features_v1"
CCF = P.EMBEDDINGS / "commoncatalog_film_features_v1"
PRIMARY = "supcon_params_vic64"


def train_multitask(Xtr, ftr, Ytr, Mtr, dev, dim=64, epochs=400, tau=0.1, lam_reg=1.0, lam_vic=0.0, supcon=True, seed=0):
    """SupCon on family labels (optional) + masked regression of the standardised treatment parameters from a linear
    read-out of the same output (+ optional VICReg); returns the 64-d output map."""
    import torch
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    lab = torch.tensor(np.unique(ftr, return_inverse=True)[1], device=dev); Y = torch.tensor(Ytr, dtype=torch.float32, device=dev); Mk = torch.tensor(Mtr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, dim)).to(dev); head = torch.nn.Linear(dim, Y.shape[1]).to(dev)
    opt = torch.optim.AdamW([*net.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-3)
    pos = (lab[:, None] == lab[None]).float(); pos.fill_diagonal_(0); eye = torch.eye(len(Xt), device=dev, dtype=torch.bool)
    for _ in range(epochs):
        h = net(Xt + 0.05 * torch.randn_like(Xt)); z = torch.nn.functional.normalize(h, dim=1); loss = 0.0
        if supcon:
            logits = (z @ z.T / tau).masked_fill(eye, -1e9); logp = logits - torch.logsumexp(logits, 1, keepdim=True); loss = (-(pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()
        loss = loss + lam_reg * (((head(h) - Y) ** 2) * Mk).sum() / Mk.sum()
        if lam_vic:
            zc = h - h.mean(0); std = torch.sqrt(zc.var(0) + 1e-4); C = (zc.T @ zc) / (len(h) - 1)
            loss = loss + lam_vic * (torch.relu(1 - std).mean() + (C - torch.diag(torch.diag(C))).pow(2).sum() / h.shape[1])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor(sc.transform(Xn), dtype=torch.float32, device=dev)).detach().cpu().numpy()


def main():
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Fc, rows_c = load_cache(); st = np.array([r["set"] for r in rows_c]); sp = np.array([r["split"] for r in rows_c]); fam = np.array([r["family"] for r in rows_c]); treated = np.array([bool(r.get("treated")) for r in rows_c])
    tr = (st == "filmsynth") & (sp == "train"); q = (st == "filmsynth") & (sp == "val") & treated; bank = (st == "source") & (sp == "val")
    Yraw = np.array([[float((r.get("params") or {}).get(t, 0.0)) for t in X.TARGETS] for r in rows_c]); mono = np.array([bool((r.get("params") or {}).get("monochrome")) for r in rows_c])
    Mk = np.ones_like(Yraw); Mk[~treated | ~(st == "filmsynth")] = 0; Mk[mono, 0] = Mk[mono, 1] = 0
    ysc = StandardScaler().fit(Yraw[tr & treated]); Y = ysc.transform(Yraw)
    ra, rg, rv, rb, L = M.labels(); A = dict(np.load(AIC.with_suffix(".npz"))); G = dict(np.load(CCF.with_suffix(".npz")))
    sets = {"aic": np.hstack([A["explicit"], A["dino_style"]])[L["ia"]], "ccf": np.hstack([G["explicit"], G["dino_style"]])[L["ig"]], "val": Fc["hybrid"][q], "bank": Fc["hybrid"][bank]}
    xsc = StandardScaler().fit(Fc["hybrid"][tr]); Xs = lambda Z: xsc.transform(Z)  # noqa: E731
    proj = X.supcon(Fc["hybrid"][tr], fam[tr], dev)
    ridge = Ridge(alpha=100.0).fit(Xs(Fc["hybrid"][tr & treated]), Y[tr & treated])          # masked rows: monochrome keeps zeros for the two colour targets
    mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=600, random_state=0).fit(Xs(Fc["hybrid"][tr & treated]), Y[tr & treated])
    heads = {"supcon_family_proj": proj, "params_ridge6": lambda Z: ridge.predict(Xs(Z)), "params_mlp6": lambda Z: mlp.predict(Xs(Z)),
             "supcon_params64": train_multitask(Fc["hybrid"][tr], fam[tr], Y[tr], Mk[tr], dev, lam_reg=1.0, lam_vic=0.0),
             PRIMARY: train_multitask(Fc["hybrid"][tr], fam[tr], Y[tr], Mk[tr], dev, lam_reg=1.0, lam_vic=1.0),
             "params_only_vic64": train_multitask(Fc["hybrid"][tr], fam[tr], Y[tr], Mk[tr], dev, lam_reg=1.0, lam_vic=1.0, supcon=False),
             "proj_plus_params70": lambda Z: np.hstack([unit(proj(Z)), unit(ridge.predict(Xs(Z)))])}
    out = {"experiment": "E41_head_treatment_params", "analysis_data_only": True, "targets": X.TARGETS, "heads": {}}
    for name, h in heads.items():
        r = M.matrix(h(sets["aic"]), h(sets["ccf"]), h(sets["val"]), L, h(sets["bank"])); out["heads"][name] = r; print(M.fmt(name, r), flush=True)
    b, c = out["heads"]["supcon_family_proj"], out["heads"][PRIMARY]
    checks = {"eff_rank_ge_8": c["eff_rank"] >= 8, "own_source_le_0.35": c["own_source_rank1"] <= 0.35, "museum_not_worse": c["museum_p5"]["precision@5"] >= b["museum_p5"]["precision@5"],
              "within_auc_ok": c["within_uploader_auc"]["auc"] >= b["within_uploader_auc"]["auc"] - 0.02, "synth_cost_ok": c["synth_family_p5"]["precision@5"] >= b["synth_family_p5"]["precision@5"] - 0.10}
    out["summary"] = {"supported": int(all(checks.values())), **{k: bool(v) for k, v in checks.items()}, "primary": PRIMARY, "primary_eff_rank": c["eff_rank"], "primary_own_source_rank1": c["own_source_rank1"],
                      "primary_museum_p5": c["museum_p5"]["precision@5"], "baseline_museum_p5": b["museum_p5"]["precision@5"], "primary_within_auc": c["within_uploader_auc"]["auc"], "baseline_within_auc": b["within_uploader_auc"]["auc"],
                      "primary_synth_family_p5": c["synth_family_p5"]["precision@5"], "baseline_synth_family_p5": b["synth_family_p5"]["precision@5"], "primary_stock_line_p5": c["stock_line_p5"]["precision@5"],
                      "ridge_own_source_rank1": out["heads"]["params_ridge6"]["own_source_rank1"], "ridge_within_auc": out["heads"]["params_ridge6"]["within_uploader_auc"]["auc"], "seconds": round(time.time() - t0, 1)}
    P.REPORTS.mkdir(exist_ok=True); (P.REPORTS / "head_treatment_params.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
