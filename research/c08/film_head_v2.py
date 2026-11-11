"""C08: film-likeness head v2 = the deployed head's architecture fed with DINOv2-small blocks 4, 8 and 12 instead of
block 12 alone (the encoder comparison found the earlier blocks carry the rendering signal; research/c08/encoders.py).

    python -m research.c08.film_head_v2            # -> reports/c08/film_head_v2.json, dist/film_head/film_head_v2layers.npz

Same images, same uploader-disjoint split, same 36 statistics as the deployed head v1 (38a3c47d). No Claude-derived
auxiliary targets: the medium label (film scan or digital) is the only target. Embeddings are computed from the full
image with the production geometry (short side 256 bicubic, centre crop 224) in fp32, which matches the production ONNX.
Promotion rule: paired AUROC gain over the deployed v1 with a 95 % interval above zero, a content-stratified gain,
calibration (ECE) no worse than 0.03, and no loss under JPEG q50 or half-size resampling.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.c08.encoders import _content_only
from research.c08.pairs import DinoLayers, dino_crop
from research.lut.engine.film_likeness import film_features

SET, REPORTS, DIST, CACHE = Path("data/lut/film_likeness"), Path("reports/c08"), Path("dist/film_head"), Path("data/c08/encoders")
V1 = DIST / "film_head_v1.json"


def rows():
    return [json.loads(l) for l in open(SET / "manifest.jsonl")]


def full(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


class Head(torch.nn.Module):
    def __init__(self, d, hidden=128, p=0.3):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(p))
        self.medium = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return self.medium(self.body(x))[:, 0]


def train(Z, y, seeds=3, epochs=60, hidden=128):
    dev = "cuda"; Xt, yt = torch.tensor(Z, device=dev), torch.tensor(y, dtype=torch.float32, device=dev)
    w = torch.tensor(float((y == 0).sum() / max(1, (y == 1).sum())), device=dev)
    models = []
    for seed in range(seeds):
        torch.manual_seed(seed); m = Head(Z.shape[1], hidden).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            m.train(); perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[b]), yt[b], pos_weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
        models.append(m.eval())
    return models


def logits(models, Z):
    with torch.no_grad():
        Zt = torch.tensor(Z, device="cuda")
        return np.mean([m(Zt).cpu().numpy() for m in models], 0)


def v1_logits(F, X):
    art = json.loads(V1.read_text())
    mu, sd = np.array(art["mean36"], np.float32), np.array(art["std36"], np.float32)
    Z = np.hstack([(F - mu) / sd, X])
    outs = []
    for s in art["seeds"]:
        h = Z
        for W, b in s["layers"]:
            a = h @ np.array(W, np.float32) + np.array(b, np.float32)
            h = 0.5 * a * (1 + np.vectorize(__import__("math").erf)(a / 2 ** 0.5))
        outs.append(h @ np.array(s["medium_w"], np.float32) + s["medium_b"])
    return np.mean(outs, 0)


def parts_normalised(L: np.ndarray) -> np.ndarray:
    """[L4 CLS, L4 mean, L8 CLS, L8 mean, L12 CLS, L12 mean], each 384 values, each scaled to unit length (the
    production ONNX returns L12 CLS and mean that way)."""
    return np.concatenate([L[:, k * 384:(k + 1) * 384] / (np.linalg.norm(L[:, k * 384:(k + 1) * 384], axis=1, keepdims=True) + 1e-12) for k in range(6)], 1)


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p > lo) & (p <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main() -> None:
    rs = rows()
    y = np.array([r["label"] for r in rs]); g = np.array([r["uploader"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs]); te = np.where(~tr)[0]
    # Inputs exactly as production computes them (research/c08/prod_inputs.py: the website's own prepare_image,
    # engine decoder, descriptor and film_features; the photo gate's pixels), each CLS / mean part L2-normalised like
    # the production ONNX output.
    d = np.load(CACHE / "prod_inputs.npz"); assert d["paths"].tolist() == [r["path"] for r in rs]
    F = np.nan_to_num(d["F36"].astype(np.float32)); L = parts_normalised(d["layers"].astype(np.float32))
    dino = DinoLayers()
    X12 = L[:, 1536:2304]
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    lmu, lsd = L[tr].mean(0), L[tr].std(0) + 1e-6
    Z = np.hstack([(F - mu) / sd, (L - lmu) / lsd]).astype(np.float32)
    # Five-fold cross-fitting over training uploaders: each fold model never sees its fold, the out-of-fold logits of
    # all 1,104 training rows fit the Platt scaling, and the exported model is the ensemble of the five fold models
    # (one seed each), so calibration and the shipped model are consistent.
    utr = np.unique(g[tr]); rng = np.random.default_rng(0); perm = rng.permutation(utr); folds = np.array_split(perm, 5)
    oof = np.zeros(len(rs), np.float32); models = []
    for k, fu in enumerate(folds):
        hold = np.where(tr & np.isin(g, fu))[0]; fit_i = np.where(tr & ~np.isin(g, fu))[0]
        mk = train(Z[fit_i], y[fit_i], seeds=1)
        torch.manual_seed(k)
        oof[hold] = logits(mk, Z[hold]); models += mk
    tri = np.where(tr)[0]; fit_i, cal_i = tri, tri
    platt = LogisticRegression(C=1e6).fit(oof[tri][:, None], y[tri]); pa, pc = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    l2 = logits(models, Z[te])
    # each fold model is calibrated like the single models whose out-of-fold logits fitted Platt; probabilities are averaged
    p2_platt = 1 / (1 + np.exp(-(pa * l2 + pc)))
    # Isotonic calibration on the same out-of-fold logits, stored as a 2-column lookup (logit -> probability) and
    # applied by linear interpolation; it can correct the mid-range shape a sigmoid cannot.
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(oof[tri], y[tri])
    iso_x, iso_y = iso.X_thresholds_.astype(np.float32), iso.y_thresholds_.astype(np.float32)
    p2 = np.interp(l2, iso_x, iso_y)
    l1 = v1_logits(F[te], X12[te])  # the deployed head on the same production inputs
    art1 = json.loads(V1.read_text()); p1 = 1 / (1 + np.exp(-(art1["platt"]["a"] * l1 + art1["platt"]["c"])))
    cos = np.ones(1)
    yt, gt = y[te], g[te]
    ug = np.unique(gt); members = {u: np.where(gt == u)[0] for u in ug}
    boots = [np.concatenate([members[u] for u in rng.choice(ug, len(ug))]) for _ in range(1000)]
    a1, a2 = roc_auc_score(yt, l1), roc_auc_score(yt, l2)
    d = [roc_auc_score(yt[b], l2[b]) - roc_auc_score(yt[b], l1[b]) for b in boots]
    v2ci = [roc_auc_score(yt[b], l2[b]) for b in boots]
    rep = {"n_test": int(len(te)), "test_uploaders": int(len(ug)), "v1_ece_on_production_inputs": ece(p1, yt), "fit_rows": int(len(fit_i)), "calibration_rows": int(len(cal_i)),
           "v1_deployed_auroc": a1, "v2_auroc": a2, "v2_ci": [float(np.percentile(v2ci, 2.5)), float(np.percentile(v2ci, 97.5))],
           "paired_gain": [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "v2_ece": ece(p2, yt), "v2_ece_platt": ece(p2_platt, yt), "platt": {"a": pa, "c": pc}, "isotonic_points": int(len(iso_x))}
    rep["reliability"] = {}
    for name, pp in (("v1", p1), ("v2", p2)):
        tab = []
        for lo in np.linspace(0, 0.9, 10):
            m = (pp > lo) & (pp <= lo + 0.1)
            if m.any():
                tab.append([round(float(lo), 1), int(m.sum()), round(float(pp[m].mean()), 3), round(float(yt[m].mean()), 3)])
        eb = [ece(pp[b], yt[b]) for b in boots[:500]]
        rep["reliability"][name] = {"bins[lo,n,mean_p,observed]": tab, "ece_ci": [float(np.percentile(eb, 2.5)), float(np.percentile(eb, 97.5))]}
    colour = np.array([bool(r.get("colour")) for r in rs])[te]
    rep["colour_rows"] = {"v1": roc_auc_score(yt[colour], l1[colour]), "v2": roc_auc_score(yt[colour], l2[colour])}
    # content-stratified: AUROC within quintiles of a content-only model's score (48 px grey thumbnails)
    Lc = np.concatenate([dino(np.stack([dino_crop(np.asarray(_content_only(Image.fromarray(full(r["path"]))))) for r in rs[i:i + 128]])) for i in range(0, len(rs), 128)]).astype(np.float32)
    sc = StandardScaler().fit(Lc[tr]); cm = LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000).fit(sc.transform(Lc[tr]), y[tr])
    cs = cm.decision_function(sc.transform(Lc[te]))
    q = np.quantile(cs, [0.2, 0.4, 0.6, 0.8]); bins = np.digitize(cs, q)
    strat = {"content_only_auroc": roc_auc_score(yt, cs)}
    for name, s in (("v1", l1), ("v2", l2)):
        vals = [roc_auc_score(yt[bins == k], s[bins == k]) for k in range(5) if len(set(yt[bins == k])) == 2]
        strat[name] = float(np.mean(vals))
    rep["content_stratified_auroc"] = strat
    # robustness: recompute statistics and embeddings from perturbed pixels for 400 test rows
    sub = rng.choice(len(te), 400, replace=False)
    from research.c08.prod_inputs import compute
    tmp = Path("data/c08/encoders/_perturbed"); tmp.mkdir(parents=True, exist_ok=True)

    def perturbed(kind):
        paths = []
        for k in sub:
            im = Image.fromarray(full(rs[te[k]]["path"]))
            if kind == "half":
                im = im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.LANCZOS)
            out = tmp / f"{kind}_{k}.jpg"; im.save(out, "JPEG", quality=50 if kind == "jpeg50" else 95); paths.append(str(out))
        Fp, Lp = compute(paths); Fp = np.nan_to_num(Fp); Lp = parts_normalised(Lp)
        return (roc_auc_score(yt[sub], v1_logits(Fp, Lp[:, 1536:2304])), roc_auc_score(yt[sub], logits(models, np.hstack([(Fp - mu) / sd, (Lp - lmu) / lsd]).astype(np.float32))))
    rep["robustness_400"] = {"clean": {"v1": roc_auc_score(yt[sub], l1[sub]), "v2": roc_auc_score(yt[sub], l2[sub])}}
    for kind in ("jpeg50", "half"):
        a, b = perturbed(kind); rep["robustness_400"][kind] = {"v1": a, "v2": b}
    # export (float16 arrays: 2,340 inputs x 128 hidden x 3 seeds)
    DIST.mkdir(parents=True, exist_ok=True)
    arrays = {"mean36": mu, "std36": sd, "mean_emb": lmu, "std_emb": lsd, "platt": np.array([pa, pc], np.float32), "iso_x": iso_x, "iso_y": iso_y}
    for k, m in enumerate(models):
        arrays[f"s{k}_W0"] = m.body[0].weight.detach().cpu().numpy().T; arrays[f"s{k}_b0"] = m.body[0].bias.detach().cpu().numpy()
        arrays[f"s{k}_W1"] = m.body[3].weight.detach().cpu().numpy().T; arrays[f"s{k}_b1"] = m.body[3].bias.detach().cpu().numpy()
        arrays[f"s{k}_w"] = m.medium.weight.detach().cpu().numpy()[0]; arrays[f"s{k}_b"] = m.medium.bias.detach().cpu().numpy()
    pick = te[:24]
    arrays["test_f36"], arrays["test_emb"], arrays["test_logit"] = F[pick], L[pick], logits(models, Z[pick])
    meta = {"version": "film-head-20261111-v2layers", "feature_names": __import__("research.lut.engine.film_likeness", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES,
            "inputs": "36 film statistics (FEATURE_NAMES order) + DINOv2-small [L4 CLS, L4 mean, L8 CLS, L8 mean, L12 CLS, L12 mean], each part L2-normalised (2,304 values)",
            "architecture": "5 fold models (2 x 128 GELU, dropout 0.3), logits averaged, isotonic calibration on out-of-fold logits",
            "validation": {"auroc": round(a2, 4), "auroc_ci": [round(x, 4) for x in rep["v2_ci"]], "ece": None, "n_test": int(len(te)), "test_uploaders": int(len(ug)),
                           "deployed_v1_auroc_same_inputs": round(a1, 4), "paired_gain": [round(x, 4) for x in rep["paired_gain"]]},
            "training": {"film_digital_train_rows": int(tr.sum()), "targets": "capture medium only (film scan or digital); no model-generated labels"},
            "note": "measured estimate of scan-likeness; not a stock, camera or process identification"}
    meta["validation"]["ece"] = round(ece(p2, yt), 4)
    arrays_out = {k: np.asarray(v, np.float32) for k, v in arrays.items()}
    arrays_out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(DIST / "film_head_v2layers.npz", **arrays_out)
    rep["artefact_bytes"] = (DIST / "film_head_v2layers.npz").stat().st_size
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "film_head_v2.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
