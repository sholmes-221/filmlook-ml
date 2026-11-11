"""Calibrate, test for robustness, and export the film head as plain NumPy weights for the website.

  python -m research.claude_annotations.film_head_export   -> dist/film_head/film_head_v1.json, reports/claude/film_head_export.json

Protocol (nothing read on test before the choices were fixed): the head is trained on the training uploaders with the
annotations as auxiliary targets (film_head.py, seeds 0..2 averaged), its logit is Platt-calibrated on grouped
out-of-fold logits of the training uploaders, and then read ONCE on the test uploaders: AUROC, ECE, accuracy at 0.5,
and the three perturbations the shipped estimate was measured with (JPEG q50, 0.5x resize, monochrome rows only).
The export holds the 36-statistic normalisation, the two hidden layers, the medium output, the Platt map and the
validation numbers; the website gates on them exactly as it gates the current artefact (AUROC >= 0.75, ECE <= 0.10).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.claude_annotations import schema as S  # noqa: E402
from research.claude_annotations.film_eval import EMB as FILM_EMB, SET as FILM_SET, rows as film_rows  # noqa: E402
from research.claude_annotations.film_head import AUX, Head, auroc_ci  # noqa: E402
from research.lut.engine.film_likeness import FEATURE_NAMES, film_features  # noqa: E402

DIST = ROOT / "dist" / "film_head"; OUT = ROOT / "reports" / "claude" / "film_head_export.json"


def ece(p, y, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p > lo) & (p <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def train(Z, A, y, seeds=3, epochs=60, hidden=128):
    dev = "cuda" if torch.cuda.is_available() else "cpu"; models = []
    Xt, At, yt = torch.tensor(Z, device=dev), torch.tensor(A, device=dev), torch.tensor(y, device=dev)
    w_pos = float((y[y >= 0] == 0).sum() / max(1, (y == 1).sum()))
    for seed in range(seeds):
        torch.manual_seed(seed); m = Head(Z.shape[1], A.shape[1], hidden).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            m.train(); perm = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]; med, aux = m(Xt[b]); known = yt[b] >= 0
                loss = torch.nn.functional.binary_cross_entropy_with_logits(aux[:, :-1], At[b][:, :-1]) + torch.nn.functional.mse_loss(aux[:, -1], At[b][:, -1])
                if known.any(): loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(med[known], yt[b][known], pos_weight=torch.tensor(w_pos, device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
        models.append(m.eval())
    return models


def logits(models, Z):
    dev = next(models[0].parameters()).device
    with torch.no_grad():
        return np.mean([m(torch.tensor(Z, device=dev))[0].cpu().numpy() for m in models], 0)


def numpy_forward(art, f36, emb):
    z = (np.asarray(f36, np.float32) - art["mean36"]) / art["std36"]; x = np.concatenate([z, np.asarray(emb, np.float32)])
    outs = []
    for seed in art["seeds"]:
        h = x
        for W, b in seed["layers"]:
            h = 0.5 * (h @ W + b) * (1 + np.vectorize(lambda v: __import__("math").erf(v / 2 ** 0.5))(h @ W + b))  # GELU
        outs.append(float(h @ seed["medium_w"] + seed["medium_b"]))
    return float(np.mean(outs))


def main() -> None:
    cheap = P.cfg()["models"]["cheap"]; rs = film_rows()
    F = np.nan_to_num(np.load(FILM_SET / "features.npy").astype(np.float32)); X = np.load(FILM_EMB)["X"].astype(np.float32)
    ann = P.annotated("filmset", cheap); ids = [f"filmset:{r['dataset']}:{Path(r['path']).name}" for r in rs]
    A = np.stack([S.vectorise(ann[i]["annotation"]) for i in ids])[:, AUX].astype(np.float32)
    y = np.array([r["label"] for r in rs], np.float32); g = np.array([r["uploader"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs])
    bank = P.annotated("bank", cheap); bank_ids = sorted(bank)
    bE = np.load(ROOT / "data/embeddings/photographers/prod.npz"); pos = {w: i for i, w in enumerate(bE["work_ids"].tolist())}
    bf = np.load(ROOT / "data/embeddings/photographers/film36.npz"); bpos = {w: i for i, w in enumerate(bf["work_ids"].tolist())}
    FB = np.nan_to_num(bf["F"][[bpos[w] for w in bank_ids]]); XB = bE["X"][[pos[w] for w in bank_ids]].astype(np.float32)
    AB = np.stack([S.vectorise(bank[w]["annotation"]) for w in bank_ids])[:, AUX].astype(np.float32)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = np.hstack([(F - mu) / sd, X]).astype(np.float32); ZB = np.hstack([(FB - mu) / sd, XB]).astype(np.float32)
    # Platt map from grouped 5-fold out-of-fold logits on the training uploaders (bank rows always in training)
    tri = np.flatnonzero(tr); ug = np.unique(g[tri]); fold = {u: i % 5 for i, u in enumerate(np.random.default_rng(0).permutation(ug))}; f = np.array([fold[u] for u in g[tri]])
    oof = np.zeros(len(tri), np.float32)
    for k in range(5):
        a, b = tri[f != k], tri[f == k]
        ms = train(np.vstack([Z[a], ZB]), np.vstack([A[a], AB]), np.concatenate([y[a], np.full(len(ZB), -1, np.float32)]), seeds=1, epochs=40)
        oof[f == k] = logits(ms, Z[b])
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof[:, None], y[tri]); pa, pc = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    models = train(np.vstack([Z[tr], ZB]), np.vstack([A[tr], AB]), np.concatenate([y[tr], np.full(len(ZB), -1, np.float32)]))
    te = np.flatnonzero(~tr); lt = logits(models, Z[te]); p = 1 / (1 + np.exp(-(pa * lt + pc)))
    base = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit((F[tr] - mu) / sd, y[tr]).decision_function((F[te] - mu) / sd)
    report = {"test": auroc_ci(y[te], lt, g[te], base), "ece": round(ece(p, y[te]), 4), "accuracy_at_0.5": round(float(((p >= 0.5) == (y[te] == 1)).mean()), 4),
              "platt": {"a": pa, "c": pc}, "n_test": int(len(te)), "test_uploaders": int(len(set(g[te])))}
    mono = np.array([not r.get("colour", True) for r in rs])[te]
    if mono.sum() > 30 and len(set(y[te][mono])) == 2:
        report["monochrome_rows_auroc"] = round(float(roc_auc_score(y[te][mono], lt[mono])), 4)
    report["colour_rows_auroc"] = round(float(roc_auc_score(y[te][~mono], lt[~mono])), 4)
    # robustness on a 400-row subsample: recompute statistics and embedding from perturbed pixels
    from research.photographers.embed import prod_encoder
    enc = prod_encoder("cpu"); rng = np.random.default_rng(0); sub = rng.choice(te, min(400, len(te)), replace=False)
    def perturbed(kind):
        fs, xs = [], []
        for i in sub:
            im = Image.open(rs[i]["path"]).convert("RGB")
            if kind == "jpeg50":
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=50); im = Image.open(buf).convert("RGB")
            elif kind == "half":
                im = im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.LANCZOS)
            fs.append(film_features(np.asarray(im))); xs.append(enc([im])[0])
        Fp = np.nan_to_num(np.stack(fs).astype(np.float32)); Zp = np.hstack([(Fp - mu) / sd, np.stack(xs).astype(np.float32)])
        return roc_auc_score(y[sub], logits(models, Zp)), roc_auc_score(y[sub], LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit((F[tr] - mu) / sd, y[tr]).decision_function((Fp - mu) / sd))
    report["robustness_400"] = {"clean": {"head": round(float(roc_auc_score(y[sub], lt[np.searchsorted(te, sub)])), 4), "measured36": round(float(roc_auc_score(y[sub], base[np.searchsorted(te, sub)])), 4)}}
    for kind in ("jpeg50", "half"):
        h, m36 = perturbed(kind); report["robustness_400"][kind] = {"head": round(float(h), 4), "measured36": round(float(m36), 4)}
    # export
    art = {"version": time.strftime("film-head-%Y%m%d") + "-v1", "inputs": "36 film statistics (FEATURE_NAMES order) + 768 DINOv2-small [CLS, mean patch]",
           "feature_names": FEATURE_NAMES, "mean36": mu.tolist(), "std36": sd.tolist(), "activation": "gelu", "platt": {"a": pa, "c": pc},
           "seeds": [{"layers": [[m.body[0].weight.detach().cpu().numpy().T.tolist(), m.body[0].bias.detach().cpu().numpy().tolist()],
                                 [m.body[3].weight.detach().cpu().numpy().T.tolist(), m.body[3].bias.detach().cpu().numpy().tolist()]],
                      "medium_w": m.medium.weight.detach().cpu().numpy()[0].tolist(), "medium_b": float(m.medium.bias.detach().cpu().numpy()[0])} for m in models],
           "validation": {"auroc": report["test"]["auroc"], "auroc_ci": report["test"]["ci"], "ece": report["ece"], "n_test": report["n_test"], "test_uploaders": report["test_uploaders"],
                          "baseline_measured36_auroc": round(report["test"]["auroc"] - report["test"]["delta_vs_measured36"], 4)},
           "training": {"film_digital_train_rows": int(tr.sum()), "auxiliary_annotated_rows": int(tr.sum()) + len(ZB), "auxiliary_targets": "claude-haiku-5-5 style-annotation-v2 rendering fields (not labels of the medium)"},
           "note": "measured estimate of scan-likeness; not a stock, camera or process identification"}
    # parity check: NumPy forward against torch on 50 rows
    art_np = {**art, "mean36": np.array(art["mean36"], np.float32), "std36": np.array(art["std36"], np.float32),
              "seeds": [{"layers": [(np.array(W, np.float32), np.array(b, np.float32)) for W, b in s["layers"]], "medium_w": np.array(s["medium_w"], np.float32), "medium_b": s["medium_b"]} for s in art["seeds"]]}
    diff = max(abs(numpy_forward(art_np, F[i], X[i]) - float(logits(models, Z[i:i + 1])[0])) for i in te[:50])
    report["numpy_parity_max_abs_diff"] = round(float(diff), 6)
    t = time.perf_counter(); [numpy_forward(art_np, F[i], X[i]) for i in te[:50]]; report["numpy_forward_ms"] = round((time.perf_counter() - t) / 50 * 1000, 2)
    DIST.mkdir(parents=True, exist_ok=True); (DIST / "film_head_v1.json").write_text(json.dumps(art))
    OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
