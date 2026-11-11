"""A small local head for the film-likeness estimate, trained on Claude's rendering annotations as auxiliary targets.

  python -m research.claude_annotations.film_head [--epochs 60] [--seeds 3]   -> reports/claude/film_head.json, dist/film_head/

Question: can a non-linear head on the production inputs (36 measured statistics + the 768-value DINOv2-small embedding)
learn what the annotator saw (grain, glow, faded blacks, clean digital rendering, the 0-4 impression) well enough to
improve the capture-medium estimate on held-out uploaders? The linear ridge heads could not (film_eval.py: +0.009).

Training rows: the film / digital set's training uploaders (1,104 rows) plus the 4,548 bank works (public domain, all
annotated), whose capture medium is unknown and which therefore contribute only to the auxiliary annotation targets.
Loss: binary cross-entropy on the medium (film-set rows only) + BCE on 15 rendering flags + MSE on the impression
(every annotated row). Held-out reading: AUROC on the film-set test uploaders (893 film / 246 digital, 502 uploaders),
bootstrap over uploaders, paired with the shipped 36-statistic logistic model. The medium label is never a Claude output;
the auxiliary targets are, under the owner's recorded permission (docs/AI_DATA_POLICY.md; preflight PASS).
Nothing here is promoted unless it beats the shipped model's AUROC with a confidence interval that excludes zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.claude_annotations import schema as S  # noqa: E402
from research.claude_annotations.film_eval import rows as film_rows, EMB as FILM_EMB, SET as FILM_SET  # noqa: E402

OUT = ROOT / "reports" / "claude" / "film_head.json"; DIST = ROOT / "dist" / "film_head"
NAMES = S.feature_names()
AUX = [i for i, n in enumerate(NAMES) if n.startswith("film_rendering:")] + [NAMES.index("film_like_impression")]


class Head(torch.nn.Module):
    def __init__(self, d_in: int, n_aux: int, hidden: int = 128, p: float = 0.3):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(p))
        self.medium = torch.nn.Linear(hidden, 1); self.aux = torch.nn.Linear(hidden, n_aux)

    def forward(self, x):
        h = self.body(x); return self.medium(h).squeeze(-1), self.aux(h)


def auroc_ci(y, s, g, base=None, n=1000):
    rng = np.random.default_rng(0); ug = np.unique(g); mem = {u: np.flatnonzero(g == u) for u in ug}; vals, diffs = [], []
    for _ in range(n):
        idx = np.concatenate([mem[u] for u in rng.choice(ug, len(ug))])
        if len(set(y[idx].tolist())) < 2: continue
        a = roc_auc_score(y[idx], s[idx]); vals.append(a)
        if base is not None: diffs.append(a - roc_auc_score(y[idx], base[idx]))
    out = {"auroc": round(float(roc_auc_score(y, s)), 4), "ci": [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]}
    if base is not None:
        out["delta_vs_measured36"] = round(out["auroc"] - float(roc_auc_score(y, base)), 4); out["delta_ci"] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-aux", action="store_true", help="ablation: the same head and data without the annotation targets (medium only)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); cheap = P.cfg()["models"]["cheap"]
    rs = film_rows(); F = np.nan_to_num(np.load(FILM_SET / "features.npy").astype(np.float32)); X = np.load(FILM_EMB)["X"].astype(np.float32)
    ann = P.annotated("filmset", cheap); ids = [f"filmset:{r['dataset']}:{Path(r['path']).name}" for r in rs]
    A = np.stack([S.vectorise(ann[i]["annotation"]) for i in ids])[:, AUX]
    y = np.array([r["label"] for r in rs], np.float32); g = np.array([r["uploader"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs])
    # bank rows: auxiliary targets only (medium unknown). Their 36 statistics must be computed with the same function.
    bank = P.annotated("bank", cheap); bank_ids = sorted(bank)
    bE = np.load(ROOT / "data/embeddings/photographers/prod.npz"); pos = {w: i for i, w in enumerate(bE["work_ids"].tolist())}
    bank_feats = ROOT / "data/embeddings/photographers/film36.npz"
    if not bank_feats.exists():
        from filmlook_ml.refdb import manifest
        from research.lut.engine.film_likeness import film_features
        from research.lut.engine.engine import load_image
        rows_b = []
        for k, w in enumerate(bank_ids):
            rows_b.append(film_features(load_image(manifest.image_path(w).read_bytes())))
            if k % 500 == 0: print("bank statistics", k, len(bank_ids), flush=True)
        np.savez(bank_feats, work_ids=np.array(bank_ids), F=np.stack(rows_b).astype(np.float32))
    bf = np.load(bank_feats); bpos = {w: i for i, w in enumerate(bf["work_ids"].tolist())}
    FB = np.nan_to_num(bf["F"][[bpos[w] for w in bank_ids]]); XB = bE["X"][[pos[w] for w in bank_ids]].astype(np.float32)
    AB = np.stack([S.vectorise(bank[w]["annotation"]) for w in bank_ids])[:, AUX]
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = np.hstack([(F - mu) / sd, X]).astype(np.float32); ZB = np.hstack([(FB - mu) / sd, XB]).astype(np.float32)
    base = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit((F[tr] - mu) / sd, y[tr]).decision_function((F[~tr] - mu) / sd)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(np.vstack([Z[tr], ZB]), device=dev); Atr = torch.tensor(np.vstack([A[tr], AB]), device=dev)
    ytr = torch.tensor(np.concatenate([y[tr], np.full(len(ZB), -1, np.float32)]), device=dev)  # -1 = unknown medium
    w_pos = float((1 - y[tr]).sum() / max(1, y[tr].sum()))
    scores = []
    for seed in range(a.seeds):
        torch.manual_seed(seed); m = Head(Z.shape[1], len(AUX), a.hidden).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        n = len(Xtr)
        for ep in range(a.epochs):
            m.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 256):
                b = perm[i:i + 256]; med, aux = m(Xtr[b]); known = ytr[b] >= 0
                loss = torch.zeros((), device=dev) if a.no_aux else torch.nn.functional.binary_cross_entropy_with_logits(aux[:, :-1], Atr[b][:, :-1]) + torch.nn.functional.mse_loss(aux[:, -1], Atr[b][:, -1])
                if known.any():
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(med[known], ytr[b][known], pos_weight=torch.tensor(w_pos, device=dev))
                if not loss.requires_grad:  # ablation batch made only of bank rows: nothing to learn from
                    continue
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            scores.append(m(torch.tensor(Z[~tr], device=dev))[0].cpu().numpy())
        if seed == 0 and not a.no_aux:
            DIST.mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), DIST / "film_head_seed0.pt")
    s = np.mean(scores, 0)
    report = {"train_rows": int(tr.sum()), "bank_rows_aux": len(ZB), "test_rows": int((~tr).sum()), "hidden": a.hidden, "epochs": a.epochs, "seeds": a.seeds,
              "measured36": auroc_ci(y[~tr], base, g[~tr]), "film_head_mean_of_seeds": auroc_ci(y[~tr], s, g[~tr], base),
              "per_seed_auroc": [round(float(roc_auc_score(y[~tr], v)), 4) for v in scores]}
    # also the two combined by logistic regression on the training rows' out-of-sample behaviour is not available; report a simple average of ranks
    from scipy.stats import rankdata
    comb = rankdata(base) + rankdata(s); report["measured36+film_head_rank_average"] = auroc_ci(y[~tr], comb, g[~tr], base)
    report["aux_targets"] = not a.no_aux
    out = OUT.with_name(OUT.stem + (a.tag or ("_noaux" if a.no_aux else "")) + ".json")
    out.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
