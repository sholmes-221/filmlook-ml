"""Train and evaluate preset classifiers on the synthetic features (scene-disjoint splits).

    python -m research.lut.train --feature-set colour --model mlp --name colour_mlp \
        --holdout "Candido_800T,Phoenix_200,Lomography_Color_400"

Inputs: data/lut/synth/features.npz (+ manifest.jsonl). Feature sets: stats24 | cr80 | labhist | colour | dinov2 |
dinov2+colour. Models: lr (multinomial logistic regression), mlp (2x256 ReLU, dropout, AdamW; GPU if available; the
GPU seconds are appended to reports/lut/gpu_ledger.jsonl), knn (k=5, cosine on z-scored features).
Held-out presets are excluded from training and used as unknown styles at test time (rejection AUROC with the
max-softmax score), together with the untouched filtered uploads (`unknown_filtered`).
Outputs: data/lut/models/<name>.npz (numpy-only weights, scaler, classes, temperature) and
reports/lut/train_<name>.json (metrics, confusion matrix, per-class accuracy, content probe).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.lut.evaluate import auroc, confusion, ece, fit_temperature, format_confusion, grouped_bootstrap, softmax, topk_hits
from research.lut.features import COLOUR_SLICES

DEFAULT_HOLDOUT = "Candido_800T,Phoenix_200,Lomography_Color_400"


def load_data(features="data/lut/synth/features.npz", manifest="data/lut/synth/manifest.jsonl"):
    z = np.load(features, allow_pickle=True)
    rows = {json.loads(l)["path"]: json.loads(l) for l in open(manifest)}
    meta = [rows[p] for p in z["paths"]]
    return z, meta


def select(z, feature_set: str) -> np.ndarray:
    parts = []
    for name in feature_set.split("+"):
        if name == "dinov2":
            parts.append(z["dinov2"].astype(np.float32))
        else:
            parts.append(z["colour"][:, COLOUR_SLICES[name]].astype(np.float32))
    X = np.concatenate(parts, 1)
    return np.nan_to_num(X)


class NumpyClassifier:
    """Scaler + (logistic | MLP) weights in numpy, so the runtime needs no torch/sklearn."""

    def __init__(self, d: dict):
        self.d = d
        self.classes = list(d["classes"])
        self.T = float(d.get("T", 1.0))

    def logits(self, X: np.ndarray) -> np.ndarray:
        h = (np.asarray(X, np.float32) - self.d["mean"]) / self.d["std"]
        n = int(self.d["n_layers"])
        for i in range(n):
            h = h @ self.d[f"W{i}"] + self.d[f"b{i}"]
            if i < n - 1:
                h = np.maximum(h, 0)
        return h

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        return softmax(self.logits(X), self.T if calibrated else 1.0)

    def save(self, path):
        np.savez(path, **{k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in self.d.items()})

    @staticmethod
    def load(path) -> "NumpyClassifier":
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["classes"] = [str(c) for c in d["classes"]]
        return NumpyClassifier(d)


def fit_lr(Xtr, ytr, n_classes, C=1.0):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(Xtr, ytr)
    W = np.zeros((Xtr.shape[1], n_classes), np.float32)
    b = np.zeros(n_classes, np.float32)
    W[:, m.classes_] = m.coef_.T
    b[m.classes_] = m.intercept_
    return {"n_layers": 1, "W0": W, "b0": b}


def fit_mlp(Xtr, ytr, Xva, yva, n_classes, hidden=256, epochs=80, seed=0, ledger="reports/lut/gpu_ledger.jsonl", tag=""):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(hidden, n_classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            xb = Xt[idx] + 0.05 * torch.randn_like(Xt[idx])     # feature-space jitter
            loss = nn.functional.cross_entropy(net(xb), yt[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xv).argmax(1) == yv).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    secs = time.time() - t0
    if dev == "cuda":
        Path(ledger).parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a") as f:
            f.write(json.dumps({"task": f"train_mlp{tag}", "gpu_seconds": round(secs, 1), "device": torch.cuda.get_device_name(0),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    net.load_state_dict(best_state)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    d = {"n_layers": len(lin)}
    for i, m in enumerate(lin):
        d[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        d[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    return d, {"best_val_acc": best, "train_seconds": secs, "device": dev}


def knn_proba(Xtr, ytr, Xte, n_classes, k=5):
    a = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-6)
    b = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-6)
    P = np.zeros((len(Xte), n_classes))
    for i in range(0, len(b), 512):
        s = b[i:i + 512] @ a.T
        nn_idx = np.argsort(-s, axis=1)[:, :k]
        for r, row in enumerate(nn_idx):
            np.add.at(P[i + r], ytr[row], 1.0 / k)
    return P


def content_probe(X: np.ndarray, scenes: np.ndarray, presets: np.ndarray, mean, std) -> dict:
    """Nearest neighbour (cosine, z-scored) within a set: how often is it the same scene vs the same preset?"""
    Z = (X - mean) / std
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-6)
    same_scene = same_preset = 0
    for i in range(0, len(Z), 512):
        s = Z[i:i + 512] @ Z.T
        for r in range(s.shape[0]):
            s[r, i + r] = -np.inf
        nn_idx = s.argmax(1)
        same_scene += (scenes[nn_idx] == scenes[i:i + 512]).sum()
        same_preset += (presets[nn_idx] == presets[i:i + 512]).sum()
    return {"nn_same_scene": float(same_scene / len(Z)), "nn_same_preset": float(same_preset / len(Z)),
            "chance_same_scene": float(1 / len(np.unique(scenes))), "chance_same_preset": float(1 / len(np.unique(presets)))}


def run(feature_set: str, model: str, name: str, holdout: str, features: str, manifest: str, out_dir="data/lut/models",
        rep_dir="reports/lut", seed=0, label_field: str = "preset") -> dict:
    z, meta = load_data(features, manifest)
    X = select(z, feature_set)
    preset = np.array([str(m[label_field]) for m in meta])
    split = np.array([m["split"] for m in meta])
    scene = np.array([m["scene_id"] for m in meta])
    held = [h for h in holdout.split(",") if h] if holdout else []
    classes = sorted(c for c in np.unique(preset) if c not in held and c != "unknown_filtered")
    cidx = {c: i for i, c in enumerate(classes)}
    known = np.array([p in cidx for p in preset])
    y = np.array([cidx.get(p, -1) for p in preset])
    tr = (split == "train") & known
    va = (split == "val") & known
    te = (split == "test") & known
    te_unknown_preset = (split == "test") & np.isin(preset, held)
    neg = split == "negatives"
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mean) / std
    d = {"classes": np.array(classes), "mean": mean.astype(np.float32), "std": std.astype(np.float32), "feature_set": feature_set, "model": model,
         "label_field": label_field}
    info = {}
    if model == "lr":
        d.update(fit_lr(Z[tr], y[tr], len(classes)))
    elif model == "mlp":
        w, info = fit_mlp(Z[tr], y[tr], Z[va], y[va], len(classes), seed=seed, tag=f"_{name}")
        d.update(w)
    elif model == "knn":
        d.update({"n_layers": 0})
    clf = NumpyClassifier({**d, "mean": np.zeros_like(mean), "std": np.ones_like(std)}) if model != "knn" else None

    def proba(mask, T=1.0):
        if model == "knn":
            return knn_proba(Z[tr], y[tr], Z[mask], len(classes))
        return softmax(clf.logits(Z[mask]), T)

    T = 1.0
    if model != "knn":
        T = fit_temperature(clf.logits(Z[va]), y[va])
    d["T"] = T
    res = {"name": name, "feature_set": feature_set, "model": model, "dim": int(X.shape[1]), "classes": classes, "holdout": held, "label_field": label_field,
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "n_test_scenes": int(len(np.unique(scene[te]))),
           "n_unknown_preset_test": int(te_unknown_preset.sum()), "n_negatives": int(neg.sum()), "temperature": T, "fit_info": info}
    for sname, mask in (("val", va), ("test", te)):
        P = proba(mask, T)
        hits1, hits3 = topk_hits(P, y[mask], 1), topk_hits(P, y[mask], 3)
        m1 = grouped_bootstrap(hits1, scene[mask])
        m3 = grouped_bootstrap(hits3, scene[mask])
        res[sname] = {"top1": m1[0], "top1_ci": m1[1:], "top3": m3[0], "top3_ci": m3[1:], "ece_calibrated": ece(P, y[mask]),
                      "ece_uncalibrated": ece(proba(mask, 1.0), y[mask]), "chance": 1 / len(classes)}
        if sname == "test":
            cm = confusion(y[mask], P.argmax(1), len(classes))
            res["test"]["confusion"] = cm.tolist()
            res["test"]["per_class_acc"] = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
            res["test"]["confusion_text"] = format_confusion(cm, classes)
            # rejection: max-softmax of known test vs unknown-preset test / filtered negatives
            conf_known = P.max(1)
            rej = {}
            if te_unknown_preset.any():
                Pu = proba(te_unknown_preset, T)
                rej["auroc_vs_heldout_presets"] = auroc(conf_known, Pu.max(1))
                rej["heldout_predicted_as"] = {c: int((Pu.argmax(1) == i).sum()) for i, c in enumerate(classes)}
                for h in held:
                    mh = te_unknown_preset & (preset == h)
                    Ph = proba(mh, T)
                    rej[f"heldout_{h}_top_pred"] = classes[int(np.bincount(Ph.argmax(1), minlength=len(classes)).argmax())]
            if neg.any():
                Pn = proba(neg, T)
                rej["auroc_vs_filtered_uploads"] = auroc(conf_known, Pn.max(1))
                rej["filtered_predicted_none_fraction"] = float((Pn.argmax(1) == cidx["none"]).mean()) if "none" in cidx else None
            res["test"]["rejection"] = rej
    res["content_probe_test"] = content_probe(X[te], scene[te], preset[te], mean, std)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if model != "knn":
        NumpyClassifier(d).save(Path(out_dir) / f"{name}.npz")
    Path(rep_dir).mkdir(parents=True, exist_ok=True)
    (Path(rep_dir) / f"train_{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"[{name}] {feature_set}/{model} dim={X.shape[1]} train={tr.sum()} test={te.sum()} ({res['n_test_scenes']} scenes) | "
          f"val top1 {res['val']['top1']:.3f} | test top1 {res['test']['top1']:.3f} [{res['test']['top1_ci'][0]:.3f}, {res['test']['top1_ci'][1]:.3f}] "
          f"top3 {res['test']['top3']:.3f} ECE {res['test']['ece_calibrated']:.3f} (T={T:.2f}) | reject AUROC heldout "
          f"{res['test']['rejection'].get('auroc_vs_heldout_presets', float('nan')):.3f} filtered {res['test']['rejection'].get('auroc_vs_filtered_uploads', float('nan')):.3f} "
          f"| NN same scene {res['content_probe_test']['nn_same_scene']:.3f} same preset {res['content_probe_test']['nn_same_preset']:.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", default="colour")
    ap.add_argument("--model", default="mlp", choices=["lr", "mlp", "knn"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    ap.add_argument("--features", default="data/lut/synth/features.npz")
    ap.add_argument("--manifest", default="data/lut/synth/manifest.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-field", default="preset", help="manifest field holding the class label (preset | lut_id | stock)")
    a = ap.parse_args()
    run(a.feature_set, a.model, a.name, a.holdout, a.features, a.manifest, seed=a.seed, label_field=a.label_field)


if __name__ == "__main__":
    main()
