#!/usr/bin/env python
"""Train and evaluate the MobileNetV3-Small student (supervised, or supervised + DINOv2 distillation).

  python scripts/train_student.py --kd 0.0 --tag sup      # supervised only
  python scripts/train_student.py --kd 1.0 --tag kd       # + cosine distillation to DINOv2-small embeddings

Training images: originals plus their edited derivatives (train split only; edits carry attribute labels; capture
type is masked on edits). Model selection uses validation originals; the reported numbers are TEST originals, the
same 127 images as the DINOv2 baseline. Output: checkpoints/student_<tag>/ (ONNX fp32 + int8 + meta.json), and
reports/student_<tag>.json.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

from filmlook_infer.compact import preprocess
from filmlook_ml import dinov2_baseline as B
from filmlook_ml import paths as P
from filmlook_ml import student as S
from filmlook_ml.annotation.workflow import discover
from filmlook_ml.compliance.preflight import require_training_preflight
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import filter_manifest, load_quarantine
from filmlook_ml.datasets.registry import Registry

HEADS = ["capture_type", "temperature", "saturation", "contrast", "key", "grain", "sharpness"]


def _sq(path):
    return S.load_square(path, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    a = ap.parse_args()
    require_training_preflight()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free < a.min_free_gb:
            raise SystemExit(f"only {free:.1f} GB free on the shared GPU")

    lcfg = P.load_yaml("labeling.yaml")
    man = filter_manifest(read_jsonl(P.MANIFESTS / "manifest.jsonl"))
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    fams = {r["family_id"] for r in eligible}
    rows = [r for r in man if r["family_id"] in fams]                      # originals + edits of eligible photos
    feats = json.loads((P.EMBEDDINGS / B.FEAT_FILE).read_text())
    z = np.load(P.EMBEDDINGS / B.EMB_FILE)
    E = dict(zip(z["sha"].tolist(), z["emb"]))
    split = np.array([r["split"] for r in rows])
    orig = np.array([r["edit_type"] == "original" for r in rows])
    emb = np.stack([E[r["sha256"]] for r in rows])
    Zt, mu = B.normalise(emb, (split == "train") & orig)

    labels = {}
    att = [B.attribute_labels(feats[r["sha256"]]) for r in rows]
    for h in HEADS:
        if h == "capture_type":
            vals = [r.get("medium") if (r["edit_type"] == "original" and r.get("medium") in B.CAPTURE_CLASSES) else None
                    for r in rows]
        else:
            vals = [x[h] for x in att]
        classes = sorted({v for v, s in zip(vals, split) if v is not None and s == "train"})
        labels[h] = (classes, np.array([classes.index(v) if v in classes else -1 for v in vals]))

    tr = np.where(split == "train")[0]
    t0 = time.time()
    with Pool(24) as pool:
        X = np.stack(pool.map(_sq, [rows[i]["path"] for i in tr], chunksize=16))
    print(f"cached {len(tr)} train squares in {time.time() - t0:.0f}s", file=sys.stderr)
    evals = {s: [i for i in np.where((split == s) & orig)[0]] for s in ("val", "test")}
    with Pool(24) as pool:
        Xe = {s: np.concatenate(pool.map(preprocess, [rows[i]["path"] for i in idx], chunksize=8))
              for s, idx in evals.items()}

    model = S.Student({h: len(labels[h][0]) for h in HEADS}).to(dev)
    weights = {}
    for h in HEADS:
        y = labels[h][1][tr]
        cnt = np.bincount(y[y >= 0], minlength=len(labels[h][0])).astype(float)
        weights[h] = torch.tensor(cnt.sum() / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": 3e-4},
                             {"params": [*model.neck.parameters(), *model.heads.parameters(), *model.kd.parameters()],
                              "lr": 1e-3}], weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(tr) / 128))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps)
    Yt = {h: torch.from_numpy(labels[h][1][tr]).to(dev) for h in HEADS}
    Kt = torch.from_numpy(Zt[tr]).float().to(dev)
    rng = np.random.default_rng(a.seed)

    def predict(xs):
        model.eval()
        out = {h: [] for h in HEADS}
        with torch.no_grad():
            for i in range(0, len(xs), 256):
                o = model(torch.from_numpy(xs[i:i + 256]).to(dev))
                for h, t in zip(HEADS, o):
                    out[h].append(torch.softmax(t.float(), 1).cpu().numpy())
        return {h: np.concatenate(v) for h, v in out.items()}

    def score(split_name):
        Pr = predict(Xe[split_name])
        idx = np.array(evals[split_name])
        res = {}
        for h in HEADS:
            classes, y = labels[h]
            m = y[idx] >= 0
            yy = np.array(classes)[y[idx][m]]
            groups = np.array([rows[i]["group_id"] for i in idx[m]], dtype=object)
            res[h] = B._metrics(yy, Pr[h][m], classes, groups if split_name == "test" else None,
                                n_boot=300 if split_name == "test" else 0)
        return res

    best, history = None, []
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for b in range(0, len(perm), 128):
            bi = perm[b:b + 128]
            ox, oy = rng.integers(0, 33, 2)
            xb = X[bi][:, oy:oy + 224, ox:ox + 224]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            x = S.to_tensor(np.ascontiguousarray(xb), dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                outs, kd = model(x, with_kd=True)
                loss = 0.0
                for h, o in zip(HEADS, outs):
                    y = Yt[h][bi]
                    m = y >= 0
                    if m.any():
                        loss = loss + nn.functional.cross_entropy(o[m].float(), y[m], weight=weights[h])
                if a.kd > 0:
                    loss = loss + a.kd * (1 - nn.functional.cosine_similarity(kd.float(), Kt[bi], dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(bi)
        val = score("val")
        vm = float(np.mean([val[h]["macro_f1"] for h in HEADS]))
        history.append({"epoch": ep, "train_loss": round(tot / len(tr), 4), "val_mean_macro_f1": round(vm, 4)})
        print(history[-1], file=sys.stderr)
        if best is None or vm > best[0]:
            best = (vm, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    test = score("test")
    out = P.CHECKPOINTS / f"student_{a.tag}"
    sizes = S.export_onnx(model, out, {"model_version": f"{S.STUDENT_VERSION}-{a.tag}",
                                       "heads": {h: labels[h][0] for h in HEADS}, "kd_weight": a.kd,
                                       "label_definitions": "capture_type: EXIF metadata; others: thresholds on measured "
                                                            "statistics (filmlook_infer.style)", "seed": a.seed})
    model.to(dev)                                 # export moved it to the CPU
    # parity: ONNX fp32 / int8 vs PyTorch on test originals
    import onnxruntime as ort
    xs = Xe["test"][:64]
    ref = predict(xs)
    parity = {}
    for f in sizes:
        s = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        o = s.run(None, {"pixel_values": xs})
        parity[f] = {h: round(float(np.mean(np.argmax(o[i], 1) == ref[h].argmax(1))), 4) for i, h in enumerate(HEADS)}
    rep = {"tag": a.tag, "kd_weight": a.kd, "best_epoch": best[1], "val_mean_macro_f1": round(best[0], 4),
           "history": history, "test": test, "onnx_mb": sizes, "onnx_top1_agreement_with_pytorch": parity,
           "params": int(sum(p.numel() for p in model.parameters())),
           "train_images": int(len(tr)), "test_originals": len(evals["test"])}
    rep["test_summary"] = {h: (test[h]["macro_f1"], test[h].get("macro_f1_ci95")) for h in HEADS}
    (P.REPORTS / f"student_{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("tag", "best_epoch", "val_mean_macro_f1", "test_summary", "onnx_mb",
                                          "onnx_top1_agreement_with_pytorch")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
