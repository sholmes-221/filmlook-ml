#!/usr/bin/env python
"""Style-sensitive MobileNetV3-Small student trained on the synthetic study, evaluated on unseen generators and real film.

  python scripts/train_style_student.py --train-generators filmsynth --tag style_fs     # leave SDXL out
  python scripts/train_style_student.py --train-generators filmsynth,sdxl --tag style_both

Objectives (all from synthetic parameter supervision or self-supervision; no human labels):
  treated   : binary, treated vs untreated (every generator);
  family    : 9-way filmsynth family (masked for other rows);
  params    : 6 filmsynth parameters, regression (masked);
  distil    : cosine to the 64-d SupCon projection of hybrid_style features (teacher trained on filmsynth train sources).
Evaluation (test sources only; BlueNeg is external):
  synthetic: family macro-F1 and treated AUROC on filmsynth test;
  LOGO     : treated AUROC on SDXL test outputs vs untreated test sources (unseen generator when trained on filmsynth);
  real     : BlueNeg previews vs CommonCatalog digital originals AUROC (P(treated)); same-stock retrieval p@5 among
             blue-intact previews, same roll excluded, with the student's 576-d and 64-d embeddings;
  content  : own-source rank-1 of treated filmsynth test images with the student embedding (low = style-dominated);
  robust   : treated prediction under JPEG q55 / 0.5x rescale / +0.5 EV on SDXL test outputs;
  baseline : the existing supervised student's P(film_scan) on the same BlueNeg-vs-digital task;
  cost     : ONNX fp32 size and 1-thread CPU latency.
"""
import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402

from filmlook_infer.compact import preprocess  # noqa: E402
from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml import student as S  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

SDX = P.DATA / "synthetic" / "sdxl_i2i_v1"
TARGETS = X.TARGETS


class StyleStudent(nn.Module):
    def __init__(self, n_family: int, kd_dim: int = 64):
        super().__init__()
        import torchvision
        m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features, self.pool = m.features, nn.AdaptiveAvgPool2d(1)
        self.treated = nn.Linear(576, 1)
        self.family = nn.Linear(576, n_family)
        self.params = nn.Linear(576, len(TARGETS))
        self.kd = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(), nn.Linear(256, kd_dim))

    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        return f, self.treated(f).squeeze(1), self.family(f), self.params(f), self.kd(f)


def _sq(p):
    return S.load_square(p, 256)


def _sq_real(p):
    return S.load_square(p, 256)


def _explicit_any(p, real):
    return X.explicit_of(X.to_short(load_image(p, max_side=1280))) if real else X._explicit_path(p)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, s)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-generators", default="filmsynth")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--synth", default="film_synth_v1", help="filmsynth folder under data/synthetic/")
    ap.add_argument("--real-distil", action="store_true", help="E23: add unlabeled CommonCatalog TRAIN originals as distillation-only inputs (teacher targets, no labels)")
    ap.add_argument("--no-blueneg-eval", action="store_true", help="selection run: skip the BlueNeg (final-only) evaluation")
    ap.add_argument("--kd-only-generators", default="", help="E23c: generators whose TRAIN outputs are distillation-only inputs (no labels), e.g. sdxl")
    ap.add_argument("--teacher-layers", default="4,8", help="E27: DINOv2 hidden-state layers (mean++std) in the teacher's style features; default L4+L8")
    ap.add_argument("--kd-target", default="supcon", choices=["supcon", "supcon_vic3", "pcaw64"],
                    help="E38: distillation target on the frozen teacher features: synthetic-family SupCon projection (default), the same with VICReg anti-collapse (lambda 3), or a label-free PCA-whitened 64-d head of the L4+L8 statistics fitted on the real distillation inputs")
    ap.add_argument("--vic-pooled", type=float, default=0.0,
                    help="E46: VICReg variance + covariance regularisation (this weight on both terms) applied to the 576-d pooled layer itself, so that its rank is kept without any teacher")
    ap.add_argument("--vic-mode", default="batch", choices=["batch", "source"],
                    help="E51: 'source' centres the VICReg features on their synthetic source's in-batch mean (batches led by 12 sources x 4 rows), expanding treatment variance only")
    a = ap.parse_args()
    SYN = P.DATA / "synthetic" / a.synth
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    t_all = time.time()
    gens = a.train_generators.split(",")

    syn = read_jsonl(SYN / "annotations.jsonl")
    sdx = read_jsonl(SDX / "annotations.jsonl")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    srcs = sorted({x["source_image_id"] for x in syn})
    split_of = {s: by_id[s]["split"] for s in srcs}
    sdx = [x for x in sdx if x["source_image_id"] in split_of]
    fams = sorted({x["augmentation_family"] for x in syn})

    # rows: (path, generator, split, source, family_idx or -1, params or None, treated)
    rows = []
    for x in syn:
        rows.append((str(P.ROOT / x["path"]), "filmsynth", split_of[x["source_image_id"]], x["source_image_id"],
                     fams.index(x["augmentation_family"]), [float(x["parameters"][t]) for t in TARGETS],
                     bool(x["parameters"].get("monochrome")), not x["augmentation_family"].startswith("identity")))
    for x in sdx:
        rows.append((str(P.ROOT / x["path"]), "sdxl", split_of[x["source_image_id"]], x["source_image_id"], -1, None, False, True))
    for s in srcs:
        rows.append((by_id[s]["path"], "source", split_of[s], s, -1, None, False, False))
    if a.real_distil:                                       # real photographs: distillation input only; treated label unknown -> masked
        for r in eligible:
            if r["dataset_id"] == "commoncatalog_ccby" and r["edit_type"] == "original" and r["image_id"] not in split_of and r["split"] == "train":
                rows.append((r["path"], "real", "train", r["image_id"], -1, None, False, False))
    gen = np.array([r[1] for r in rows]); split = np.array([r[2] for r in rows]); src = np.array([r[3] for r in rows])
    fam_i = np.array([r[4] for r in rows]); treated = np.array([r[7] for r in rows])
    Y = np.array([r[5] if r[5] else [0.0] * len(TARGETS) for r in rows], dtype=np.float32)
    M = np.array([[r[5] is not None] * len(TARGETS) for r in rows])
    mono = np.array([r[6] for r in rows])
    M[mono, 0] = M[mono, 1] = False
    kd_only = [g for g in a.kd_only_generators.split(",") if g]
    train_mask = (split == "train") & np.isin(gen, [*gens, "source"] + (["real"] if a.real_distil else []) + kd_only)
    treated_known = (gen != "real") & ~np.isin(gen, kd_only)
    teacher_layers = tuple(int(x) for x in a.teacher_layers.split(","))
    rep = {"tag": a.tag, "train_generators": gens, "synth": a.synth, "n_train": int(train_mask.sum()), "real_distil": a.real_distil, "kd_only_generators": kd_only,
           "teacher_layers": list(teacher_layers),
           "train_by_generator": dict(Counter(gen[train_mask].tolist())), "families": fams}

    # teacher: hybrid_style -> SupCon projection (filmsynth train sources only)
    t0 = time.time()
    paths = [r[0] for r in rows]
    with Pool(16) as pool:
        ex = np.stack(pool.starmap(_explicit_any, [(p, g in ("source", "real")) for p, g in zip(paths, gen)], chunksize=8))
    ims = [Image.open(p).convert("RGB") if g not in ("source", "real") else X.to_short(load_image(p, max_side=1280)) for p, g in zip(paths, gen)]
    fs_tr = (gen == "filmsynth") & (split == "train")
    fam_names = np.array([fams[i] if i >= 0 else "none" for i in fam_i])
    if a.kd <= 0:                                            # E46: no teacher at all (the DINOv2 pass is skipped; the head trains on nothing)
        T = np.zeros((len(rows), 64), np.float32); a.kd_target = "none"
    else:
        dino = X.Dino(dev, layers=teacher_layers)
        dg, ds = dino(ims)
        hyb = np.hstack([ex, ds])
    if a.kd <= 0:
        pass
    elif a.kd_target == "supcon":
        T = X.supcon(hyb[fs_tr], fam_names[fs_tr], dev)(hyb)
    elif a.kd_target == "supcon_vic3":
        from head_anticollapse import train_head
        T = train_head(hyb[fs_tr], fam_names[fs_tr], dev, tau=0.1, lam_vic=3.0, seed=a.seed)(hyb)
    else:                                                    # label-free: whitening fitted on the real distillation inputs (or the synthetic train sources without them)
        from frozen_head_choice import PCAW
        fit_rows = (gen == "real") if (gen == "real").any() else fs_tr
        T = PCAW(64).fit(ds[fit_rows])(ds)
    rep["kd_target"] = a.kd_target
    rep["teacher_seconds"] = round(time.time() - t0, 1)

    # student data
    with Pool(16) as pool:
        Xu8 = np.stack(pool.map(_sq, paths, chunksize=8))
    Xe = np.concatenate([preprocess(p) for p in paths])         # runtime geometry for evaluation
    net = StyleStudent(len(fams)).to(dev)
    opt = torch.optim.AdamW([{"params": net.features.parameters(), "lr": 3e-4},
                             {"params": [p for n, p in net.named_parameters() if not n.startswith("features")], "lr": 1e-3}],
                            weight_decay=1e-4)
    tr = np.where(train_mask)[0]
    steps = a.epochs * int(np.ceil(len(tr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps)
    Tt = torch.tensor(T, dtype=torch.float32, device=dev)
    yt = torch.tensor(treated.astype(np.float32), device=dev); yf = torch.tensor(fam_i, device=dev); tk = torch.tensor(treated_known, device=dev)
    Yt, Mt = torch.tensor(Y, device=dev), torch.tensor(M, device=dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hist = []
    vic_src = None
    if a.vic_mode == "source":                                   # E51: synthetic sources with at least four training rows (the source itself and its treatments)
        from collections import defaultdict
        grp = defaultdict(list)
        for i in tr:
            if gen[i] in ("filmsynth", "source"):
                grp[src[i]].append(i)
        vic_src = [np.array(v) for v in grp.values() if len(v) >= 4]
        rep["vic_source_groups"] = len(vic_src)
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]
            if vic_src is not None:                              # E51: 12 synthetic sources x 4 of their rows lead the batch; 16 random rows follow
                bi = np.concatenate([*(rng.choice(vic_src[g], 4, replace=False) for g in rng.choice(len(vic_src), 12, replace=False)), perm[b:b + 16]])
            ox, oy = rng.integers(0, 33, 2)
            xb = Xu8[bi][:, oy:oy + 224, ox:ox + 224]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            x = S.to_tensor(np.ascontiguousarray(xb), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f, lt, lf, lp, kd = net(x)
            lt, lf, lp, kd = lt.float(), lf.float(), lp.float(), kd.float()
            bi_t = torch.as_tensor(bi, device=dev)
            km = tk[bi_t]
            loss = nn.functional.binary_cross_entropy_with_logits(lt[km], yt[bi_t][km]) if km.any() else lt.sum() * 0
            fm = yf[bi_t] >= 0
            if fm.any():
                loss = loss + nn.functional.cross_entropy(lf[fm], yf[bi_t][fm])
            pm = Mt[bi_t]
            if pm.any():
                loss = loss + ((lp - Yt[bi_t]) ** 2 * pm).sum() / pm.sum()
            if a.kd > 0:
                loss = loss + a.kd * (1 - nn.functional.cosine_similarity(kd, Tt[bi_t], dim=1)).mean()
            if a.vic_pooled > 0 and len(bi) > 1:              # E46: VICReg on the pooled layer (variance hinge at 1, off-diagonal covariance)
                fc = f.float()
                if vic_src is not None:                          # E51: centre each row on its source's in-batch mean, so only treatment-induced variance is expanded
                    g = fc[:48].reshape(12, 4, -1); fc = (g - g.mean(1, keepdim=True)).reshape(48, -1)
                else:
                    fc = fc - fc.mean(0)
                std = torch.sqrt(fc.var(0) + 1e-4); C = (fc.T @ fc) / (len(fc) - 1)
                loss = loss + a.vic_pooled * (torch.relu(1 - std).mean() + (C - torch.diag(torch.diag(C))).pow(2).sum() / fc.shape[1])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(bi)
        hist.append(round(tot / len(tr), 4))
    rep["train"] = {"seconds": round(time.time() - t0, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "loss_curve": hist, "params": int(sum(p.numel() for p in net.parameters())), "seed": a.seed,
                    "epochs": a.epochs, "kd_weight": a.kd, "vic_pooled": a.vic_pooled, "vic_mode": a.vic_mode, "gpu": torch.cuda.get_device_name(0) + " (dev workstation)"}

    net.eval()

    @torch.no_grad()
    def run(xs):
        out = {"f": [], "t": [], "fam": [], "kd": []}
        for i in range(0, len(xs), 128):
            f, lt, lf, lp, kd = net(torch.from_numpy(xs[i:i + 128]).to(dev))
            out["f"].append(f.cpu().numpy()); out["t"].append(torch.sigmoid(lt).cpu().numpy())
            out["fam"].append(lf.cpu().numpy()); out["kd"].append(kd.cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}
    O = run(Xe)
    te = split == "test"
    from sklearn.metrics import f1_score
    fs_te = te & (gen == "filmsynth")
    rep["synthetic_test"] = {"family_macro_f1": round(float(f1_score(fam_i[fs_te], O["fam"][fs_te].argmax(1), average="macro")), 4),
                             "treated_auroc_filmsynth": auroc(treated[fs_te], O["t"][fs_te])}
    sx_te, so_te = te & (gen == "sdxl"), te & (gen == "source")
    rep["logo_sdxl_test"] = {"treated_auroc_sdxl_vs_sources": auroc(np.r_[np.ones(sx_te.sum()), np.zeros(so_te.sum())], np.r_[O["t"][sx_te], O["t"][so_te]]),
                             "seen_in_training": "sdxl" in gens, "mean_p_treated": {"sdxl": round(float(O["t"][sx_te].mean()), 3), "sources": round(float(O["t"][so_te].mean()), 3)}}

    # content invariance: own-source rank-1 among test sources with the student embedding
    def rank1(F, mask_q, mask_bank):
        Zq = F[mask_q] / (np.linalg.norm(F[mask_q], axis=1, keepdims=True) + 1e-8)
        Zb = F[mask_bank] / (np.linalg.norm(F[mask_bank], axis=1, keepdims=True) + 1e-8)
        S_ = Zq @ Zb.T; bs = src[mask_bank]
        return round(float(np.mean([bs[S_[i].argmax()] == s for i, s in enumerate(src[mask_q])])), 4)
    rep["content_invariance_test"] = {"own_source_rank1_576d": rank1(O["f"], fs_te & treated, so_te),
                                      "own_source_rank1_kd64": rank1(O["kd"], fs_te & treated, so_te),
                                      "reading": "1.0 = embedding retrieves the same photo regardless of treatment (content-dominated)"}

    # real film: BlueNeg vs CommonCatalog digital (final-only; skipped on selection runs)
    bn = [] if a.no_blueneg_eval else read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    used = set(srcs)
    dig = [r for r in eligible if r["image_id"] not in used and r.get("medium") in ("digital_camera", "smartphone") and r["split"] in ("val", "test")]
    Xd = np.concatenate([preprocess(r["path"]) for r in dig]); Od = run(Xd)
    if bn:
        Xb = np.concatenate([preprocess(r["path"]) for r in bn]); Ob = run(Xb)
        intact = np.array([r["partition"] == "blue-intact" for r in bn])
        real = {"film_vs_digital_auroc_p_treated": {"all": auroc(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[Ob["t"], Od["t"]]),
                                                     "blue_intact": auroc(np.r_[np.ones(int(intact.sum())), np.zeros(len(dig))], np.r_[Ob["t"][intact], Od["t"]]),
                                                     "n_blueneg": len(bn), "n_digital": len(dig)}}
    else:
        Xb, Ob, real = Xd[:0], None, {"skipped": "selection run (--no-blueneg-eval); BlueNeg is final-only"}
    stock = np.array([r["film_stock_verified"] for r in bn]); roll = np.array([r["roll_id"] for r in bn])
    for name, F in (() if a.no_blueneg_eval else (("576d", Ob["f"]), ("kd64", Ob["kd"]))):
        idx = np.where(intact)[0]
        Z = F[idx] / (np.linalg.norm(F[idx], axis=1, keepdims=True) + 1e-8); Sm = Z @ Z.T
        prec, ch = [], []
        for q in range(len(idx)):
            ok = roll[idx] != roll[idx][q]
            if (stock[idx][ok] == stock[idx][q]).sum() == 0:
                continue
            order = [j for j in np.argsort(-Sm[q]) if ok[j]][:5]
            prec.append(np.mean(stock[idx][order] == stock[idx][q])); ch.append((stock[idx][ok] == stock[idx][q]).mean())
        real[f"same_stock_precision@5_{name}"] = {"value": round(float(np.mean(prec)), 4), "chance": round(float(np.mean(ch)), 4)}
    # existing supervised student baseline: P(film_scan) from its capture_type head
    try:
        import onnxruntime as ort
        meta = json.loads((P.CHECKPOINTS / "student_sup" / "meta.json").read_text())
        sess = ort.InferenceSession(str(P.CHECKPOINTS / "student_sup" / "model.onnx"), providers=["CPUExecutionProvider"])
        ci = meta["outputs"].index("capture_type"); fi = meta["heads"]["capture_type"].index("film_scan")
        def pfilm(xs):
            out = []
            for i in range(0, len(xs), 64):
                lg = sess.run(None, {"pixel_values": xs[i:i + 64]})[ci]
                e = np.exp(lg - lg.max(1, keepdims=True)); out.append((e / e.sum(1, keepdims=True))[:, fi])
            return np.concatenate(out)
        if bn:
            real["baseline_existing_student_p_film_scan_auroc"] = {"all": auroc(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pfilm(Xb), pfilm(Xd)]),
                                                                    "note": "checkpoints/student_sup trained on EXIF capture type of CommonCatalog originals"}
    except Exception as e:  # noqa: BLE001
        real["baseline_existing_student_p_film_scan_auroc"] = f"unavailable: {e}"
    rep["real_film_blueneg"] = real

    # robustness on SDXL test outputs
    import io
    def perturb(im, kind):
        if kind == "jpeg_q55":
            b = io.BytesIO(); im.save(b, "JPEG", quality=55); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        if kind == "rescale_0.5":
            return im.resize((im.width // 2, im.height // 2), Image.BILINEAR).resize(im.size, Image.BILINEAR)
        return Image.fromarray(np.clip(np.asarray(im, dtype=np.float32) * 2 ** 0.5, 0, 255).astype(np.uint8))
    sx_idx = np.where(sx_te)[0]
    base = O["t"][sx_idx] > 0.5
    rob = {}
    for kind in ("jpeg_q55", "rescale_0.5", "exposure_+0.5ev"):
        xs = np.concatenate([preprocess(perturb(ims[i], kind)) for i in sx_idx])
        p = run(xs)["t"] > 0.5
        rob[kind] = {"still_treated": round(float(p.mean()), 4), "unchanged": round(float((p == base).mean()), 4)}
    rep["robustness_sdxl_test"] = rob

    # export + CPU latency
    out = P.CHECKPOINTS / f"student_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    class W(nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): f, t, fam, p, kd = s.n(x); return t, fam, kd, f          # E42: the 576-d pooled features are exported too
    torch.onnx.export(W(net.cpu().eval()), (torch.zeros(1, 3, 224, 224),), str(out / "model.onnx"), input_names=["pixel_values"],
                      output_names=["treated_logit", "family_logits", "style_embedding", "features_576"], dynamic_axes={"pixel_values": {0: "batch"}},
                      opset_version=17, dynamo=False)
    (out / "meta.json").write_text(json.dumps({"model_version": f"filmlook-style-student-{a.tag}", "families": fams, "train_generators": gens, "teacher_layers": list(teacher_layers), "kd_target": a.kd_target, "kd_weight": a.kd, "vic_pooled": a.vic_pooled, "vic_mode": a.vic_mode,
                                               "outputs": ["treated_logit", "family_logits", "style_embedding", "features_576"], "input_size": 224,
                                               "label_sources": "synthetic parameter supervision + SupCon teacher distillation; no human labels"}, indent=1))
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out / "model.onnx"), so, providers=["CPUExecutionProvider"])
    x1 = Xe[:1]; sess.run(None, {"pixel_values": x1})
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); sess.run(None, {"pixel_values": x1}); ts.append(time.perf_counter() - t0)
    rep["cost"] = {"onnx_mb": round((out / "model.onnx").stat().st_size / 1e6, 2), "cpu_1thread_ms_p50": round(1000 * float(np.median(ts)), 1),
                   "cpu": "i9-14900KF dev workstation (not the E2 host)"}
    rep["total_seconds"] = round(time.time() - t_all, 1)
    (P.REPORTS / f"student_{a.tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("tag", "n_train", "synthetic_test", "logo_sdxl_test", "content_invariance_test", "real_film_blueneg", "robustness_sdxl_test", "cost")}, indent=1, default=str))


if __name__ == "__main__":
    main()
