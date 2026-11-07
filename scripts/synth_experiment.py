#!/usr/bin/env python
"""Controlled synthetic film-aesthetic experiment (film-synth-v1, 100 sources x 8 variants).

  python scripts/synth_experiment.py            # -> reports/synth_experiment.json

Representations (all images at a 384 px short side, then the same downstream geometry):
  explicit       28 deterministic colour, tone and texture statistics (filmlook_infer.features)
  dino_global    frozen DINOv2-small CLS ++ mean patch (768)
  dino_style     frozen DINOv2-small patch-token mean ++ std from layers 4 and 8 (1536): a style-sensitive descriptor
  hybrid         explicit ++ dino_global
  hybrid_style   explicit ++ dino_style
Experiments: A family classification (logistic), B parameter regression (ridge), C contrastive projection on
hybrid_style (positives = same family, other sources), D fine-tuned MobileNetV3-Small (end-to-end).
Shortcut checks: nuisance -> family, same-source retrieval, clipping per family, robustness to recompression,
rescaling and exposure. Transfer: the dataset's independently implemented edits (datasets/edits.py) and REAL
photographs (EXIF capture medium, measured statistics, uploader film-stock tags). Synthetic and real results are
reported separately.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

from filmlook_infer.features import compute_features, feature_vector
from filmlook_infer.imageio import load_image
from filmlook_infer.pixels import dinov2_pixels
from filmlook_ml import paths as P
from filmlook_ml.annotation.workflow import discover, film_stock_tags, load_uploader_tags
from filmlook_ml.compliance.preflight import require_training_preflight
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry

SYN = P.DATA / "synthetic" / "film_synth_v1"
TARGETS = ["temperature_shift", "saturation_adjustment", "contrast_adjustment", "grain_strength", "shadow_lift",
           "highlight_compression"]
MONO_FAMS = {"high_contrast_bw_v1", "finegrain_bw_v1", "grainy_highspeed_mono_v1"}
SHORT = 384


def to_short(img: Image.Image, short=SHORT) -> Image.Image:
    s = short / min(img.size)
    return img.convert("RGB").resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)


def explicit_of(img):
    return feature_vector(compute_features(img))


def _explicit_path(p):
    return explicit_of(Image.open(p).convert("RGB"))


def _explicit_real(p):
    return explicit_of(to_short(load_image(p, max_side=1280)))


class Dino:
    def __init__(self, dev):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, imgs: list[Image.Image]):
        g, s = [], []
        for i in range(0, len(imgs), 64):
            px = torch.from_numpy(np.stack([dinov2_pixels(im, 224) for im in imgs[i:i + 64]])).to(self.dev)
            out = self.m(pixel_values=px, output_hidden_states=True)
            h = out.last_hidden_state
            g.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
            st = [torch.cat([hs[:, 1:].mean(1), hs[:, 1:].std(1)], 1) for hs in (out.hidden_states[4], out.hidden_states[8])]
            s.append(torch.cat(st, 1).float().cpu().numpy())
        return np.concatenate(g), np.concatenate(s)


def reps_from(explicit, dg, ds):
    return {"explicit": explicit, "dino_global": dg, "dino_style": ds,
            "hybrid": np.hstack([explicit, dg]), "hybrid_style": np.hstack([explicit, ds])}


# ----------------------------------------------------------------------------- models
def fit_cls(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=4000, C=C).fit(sc.transform(Xtr), ytr)
        f = f1_score(yva, m.predict(sc.transform(Xva)), average="macro")
        if best is None or f > best[0]:
            best = (f, C, sc, m)
    return best


def fit_reg(Xtr, Ytr, Mtr, Xva, Yva, Mva):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    models = {}
    for j, t in enumerate(TARGETS):
        best = None
        for a in (0.1, 1, 10, 100, 1000):
            m = Ridge(alpha=a).fit(sc.transform(Xtr[Mtr[:, j]]), Ytr[Mtr[:, j], j])
            e = np.abs(m.predict(sc.transform(Xva[Mva[:, j]])) - Yva[Mva[:, j], j]).mean()
            if best is None or e < best[0]:
                best = (e, a, m)
        models[t] = best[2]
    return sc, models


def cls_report(m, sc, X, y, classes):
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    pred = m.predict(sc.transform(X))
    _, _, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return {"accuracy": round(float((pred == y).mean()), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)), 4),
            "per_class_f1": {c: round(float(f1[i]), 3) for i, c in enumerate(classes)},
            "confusion": confusion_matrix(y, pred, labels=classes).tolist()}


def retrieval(Z, fam, src, idx, k=5):
    Zn = Z[idx] / (np.linalg.norm(Z[idx], axis=1, keepdims=True) + 1e-8)
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    f, s = fam[idx], src[idx]
    same_src_top1, prec = [], []
    for i in range(len(idx)):
        order = np.argsort(-S[i])
        same_src_top1.append(s[order[0]] == s[i])
        other = [j for j in order if s[j] != s[i]][:k]
        prec.append(np.mean(f[other] == f[i]))
    chance = np.mean([(np.sum((f == f[i]) & (s != s[i]))) / np.sum(s != s[i]) for i in range(len(idx))])
    return {f"family_precision@{k}_other_sources": round(float(np.mean(prec)), 4),
            "chance": round(float(chance), 4), "same_source_top1_rate": round(float(np.mean(same_src_top1)), 4)}


def supcon(Xtr, ftr, dev, dim=64, epochs=400, tau=0.1, seed=0):
    torch.manual_seed(seed)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    X = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    lab = torch.tensor(np.unique(ftr, return_inverse=True)[1], device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, dim)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = (lab[:, None] == lab[None]).float()
    pos.fill_diagonal_(0)
    eye = torch.eye(len(X), device=dev, dtype=torch.bool)
    for _ in range(epochs):
        z = torch.nn.functional.normalize(net(X + 0.05 * torch.randn_like(X)), dim=1)
        logits = (z @ z.T / tau).masked_fill(eye, -1e9)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -(pos * logp).sum(1) / pos.sum(1).clamp(min=1)
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
    net.eval()
    return lambda Xn: net(torch.tensor(sc.transform(Xn), dtype=torch.float32, device=dev)).detach().cpu().numpy()


def finetune_mnv3(paths, fam, Y, M, split, dev, classes, epochs=30, seed=0):
    import torchvision
    from filmlook_infer.compact import preprocess
    torch.manual_seed(seed)
    X = np.concatenate([preprocess(p) for p in paths])
    m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    m.classifier = torch.nn.Identity()
    net = torch.nn.ModuleDict({"b": m, "c": torch.nn.Linear(576, len(classes)), "r": torch.nn.Linear(576, len(TARGETS))}).to(dev)
    yi = torch.tensor([classes.index(f) for f in fam], device=dev)
    Yt, Mt = torch.tensor(Y, dtype=torch.float32, device=dev), torch.tensor(M, device=dev)
    tr = np.where(split == "train")[0]
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    t0 = time.time()
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            x = torch.from_numpy(X[b]).to(dev)
            if rng.random() < 0.5:
                x = x.flip(3)
            f = net["b"](x)
            loss = torch.nn.functional.cross_entropy(net["c"](f), yi[b])
            r = net["r"](f)
            loss = loss + ((r - Yt[b]) ** 2 * Mt[b]).sum() / Mt[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append(round(float(loss), 4))
    train_s = time.time() - t0
    net.eval()
    with torch.no_grad():
        F = torch.cat([net["b"](torch.from_numpy(X[i:i + 128]).to(dev)) for i in range(0, len(X), 128)])
        pred = net["c"](F).argmax(1).cpu().numpy()
        reg = net["r"](F).cpu().numpy()
    gpu = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None
    return pred, reg, F.cpu().numpy(), {"train_seconds": round(train_s, 1), "peak_gpu_gb": gpu, "loss_curve": hist}


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    return round(float(roc_auc_score(label, score)), 4) if 0 < label.sum() < len(label) else None


def spearman(a, b):
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b).statistic), 4)


def main():
    require_training_preflight()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_all = time.time()
    anns = read_jsonl(SYN / "annotations.jsonl")
    paths = [str(P.ROOT / a["path"]) for a in anns]
    fam = np.array([a["augmentation_family"] for a in anns])
    src = np.array([a["source_image_id"] for a in anns])
    split = np.array([a["split"] for a in anns])
    classes = sorted(set(fam))
    Y = np.array([[float(a["parameters"][t]) for t in TARGETS] for a in anns])
    M = np.ones_like(Y, dtype=bool)
    mono = np.isin(fam, list(MONO_FAMS))
    M[mono, 0] = M[mono, 1] = False                 # temperature/saturation undefined after the monochrome mix
    rep = {"version": anns[0]["augmentation_version"], "n_variants": len(anns), "classes": classes,
           "splits_variants": dict(Counter(split)), "splits_sources": {s: len(set(src[split == s])) for s in ("train", "val", "test")}}
    assert not (set(src[split == "train"]) & set(src[split == "test"])), "source leakage across splits"

    # features
    t0 = time.time()
    with Pool(16) as pool:
        ex = np.stack(pool.map(_explicit_path, paths, chunksize=8))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    dino = Dino(dev)
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    dg, ds = dino(imgs)
    rep["feature_extraction"] = {"seconds": round(time.time() - t0, 1),
                                 "dino_peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None}
    R = reps_from(ex, dg, ds)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    # A: family classification; B: regression
    rep["A_family_classification"], rep["B_parameter_regression"], fitted = {}, {}, {}
    for name, X in R.items():
        f, C, sc, m = fit_cls(X[tr], fam[tr], X[va], fam[va])
        fitted[name] = (sc, m)
        rep["A_family_classification"][name] = {"C": C, **{s: cls_report(m, sc, X[mk], fam[mk], classes)
                                                           for s, mk in (("train", tr), ("val", va), ("test", te))}}
        rsc, rm = fit_reg(X[tr], Y[tr], M[tr], X[va], Y[va], M[va])
        res = {}
        for j, t in enumerate(TARGETS):
            mk = te & M[:, j]
            p = rm[t].predict(rsc.transform(X[mk]))
            base = np.abs(Y[mk, j] - Y[tr & M[:, j], j].mean()).mean()
            res[t] = {"test_mae": round(float(np.abs(p - Y[mk, j]).mean()), 4), "mean_baseline_mae": round(float(base), 4),
                      "r2": round(float(1 - ((p - Y[mk, j]) ** 2).sum() / ((Y[mk, j] - Y[mk, j].mean()) ** 2).sum()), 4)}
        rep["B_parameter_regression"][name] = res
        fitted[name + "_reg"] = (rsc, rm)
    rep["A_summary_test_macro_f1"] = {n: r["test"]["macro_f1"] for n, r in rep["A_family_classification"].items()}
    rep["B_summary_test_mae"] = {n: {t: v["test_mae"] for t, v in r.items()} for n, r in rep["B_parameter_regression"].items()}

    # C: contrastive projection + style sensitivity / source memorisation (test sources only)
    proj = supcon(R["hybrid_style"][tr], fam[tr], dev)
    Zp = proj(R["hybrid_style"])
    te_idx = np.where(te)[0]
    rep["C_retrieval_test"] = {**{n: retrieval(X, fam, src, te_idx) for n, X in R.items()},
                               "supcon_hybrid_style": retrieval(Zp, fam, src, te_idx)}

    # D: fine-tuned MobileNetV3-Small
    pred, reg, Fm, info = finetune_mnv3(paths, fam, Y, M, split, dev, classes)
    from sklearn.metrics import f1_score
    rep["D_finetune_mnv3"] = {**info, **{s: {"accuracy": round(float((np.array(classes)[pred[mk]] == fam[mk]).mean()), 4),
                                             "macro_f1": round(float(f1_score(fam[mk], np.array(classes)[pred[mk]], average="macro")), 4)}
                                         for s, mk in (("train", tr), ("val", va), ("test", te))},
                              "test_mae": {t: round(float(np.abs(reg[te & M[:, j], j] - Y[te & M[:, j], j]).mean()), 4)
                                           for j, t in enumerate(TARGETS)},
                              "retrieval_test": retrieval(Fm, fam, src, te_idx)}

    # shortcut checks
    nz = np.array([[a["nuisance"]["jpeg_quality"], a["nuisance"]["exposure_ev"], a["nuisance"]["crop_fraction"],
                    ["bilinear", "bicubic", "lanczos"].index(a["nuisance"]["resample"])] for a in anns], dtype=float)
    f, C, sc, m = fit_cls(nz[tr], fam[tr], nz[va], fam[va])
    clip = defaultdict(list)
    for i, im in enumerate(imgs):
        a = np.asarray(im)
        clip[fam[i]].append(float(((a == 0) | (a == 255)).all(2).mean()))
    rep["shortcut_checks"] = {
        "family_from_nuisance_test_accuracy": round(float((m.predict(sc.transform(nz[te])) == fam[te]).mean()), 4),
        "chance": round(1 / len(classes), 4),
        "clipped_pixel_fraction_by_family": {k: round(float(np.mean(v)), 4) for k, v in sorted(clip.items())},
    }
    # robustness: perturb test variants (recompress q55, 0.5x down/up, +0.5 EV), re-extract, re-classify
    import io
    def perturb(im, kind):
        if kind == "jpeg_q55":
            b = io.BytesIO(); im.save(b, "JPEG", quality=55); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        if kind == "rescale_0.5":
            return im.resize((im.width // 2, im.height // 2), Image.BILINEAR).resize(im.size, Image.BILINEAR)
        a = np.clip(np.asarray(im, dtype=np.float32) * 2 ** 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(a)
    rob = {}
    for kind in ("jpeg_q55", "rescale_0.5", "exposure_+0.5ev"):
        pim = [perturb(imgs[i], kind) for i in te_idx]
        pex = np.stack([explicit_of(im) for im in pim])
        pdg, pds = dino(pim)
        PR = reps_from(pex, pdg, pds)
        rob[kind] = {}
        for name in ("explicit", "dino_global", "dino_style", "hybrid_style"):
            sc, m = fitted[name]
            p0 = m.predict(sc.transform(R[name][te_idx]))
            p1 = m.predict(sc.transform(PR[name]))
            rob[kind][name] = {"accuracy": round(float((p1 == fam[te_idx]).mean()), 4),
                               "unchanged_prediction": round(float((p1 == p0).mean()), 4)}
    rep["robustness_test"] = rob

    # ----------------------------------------------------------------------------- transfer
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    used = set(src)
    real = [r for r in eligible if r["image_id"] not in used and r["split"] in ("val", "test")]
    feats = json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text())
    with Pool(16) as pool:
        rex = np.stack(pool.map(_explicit_real, [r["path"] for r in real], chunksize=8))
    rim = [to_short(load_image(r["path"], max_side=1280)) for r in real]
    rdg, rds = dino(rim)
    RR = reps_from(rex, rdg, rds)
    transfer = {"n_real": len(real), "note": "real photographs never used as sources; no film-family labels exist for them"}
    medium = np.array([r.get("medium") or "pd12m" for r in real])
    measured = [feats[r["sha256"]] for r in real]
    is_mono = np.array([bool(f["is_mono"]) for f in measured])
    film = np.isin(medium, ["film_scan"])
    digital = np.isin(medium, ["digital_camera", "smartphone"])
    for name in ("explicit", "dino_global", "hybrid_style"):
        sc, m = fitted[name]
        Pm = m.predict_proba(sc.transform(RR[name]))
        cls = list(m.classes_)
        p_mono = Pm[:, [cls.index(c) for c in MONO_FAMS]].sum(1)
        p_identity = Pm[:, cls.index("identity_v1")]
        rsc, rm = fitted[name + "_reg"]
        pr = {t: rm[t].predict(rsc.transform(RR[name])) for t in TARGETS}
        fd = film | digital
        transfer[name] = {
            "monochrome_auroc_p_mono_family": auroc(p_mono, is_mono),
            "film_scan_vs_digital_auroc_1_minus_p_identity": auroc((1 - p_identity)[fd], film[fd]),
            "film_scan_vs_digital_auroc_pred_grain": auroc(pr["grain_strength"][fd], film[fd]),
            "spearman_pred_temperature_vs_measured_warmth": spearman(pr["temperature_shift"][~is_mono],
                                                                     [f["warmth"] for f, mo in zip(measured, is_mono) if not mo]),
            "spearman_pred_saturation_vs_measured_sat_mean": spearman(pr["saturation_adjustment"][~is_mono],
                                                                      [f["sat_mean"] for f, mo in zip(measured, is_mono) if not mo]),
            "spearman_pred_contrast_vs_measured_y_std": spearman(pr["contrast_adjustment"], [f["y_std"] for f in measured]),
            "spearman_pred_grain_vs_measured_noise_std": spearman(pr["grain_strength"], [f["noise_std"] for f in measured]),
            "predicted_family_distribution": dict(Counter(m.predict(sc.transform(RR[name])))),
        }
    # uploader film-stock tags (asserted metadata; never trained on)
    ut = load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl")
    tagged = [r for r in eligible if r["image_id"] not in used and r["dataset_id"] == "commoncatalog_ccby"
              and film_stock_tags(ut.get(str(r.get("source_id")), ""))]
    if tagged:
        tim = [to_short(load_image(r["path"], max_side=1280)) for r in tagged]
        tex = np.stack([explicit_of(i) for i in tim])
        tdg, tds = dino(tim)
        TR = reps_from(tex, tdg, tds)
        sc, m = fitted["hybrid_style"]
        pf = m.predict(sc.transform(TR["hybrid_style"]))
        transfer["uploader_stock_tags_hybrid_style"] = [
            {"tags": [x.value for x in film_stock_tags(ut.get(str(r.get("source_id")), ""))],
             "measured_mono": bool(feats[r["sha256"]]["is_mono"]), "predicted_family": p} for r, p in zip(tagged, pf)]
    # independently implemented edits (datasets/edits.py) on real photos not used as sources
    fams_real = {r["family_id"] for r in real}
    ed = [r for r in man if r["family_id"] in fams_real and r["edit_type"] != "original"]
    eim = [to_short(load_image(r["path"], max_side=1280)) for r in ed]
    eex = np.stack([explicit_of(i) for i in eim])
    edg, eds = dino(eim)
    ER = reps_from(eex, edg, eds)
    orig_idx = {r["family_id"]: i for i, r in enumerate(real)}
    cross = {}
    for name in ("explicit", "hybrid_style"):
        sc, m = fitted[name]
        pf = m.predict(sc.transform(ER[name]))
        rsc, rm = fitted[name + "_reg"]
        pe = {t: rm[t].predict(rsc.transform(ER[name])) for t in ("grain_strength", "contrast_adjustment", "saturation_adjustment")}
        po = {t: rm[t].predict(rsc.transform(RR[name])) for t in pe}
        by = defaultdict(list)
        for i, r in enumerate(ed):
            o = orig_idx[r["family_id"]]
            by[r["edit_type"]].append((pf[i], {t: pe[t][i] - po[t][o] for t in pe}))
        cross[name] = {k: {"predicted_family": dict(Counter(x[0] for x in v).most_common(3)),
                           "mean_delta_vs_original": {t: round(float(np.mean([x[1][t] for x in v])), 3) for t in pe},
                           "n": len(v)} for k, v in sorted(by.items())}
    transfer["independent_edit_pipeline"] = cross
    rep["real_world_transfer"] = transfer
    rep["total_seconds"] = round(time.time() - t_all, 1)
    rep["gpu"] = torch.cuda.get_device_name(0) + " (shared; NOT the owner's RTX 4070)" if dev.type == "cuda" else "cpu"
    (P.REPORTS / "synth_experiment.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("A_summary_test_macro_f1", "B_summary_test_mae", "C_retrieval_test",
                                          "shortcut_checks", "robustness_test", "total_seconds")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
