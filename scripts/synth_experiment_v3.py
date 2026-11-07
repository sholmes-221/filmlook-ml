#!/usr/bin/env python
"""Generator-leakage benchmark v3: nuisance parity and grain randomisation, cell by cell.

  python scripts/synth_experiment_v3.py      # -> reports/synth_experiment_v3.json

Cells (each adds one fix to the previous):
  C0 baseline     filmsynth v1 (treated vs identity) <-> SDXL (treated vs RAW source), no standardisation  [reproduces v2]
  C1 +standardise every image (controls, BlueNeg, digital included) -> 384 px short side, bicubic, JPEG q80-95 (seeded), EXIF dropped
  C2 +VAE control SDXL untreated control = the source passed through the SDXL VAE encode/decode (vae_identity)
  C3 +grain rand. filmsynth v2: grain model and strength drawn independently of the treated/untreated label
Per cell and representation (hybrid_style = explicit statistics + DINOv2 intermediate; dino_global for content):
  leave-one-generator-out AUROC in both directions with a source-level bootstrap 95% CI, generator-identity
  3-way accuracy (chance stated), own-source rank-1 (content leakage), cross-generator treated retrieval p@5,
  real film: BlueNeg vs CommonCatalog digital AUROC of P(treated) from the filmsynth-trained probe, and same-stock
  retrieval p@5 among blue-intact previews (same roll excluded) with a SupCon projection trained on filmsynth.
Synthetic and real numbers are reported in separate blocks. All real-film numbers are proxies: BlueNeg is one
photographer's 1990s consumer film; EXIF/owner records are the only labels.
"""
import hashlib
import io
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment as X  # noqa: E402

from filmlook_infer.imageio import load_image  # noqa: E402
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402

SHORT = 384
N_BOOT = 300


def standardize(im: Image.Image, key: str) -> Image.Image:
    """Shared post-pipeline: 384 px short side (bicubic) -> JPEG q80-95 (seeded by key) -> decode. No metadata."""
    im = X.to_short(im, SHORT)
    q = 80 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "little") % 16
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=0)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _load(args):
    path, key, std, real = args
    im = X.to_short(load_image(path, max_side=1280)) if real else Image.open(path).convert("RGB")
    if std:
        im = standardize(im, key)
    return np.asarray(im, dtype=np.uint8)


class Feats:
    def __init__(self, dev):
        self.dino = X.Dino(dev)

    def __call__(self, paths, keys, std, real=False):
        with Pool(16) as pool:
            arrs = pool.map(_load, [(p, k, std, real) for p, k in zip(paths, keys)], chunksize=8)
        ims = [Image.fromarray(a) for a in arrs]
        ex = np.stack([X.explicit_of(im) for im in ims])
        dg, ds = self.dino(ims)
        return {"hybrid_style": np.hstack([ex, ds]), "dino_global": dg, "explicit": ex}


def probe_auroc(Xtr, ytr, Xte, yte, src_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    auc = roc_auc_score(yte, p)
    rng = np.random.default_rng(0)
    us = np.unique(src_te)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.where(src_te == s)[0] for s in pick])
        if len(set(yte[idx])) > 1:
            boots.append(roc_auc_score(yte[idx], p[idx]))
    return {"auroc": round(float(auc), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "n_test": int(len(yte))}, (sc, m)


def gen_id(Xs, ys, split, n_cls):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr, te = split == "train", split == "test"
    sc = StandardScaler().fit(Xs[tr])
    m = LogisticRegression(max_iter=4000, C=0.1).fit(sc.transform(Xs[tr]), ys[tr])
    pred = m.predict(sc.transform(Xs[te]))
    return {"accuracy": round(float((pred == ys[te]).mean()), 4), "chance": round(float(max(Counter(ys[te]).values()) / te.sum()), 4)}


def rank1(F, src_q, Fo, src_o):
    Zq = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8); Zo = Fo / (np.linalg.norm(Fo, axis=1, keepdims=True) + 1e-8)
    S = Zq @ Zo.T
    return round(float(np.mean([src_o[S[i].argmax()] == s for i, s in enumerate(src_q)])), 4)


def treated_retrieval(Fq, src_q, Fb, src_b, treated_b):
    Zq = Fq / (np.linalg.norm(Fq, axis=1, keepdims=True) + 1e-8); Zb = Fb / (np.linalg.norm(Fb, axis=1, keepdims=True) + 1e-8)
    S = Zq @ Zb.T
    prec = []
    for i in range(len(Fq)):
        order = [j for j in np.argsort(-S[i]) if src_b[j] != src_q[i]][:5]
        prec.append(treated_b[order].mean())
    return {"treated_precision@5_other_sources": round(float(np.mean(prec)), 4), "chance": round(float(treated_b.mean()), 4)}


def same_stock(F, stock, roll):
    Z = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8); S = Z @ Z.T
    prec, ch = [], []
    for q in range(len(F)):
        ok = roll != roll[q]
        if (stock[ok] == stock[q]).sum() == 0:
            continue
        order = [j for j in np.argsort(-S[q]) if ok[j]][:5]
        prec.append(np.mean(stock[order] == stock[q])); ch.append((stock[ok] == stock[q]).mean())
    return {"precision@5": round(float(np.mean(prec)), 4), "chance": round(float(np.mean(ch)), 4), "n": len(prec)}


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    v1 = read_jsonl(P.DATA / "synthetic" / "film_synth_v1" / "annotations.jsonl")
    v2 = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in v1} & {a["source_image_id"] for a in v2} & {a["source_image_id"] for a in sdx})
    sp = {s: by_id[s]["split"] for s in srcs}
    v1 = [a for a in v1 if a["source_image_id"] in sp]; v2 = [a for a in v2 if a["source_image_id"] in sp]
    sdx_t = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True)]
    sdx_c = [a for a in sdx if a["source_image_id"] in sp and a.get("treated", True) is False]
    bn = read_jsonl(P.MANIFESTS / "blueneg_manifest.jsonl")
    dig = [r for r in eligible if r["image_id"] not in set(srcs) and r.get("medium") in ("digital_camera", "smartphone") and r["split"] in ("val", "test")]
    rep = {"sources": len(srcs), "splits": dict(Counter(sp.values())), "n": {"filmsynth_v1": len(v1), "filmsynth_v2": len(v2),
           "sdxl_treated": len(sdx_t), "sdxl_vae_control": len(sdx_c), "blueneg": len(bn), "digital": len(dig)}}
    assert len(sdx_c) == len(srcs), "vae_identity control missing for some sources"
    feats = Feats(dev)

    def fset(rows, std, real=False, pathkey="path", idkey="variant_id"):
        paths = [(r[pathkey] if real else str(P.ROOT / r[pathkey])) for r in rows]
        keys = [r[idkey] for r in rows]
        return feats(paths, keys, std, real)

    cells = {}
    cache = {}
    for cell, std, control, fsv in (("C0_baseline", False, "raw", "v1"), ("C1_standardised", True, "raw", "v1"),
                                    ("C2_vae_control", True, "vae", "v1"), ("C3_grain_randomised", True, "vae", "v2")):
        fs = v1 if fsv == "v1" else v2
        ident = "identity_v1" if fsv == "v1" else "identity_v2"
        k = (fsv, std)
        if k not in cache:
            cache[k] = fset(fs, std)
        Fs = cache[k]
        if ("sdx", std) not in cache:
            cache[("sdx", std)] = fset(sdx_t, std)
            cache[("src", std)] = feats([by_id[s]["path"] for s in srcs], srcs, std, real=True)
            cache[("vae", std)] = fset(sdx_c, std)
            cache[("bn", std)] = feats([r["path"] for r in bn], [r["image_id"] for r in bn], std, real=True)
            cache[("dig", std)] = feats([r["path"] for r in dig], [r["image_id"] for r in dig], std, real=True)
        Fx, Fo, Fv, Fb, Fd = (cache[(n, std)] for n in ("sdx", "src", "vae", "bn", "dig"))
        Fc = Fv if control == "vae" else Fo
        src_s = np.array([a["source_image_id"] for a in fs]); sp_s = np.array([sp[s] for s in src_s])
        tr_s = np.array([a["augmentation_family"] != ident for a in fs]).astype(int)
        src_x = np.array([a["source_image_id"] for a in sdx_t]); sp_x = np.array([sp[s] for s in src_x])
        src_c = np.array([a["source_image_id"] for a in sdx_c]) if control == "vae" else np.array(srcs)
        sp_c = np.array([sp[s] for s in src_c])
        out = {"standardised": std, "sdxl_control": control, "filmsynth": fsv}
        for name in ("hybrid_style", "explicit"):
            # LOGO filmsynth -> sdxl
            Xtr, ytr = Fs[name][sp_s == "train"], tr_s[sp_s == "train"]
            Xte = np.vstack([Fx[name][sp_x == "test"], Fc[name][sp_c == "test"]])
            yte = np.r_[np.ones((sp_x == "test").sum()), np.zeros((sp_c == "test").sum())].astype(int)
            ste = np.r_[src_x[sp_x == "test"], src_c[sp_c == "test"]]
            a, fitted = probe_auroc(Xtr, ytr, Xte, yte, ste)
            # LOGO sdxl -> filmsynth
            Xtr2 = np.vstack([Fx[name][sp_x == "train"], Fc[name][sp_c == "train"]])
            ytr2 = np.r_[np.ones((sp_x == "train").sum()), np.zeros((sp_c == "train").sum())].astype(int)
            b, _ = probe_auroc(Xtr2, ytr2, Fs[name][sp_s == "test"], tr_s[sp_s == "test"], src_s[sp_s == "test"])
            # generator identity: filmsynth-treated / sdxl-treated / controls (filmsynth identity + sdxl control)
            Xg = np.vstack([Fs[name][tr_s == 1], Fx[name], Fs[name][tr_s == 0], Fc[name]])
            yg = np.r_[np.zeros((tr_s == 1).sum()), np.ones(len(sdx_t)), 2 * np.ones((tr_s == 0).sum()), 2 * np.ones(len(src_c))].astype(int)
            spg = np.r_[sp_s[tr_s == 1], sp_x, sp_s[tr_s == 0], sp_c]
            g = gen_id(Xg, yg, spg, 3)
            # real film with the filmsynth-trained probe
            from sklearn.metrics import roc_auc_score
            sc, m = fitted
            pb = m.predict_proba(sc.transform(Fb[name]))[:, 1]; pd = m.predict_proba(sc.transform(Fd[name]))[:, 1]
            intact = np.array([r["partition"] == "blue-intact" for r in bn])
            real = {"blueneg_vs_digital_auroc": round(float(roc_auc_score(np.r_[np.ones(len(bn)), np.zeros(len(dig))], np.r_[pb, pd])), 4),
                    "blue_intact_vs_digital_auroc": round(float(roc_auc_score(np.r_[np.ones(intact.sum()), np.zeros(len(dig))], np.r_[pb[intact], pd])), 4),
                    "mean_p_treated": {"blueneg": round(float(pb.mean()), 3), "digital": round(float(pd.mean()), 3)}}
            out[name] = {"logo_filmsynth_to_sdxl": a, "logo_sdxl_to_filmsynth": b, "generator_identity_3way": g, "real_film": real}
        # content leakage and retrieval (hybrid_style + dino_global), SupCon on filmsynth train
        te_x = sp_x == "test"
        out["content_own_source_rank1"] = {"sdxl_treated_dino_global": rank1(Fx["dino_global"][te_x], src_x[te_x], Fo["dino_global"], np.array(srcs)),
                                           "sdxl_treated_hybrid_style": rank1(Fx["hybrid_style"][te_x], src_x[te_x], Fo["hybrid_style"], np.array(srcs))}
        fam = np.array([a["augmentation_family"] for a in fs])
        proj = X.supcon(Fs["hybrid_style"][sp_s == "train"], fam[sp_s == "train"], dev)
        Ps, Px, Pc = proj(Fs["hybrid_style"]), proj(Fx["hybrid_style"]), proj(Fc["hybrid_style"])
        bank = np.vstack([Ps, Pc]); bsrc = np.r_[src_s, src_c]; btr = np.r_[tr_s, np.zeros(len(src_c))].astype(bool)
        out["cross_generator_retrieval_sdxl_query"] = {"hybrid_style": treated_retrieval(Fx["hybrid_style"][te_x], src_x[te_x], np.vstack([Fs["hybrid_style"], Fc["hybrid_style"]]), bsrc, btr),
                                                       "supcon": treated_retrieval(Px[te_x], src_x[te_x], bank, bsrc, btr)}
        intact = np.array([r["partition"] == "blue-intact" for r in bn])
        stock = np.array([r["film_stock_verified"] for r in bn])[intact]; roll = np.array([r["roll_id"] for r in bn])[intact]
        out["blueneg_same_stock"] = {"hybrid_style": same_stock(Fb["hybrid_style"][intact], stock, roll),
                                     "supcon_filmsynth": same_stock(proj(Fb["hybrid_style"])[intact], stock, roll)}
        cells[cell] = out
        print(cell, json.dumps({k: out[k] for k in ("hybrid_style", "content_own_source_rank1")}, default=str)[:600], file=sys.stderr)
    rep["cells"] = cells
    rep["summary"] = {c: {"logo_fs_to_sdxl": o["hybrid_style"]["logo_filmsynth_to_sdxl"]["auroc"], "logo_sdxl_to_fs": o["hybrid_style"]["logo_sdxl_to_filmsynth"]["auroc"],
                          "gen_id_acc": o["hybrid_style"]["generator_identity_3way"]["accuracy"], "blueneg_auroc": o["hybrid_style"]["real_film"]["blueneg_vs_digital_auroc"],
                          "same_stock_supcon": o["blueneg_same_stock"]["supcon_filmsynth"]["precision@5"]} for c, o in cells.items()}
    rep["total_seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / "synth_experiment_v3.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"n": rep["n"], "summary": rep["summary"], "seconds": rep["total_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
