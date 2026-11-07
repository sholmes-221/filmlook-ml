#!/usr/bin/env python
"""Cross-generator study v2: filmsynth (programmatic) vs SDXL image-to-image, plus real-film transfer to BlueNeg.

  python scripts/synth_experiment_v2.py      # -> reports/synth_experiment_v2.json

Generators share the same 100 source photos and splits. Representations: explicit statistics, DINOv2 global,
DINOv2 intermediate statistics, hybrid_style (explicit + intermediate), and a SupCon projection of hybrid_style
trained on filmsynth train sources only.

Tests:
 1. generator identity leakage: can a linear probe tell filmsynth from SDXL from untreated on test sources?
 2. content preservation of SDXL img2img: DINOv2 cosine to the source and own-source rank-1 retrieval.
 3. leave-one-generator-out, task "treated vs untreated": train on one generator, test on the other.
 4. leave-one-generator-out, pairwise attributes (warmer / more contrast / grainier / more saturated): train on
    filmsynth parameter labels, test on SDXL pairs judged by measured-statistic differences, and the reverse
    direction trained on measured labels.
 5. cross-generator style retrieval: query an SDXL treatment, retrieve among filmsynth families (and vice versa);
    reported as precision of retrieving a "film-treated" image, since the two generators share no family taxonomy.
 6. real film (BlueNeg, owner-verified stock labels, one photographer, external set):
    a. film-vs-digital AUROC of the synthetic "treated" probability on BlueNeg previews vs CommonCatalog digital originals;
    b. same-stock retrieval among blue-intact previews with the same roll excluded (precision@5 vs chance),
       for each representation and for the SupCon projection.
 7. robustness of the treated-vs-untreated probe to JPEG q55 / 0.5x rescale / +0.5 EV on SDXL outputs.
Synthetic and real results are reported in separate sections.
"""
import io
import json
import sys
import time
from collections import Counter, defaultdict
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

SYN = P.DATA / "synthetic" / "film_synth_v1"
SDX = P.DATA / "synthetic" / "sdxl_i2i_v1"
STATS = {"warmer": "warmth", "more_contrast": "y_std", "grainier": "noise_std", "more_saturated": "sat_mean"}
PARAMS = {"warmer": "temperature_shift", "more_contrast": "contrast_adjustment", "grainier": "grain_strength",
          "more_saturated": "saturation_adjustment"}


def _ex(p):
    return X.explicit_of(Image.open(p).convert("RGB"))


def _ex_real(p):
    return X._explicit_real(p)


def probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return {"accuracy": round(float((m.predict(sc.transform(Xte)) == yte).mean()), 4),
            "auroc": round(float(roc_auc_score(yte, p)), 4) if len(set(yte)) > 1 else None, "n_test": int(len(yte))}, (sc, m)


def rank_probe(F, pairs, Fte, pairs_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    D = np.array([F[i] - F[j] for i, j, _ in pairs] + [F[j] - F[i] for i, j, _ in pairs])
    y = np.array([l for *_, l in pairs] + [1 - l for *_, l in pairs])
    sc = StandardScaler(with_mean=False).fit(D)
    m = LogisticRegression(max_iter=4000, C=0.1, fit_intercept=False).fit(sc.transform(D), y)
    if not pairs_te:
        return None
    acc = np.mean([(m.decision_function(sc.transform((Fte[i] - Fte[j])[None]))[0] > 0) == bool(l) for i, j, l in pairs_te])
    return round(float(acc), 4)


def measured_pairs(stats_a, stats_b, idx_pairs, key, mono_a=None, mono_b=None, scale=None, min_z=0.25):
    out = []
    for i, j in idx_pairs:
        if key in ("warmth", "sat_mean") and ((mono_a is not None and mono_a[i]) or (mono_b is not None and mono_b[j])):
            continue
        d = stats_a[i][key] - stats_b[j][key]
        if abs(d) / scale >= min_z:
            out.append((i, j, 1 if d > 0 else 0))
    return out


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    syn = read_jsonl(SYN / "annotations.jsonl")
    sdx = read_jsonl(SDX / "annotations.jsonl")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    srcs = sorted({a["source_image_id"] for a in syn} & {a["source_image_id"] for a in sdx})
    split_of = {s: by_id[s]["split"] for s in srcs}
    rep = {"sources_shared": len(srcs), "splits": dict(Counter(split_of.values()))}

    # ---------------------------------------------------------------- feature extraction (one pass per set)
    dino = X.Dino(dev)

    def featurize(paths, real=False):
        with Pool(16) as pool:
            ex = np.stack(pool.map(_ex_real if real else _ex, paths, chunksize=8))
        ims = [X.to_short(load_image(p, max_side=1280)) if real else Image.open(p).convert("RGB") for p in paths]
        dg, ds = dino(ims)
        return X.reps_from(ex, dg, ds), ims

    syn = [a for a in syn if a["source_image_id"] in split_of]
    sdx = [a for a in sdx if a["source_image_id"] in split_of]
    Rs, ims_s = featurize([str(P.ROOT / a["path"]) for a in syn])
    Rx, ims_x = featurize([str(P.ROOT / a["path"]) for a in sdx])
    Ro, ims_o = featurize([by_id[s]["path"] for s in srcs], real=True)        # untreated sources, same geometry
    from filmlook_infer.features import compute_features
    st_s = [compute_features(im) for im in ims_s]
    st_x = [compute_features(im) for im in ims_x]
    st_o = [compute_features(im) for im in ims_o]
    sp_s = np.array([split_of[a["source_image_id"]] for a in syn])
    sp_x = np.array([split_of[a["source_image_id"]] for a in sdx])
    sp_o = np.array([split_of[s] for s in srcs])
    src_s = np.array([a["source_image_id"] for a in syn])
    src_x = np.array([a["source_image_id"] for a in sdx])
    src_o = np.array(srcs)
    treated_s = np.array([a["augmentation_family"] != "identity_v1" for a in syn])
    mono_s = np.array([bool(a["parameters"].get("monochrome")) for a in syn])
    oi = {s: i for i, s in enumerate(srcs)}
    rep["feature_seconds"] = round(time.time() - t0, 1)
    REPS = ["explicit", "dino_global", "dino_style", "hybrid_style"]

    # ---------------------------------------------------------------- 1. generator identity leakage
    gen = {}
    for name in REPS:
        Xa = np.vstack([Rs[name][treated_s], Rx[name], Ro[name]])
        ya = np.array([0] * int(treated_s.sum()) + [1] * len(sdx) + [2] * len(srcs))
        spa = np.concatenate([sp_s[treated_s], sp_x, sp_o])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        tr, te = spa == "train", spa == "test"
        sc = StandardScaler().fit(Xa[tr])
        m = LogisticRegression(max_iter=4000, C=0.1).fit(sc.transform(Xa[tr]), ya[tr])
        pred = m.predict(sc.transform(Xa[te]))
        gen[name] = {"3way_accuracy_test": round(float((pred == ya[te]).mean()), 4),
                     "chance": round(float(max(Counter(ya[te]).values()) / te.sum()), 4)}
    rep["1_generator_identity_leakage"] = {**gen, "reading": "high accuracy = each generator leaves a recognisable fingerprint in that representation"}

    # ---------------------------------------------------------------- 2. SDXL content preservation
    cos, rank1 = [], []
    Zo = Ro["dino_global"] / np.linalg.norm(Ro["dino_global"], axis=1, keepdims=True)
    Zx = Rx["dino_global"] / np.linalg.norm(Rx["dino_global"], axis=1, keepdims=True)
    S = Zx @ Zo.T
    for i, a in enumerate(sdx):
        o = oi[a["source_image_id"]]
        cos.append(S[i, o])
        rank1.append(int(S[i].argmax() == o))
    Zs = Rs["dino_global"] / np.linalg.norm(Rs["dino_global"], axis=1, keepdims=True)
    Ss = Zs @ Zo.T
    r1s = [int(Ss[i].argmax() == oi[a["source_image_id"]]) for i, a in enumerate(syn) if treated_s[i]]
    rep["2_content_preservation"] = {
        "sdxl_dino_cosine_to_source": {"mean": round(float(np.mean(cos)), 4), "p10": round(float(np.percentile(cos, 10)), 4)},
        "sdxl_own_source_rank1": round(float(np.mean(rank1)), 4), "filmsynth_own_source_rank1": round(float(np.mean(r1s)), 4),
        "by_treatment": {t: round(float(np.mean([c for c, a in zip(cos, sdx) if a["treatment"] == t])), 4)
                         for t in sorted({a["treatment"] for a in sdx})},
        "reading": "rank1 near 1 and cosine high = subject and composition kept; the treatment changed rendering, not content"}

    # ---------------------------------------------------------------- 3. leave-one-generator-out: treated vs untreated
    logo = {}
    for name in REPS:
        # train on filmsynth (treated vs identity), test on SDXL treated vs untreated sources
        Xtr = np.vstack([Rs[name][sp_s == "train"]]); ytr = treated_s[sp_s == "train"].astype(int)
        Xte = np.vstack([Rx[name][sp_x == "test"], Ro[name][sp_o == "test"]])
        yte = np.array([1] * int((sp_x == "test").sum()) + [0] * int((sp_o == "test").sum()))
        a, fitted = probe(Xtr, ytr, Xte, yte)
        # reverse: train on SDXL treated vs sources, test on filmsynth treated vs identity (test sources)
        Xtr2 = np.vstack([Rx[name][sp_x == "train"], Ro[name][sp_o == "train"]])
        ytr2 = np.array([1] * int((sp_x == "train").sum()) + [0] * int((sp_o == "train").sum()))
        b, _ = probe(Xtr2, ytr2, Rs[name][sp_s == "test"], treated_s[sp_s == "test"].astype(int))
        logo[name] = {"train_filmsynth_test_sdxl": a, "train_sdxl_test_filmsynth": b}
        if name == "hybrid_style":
            treated_probe = fitted
    rep["3_logo_treated_vs_untreated"] = logo

    # ---------------------------------------------------------------- 4. leave-one-generator-out pairwise attributes
    scale = {k: float(np.std([s[k] for s in st_o])) + 1e-9 for k in STATS.values()}
    pw = {}
    for name in REPS:
        pw[name] = {}
        for attr, key in STATS.items():
            # filmsynth train pairs from parameters (same source), test = SDXL output vs its own source, measured judge
            trs = [(i, j) for i in np.where(sp_s == "train")[0] for j in np.where(sp_s == "train")[0]
                   if i < j and src_s[i] == src_s[j]]
            prm = PARAMS[attr]
            tr_pairs = []
            for i, j in trs:
                if prm in ("temperature_shift", "saturation_adjustment") and (mono_s[i] or mono_s[j]):
                    continue
                d = syn[i]["parameters"][prm] - syn[j]["parameters"][prm]
                if abs(d) >= 0.15:
                    tr_pairs.append((i, j, 1 if d > 0 else 0))
            te_idx = [(i, oi[a["source_image_id"]]) for i, a in enumerate(sdx) if sp_x[i] == "test"]
            Fcat = np.vstack([Rx[name], Ro[name]])
            te_pairs = measured_pairs(st_x, st_o, te_idx, key, scale=scale[key])
            te_pairs = [(i, len(sdx) + j, l) for i, j, l in te_pairs]
            acc = rank_probe(Rs[name], tr_pairs, Fcat, te_pairs) if len(tr_pairs) > 20 else None
            pw[name][attr] = {"train_pairs_filmsynth": len(tr_pairs), "test_pairs_sdxl_measured": len(te_pairs),
                              "acc_filmsynth_to_sdxl": acc}
    rep["4_logo_pairwise_attributes"] = {**pw, "reading": "judge for SDXL pairs is the sign of the measured-statistic difference (deterministic proxy, not perception)"}

    # ---------------------------------------------------------------- 5. cross-generator style retrieval
    proj = X.supcon(Rs["hybrid_style"][sp_s == "train"], np.array([a["augmentation_family"] for a in syn])[sp_s == "train"], dev)
    Ps, Px, Po = proj(Rs["hybrid_style"]), proj(Rx["hybrid_style"]), proj(Ro["hybrid_style"])
    xr = {}
    for name, (A, B, C) in {**{n: (Rs[n], Rx[n], Ro[n]) for n in REPS}, "supcon_hybrid_style": (Ps, Px, Po)}.items():
        te_q = np.where(sp_x == "test")[0]
        bank = np.vstack([A, C]); bank_treated = np.concatenate([treated_s, np.zeros(len(srcs), bool)])
        bank_src = np.concatenate([src_s, src_o])
        Zb = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
        Zq = B[te_q] / (np.linalg.norm(B[te_q], axis=1, keepdims=True) + 1e-8)
        Sq = Zq @ Zb.T
        prec, same = [], []
        for qi, i in enumerate(te_q):
            order = np.argsort(-Sq[qi])
            same.append(bank_src[order[0]] == src_x[i])
            other = [j for j in order if bank_src[j] != src_x[i]][:5]
            prec.append(np.mean(bank_treated[other]))
        xr[name] = {"treated_precision@5_other_sources": round(float(np.mean(prec)), 4),
                    "chance": round(float(bank_treated.mean()), 4), "same_source_top1": round(float(np.mean(same)), 4)}
    rep["5_cross_generator_retrieval_sdxl_query_filmsynth_bank"] = xr

    # ---------------------------------------------------------------- 6. real film: BlueNeg
    bn_path = P.MANIFESTS / "blueneg_manifest.jsonl"
    real = {}
    if bn_path.exists():
        bn = read_jsonl(bn_path)
        Rb, ims_b = featurize([r["path"] for r in bn], real=True)
        used = set(srcs)
        dig = [r for r in eligible if r["image_id"] not in used and r.get("medium") in ("digital_camera", "smartphone")
               and r["split"] in ("val", "test")]
        Rd, _ = featurize([r["path"] for r in dig], real=True)
        from sklearn.metrics import roc_auc_score
        sc, m = treated_probe
        pb = m.predict_proba(sc.transform(Rb["hybrid_style"]))[:, 1]
        pd = m.predict_proba(sc.transform(Rd["hybrid_style"]))[:, 1]
        intact = np.array([r["partition"] == "blue-intact" for r in bn])
        real["film_vs_digital_auroc_p_treated_hybrid_style"] = {
            "all_blueneg_vs_digital": round(float(roc_auc_score([1] * len(bn) + [0] * len(dig), np.concatenate([pb, pd]))), 4),
            "blue_intact_vs_digital": round(float(roc_auc_score([1] * int(intact.sum()) + [0] * len(dig), np.concatenate([pb[intact], pd]))), 4),
            "n_blueneg": len(bn), "n_blue_intact": int(intact.sum()), "n_digital": len(dig),
            "mean_p_treated": {"blueneg_intact": round(float(pb[intact].mean()), 3), "blueneg_corrupted": round(float(pb[~intact].mean()), 3), "digital": round(float(pd.mean()), 3)}}
        # same-stock retrieval among blue-intact, same roll excluded
        idx = np.where(intact)[0]
        stock = np.array([bn[i]["film_stock_verified"] for i in idx]); roll = np.array([bn[i]["roll_id"] for i in idx])
        ret = {}
        for name, F in {**{n: Rb[n] for n in REPS}, "supcon_hybrid_style": proj(Rb["hybrid_style"])}.items():
            Z = F[idx] / (np.linalg.norm(F[idx], axis=1, keepdims=True) + 1e-8)
            Sm = Z @ Z.T
            prec, ch = [], []
            for q in range(len(idx)):
                ok = roll != roll[q]
                if (stock[ok] == stock[q]).sum() == 0:
                    continue
                order = [j for j in np.argsort(-Sm[q]) if ok[j]][:5]
                prec.append(np.mean(stock[order] == stock[q]))
                ch.append((stock[ok] == stock[q]).mean())
            ret[name] = {"same_stock_precision@5_other_rolls": round(float(np.mean(prec)), 4), "chance": round(float(np.mean(ch)), 4), "n_queries": len(prec)}
        real["same_stock_retrieval_blue_intact"] = {**ret, "stocks": dict(Counter(stock.tolist())),
                                                     "reading": "owner-verified stock labels, one photographer; same-roll neighbours excluded so scene overlap is reduced but not removed (same trips span rolls)"}
    rep["6_real_film_blueneg"] = real or {"status": "blueneg manifest missing; run scripts/ingest_blueneg.py"}

    # ---------------------------------------------------------------- 7. robustness of treated-vs-untreated on SDXL
    sc, m = treated_probe
    te_x = np.where(sp_x == "test")[0]
    base_pred = m.predict(sc.transform(Rx["hybrid_style"][te_x]))

    def perturb(im, kind):
        if kind == "jpeg_q55":
            b = io.BytesIO(); im.save(b, "JPEG", quality=55); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        if kind == "rescale_0.5":
            return im.resize((im.width // 2, im.height // 2), Image.BILINEAR).resize(im.size, Image.BILINEAR)
        return Image.fromarray(np.clip(np.asarray(im, dtype=np.float32) * 2 ** 0.5, 0, 255).astype(np.uint8))
    rob = {}
    for kind in ("jpeg_q55", "rescale_0.5", "exposure_+0.5ev"):
        pim = [perturb(ims_x[i], kind) for i in te_x]
        pex = np.stack([X.explicit_of(im) for im in pim]); pdg, pds = dino(pim)
        PR = X.reps_from(pex, pdg, pds)
        pred = m.predict(sc.transform(PR["hybrid_style"]))
        rob[kind] = {"still_predicted_treated": round(float(pred.mean()), 4), "unchanged_prediction": round(float((pred == base_pred).mean()), 4)}
    rep["7_robustness_sdxl_treated_probe"] = rob
    rep["total_seconds"] = round(time.time() - t0, 1)
    rep["gpu"] = torch.cuda.get_device_name(0) + " (dev workstation)" if dev.type == "cuda" else "cpu"
    (P.REPORTS / "synth_experiment_v2.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
