"""C08: does a different or deeper image encoder help the one film task with real ground truth (film scan vs digital)?

    python -m research.c08.encoders embed     # -> data/c08/encoders/<name>.npy (2,243 film-likeness images)
    python -m research.c08.encoders evaluate  # -> reports/c08/encoders.json

Same images and uploader-disjoint split as the deployed film head (data/lut/film_likeness). Protocol: balanced logistic
regression on z-scored [36 measured statistics + embedding] and on the embedding alone; AUROC on held-out uploaders
with a paired uploader bootstrap against the production embedding (DINOv2-small CLS + mean patch, block 12).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SET, OUT, REPORTS = Path("data/lut/film_likeness"), Path("data/c08/encoders"), Path("reports/c08")


def rows():
    return [json.loads(l) for l in open(SET / "manifest.jsonl")]


def _img(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = 512 / max(im.size)
    return im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.LANCZOS) if s < 1 else im


def cmd_embed(a) -> None:
    from research.c08.pairs import DinoLayers, dino_crop
    from research.photographers.embed import hf_encoder
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows()
    imgs = [_img(r["path"]) for r in rs]
    if not (OUT / "dinov2_small_layers.npy").exists():
        d = DinoLayers()
        np.save(OUT / "dinov2_small_layers.npy", d(np.stack([dino_crop(np.asarray(i)) for i in imgs])))
        print("dinov2_small_layers done", flush=True)
    for name in ("dinov2_base", "siglip_base", "clip_b32"):
        if (OUT / f"{name}.npy").exists():
            continue
        enc = hf_encoder(name, "cuda")
        X = np.concatenate([enc(imgs[i:i + 64]) for i in range(0, len(imgs), 64)])
        np.save(OUT / f"{name}.npy", X.astype(np.float32)); print(name, X.shape, flush=True)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def cmd_evaluate(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rs = rows()
    y = np.array([r["label"] for r in rs]); g = np.array([r["uploader"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs])
    F = np.nan_to_num(np.load(SET / "features.npy").astype(np.float32))
    L = np.load(OUT / "dinov2_small_layers.npy").astype(np.float32)
    encs = {"prod_dinov2_small_L12": L[:, 1536:2304], "dinov2_small_L4": L[:, 0:768], "dinov2_small_L8": L[:, 768:1536], "dinov2_small_L4+L8+L12": L,
            "dinov2_base": np.load(OUT / "dinov2_base.npy"), "siglip_base": np.load(OUT / "siglip_base.npy"), "clip_b32": np.load(OUT / "clip_b32.npy")}
    scores = {}
    def fit(X):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        return m.decision_function(sc.transform(X[~tr]))
    scores["stats36_only"] = fit(F)
    for name, E in encs.items():
        scores[f"{name}"] = fit(E)
        scores[f"stats36+{name}"] = fit(np.c_[F, E])
    yt, gt = y[~tr], g[~tr]
    ug = np.unique(gt); members = {u: np.where(gt == u)[0] for u in ug}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[u] for u in rng.choice(ug, len(ug))]) for _ in range(500)]
    base = scores["stats36+prod_dinov2_small_L12"]
    out = {}
    for k, s in scores.items():
        vals = [_auc(yt[b], s[b]) for b in boots]
        diff = [_auc(yt[b], s[b]) - _auc(yt[b], base[b]) for b in boots]
        out[k] = {"auroc": _auc(yt, s), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                  "diff_vs_stats+prod": [float(np.mean(diff)), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        print(f"{k:36s} AUROC {out[k]['auroc']:.3f} [{out[k]['ci'][0]:.3f},{out[k]['ci'][1]:.3f}]  diff vs stats+prod {out[k]['diff_vs_stats+prod'][0]:+.3f} [{out[k]['diff_vs_stats+prod'][1]:+.3f},{out[k]['diff_vs_stats+prod'][2]:+.3f}]", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "encoders.json").write_text(json.dumps({"n_test": int((~tr).sum()), "uploaders_test": int(len(ug)), "results": out}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed"); sub.add_parser("evaluate"); sub.add_parser("controls"); sub.add_parser("probes")
    a = ap.parse_args()
    {"embed": cmd_embed, "evaluate": cmd_evaluate, "controls": cmd_controls, "probes": cmd_probes}[a.cmd](a)



def cmd_controls(a) -> None:
    """Confound controls: colour images only (monochrome is 345 film vs 12 digital), and colour film vs colour
    digital-camera images only (no smartphones), trained and tested within the restricted set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rs = rows()
    y = np.array([r["label"] for r in rs]); g = np.array([r["uploader"] for r in rs]); tr0 = np.array([r["split"] == "train" for r in rs])
    colour = np.array([bool(r.get("colour")) for r in rs]); ds = np.array([r["dataset"] for r in rs])
    F = np.nan_to_num(np.load(SET / "features.npy").astype(np.float32)); L = np.load(OUT / "dinov2_small_layers.npy").astype(np.float32)
    encs = {"stats36": F, "stats36+prod_L12": np.c_[F, L[:, 1536:2304]], "stats36+dinov2_small_L4+L8+L12": np.c_[F, L], "stats36+siglip_base": np.c_[F, np.load(OUT / "siglip_base.npy")],
            "stats36+clip_b32": np.c_[F, np.load(OUT / "clip_b32.npy")]}
    subsets = {"colour_only": colour, "colour_film_vs_colour_digital_camera": colour & ((y == 1) | np.isin(ds, ["commoncatalog_digital_camera", "commoncatalog_digital_filtered"]))}
    out = {}
    for sname, keep in subsets.items():
        tr, te = keep & tr0, keep & ~tr0
        ug = np.unique(g[te]); members = {u: np.where(g[te] == u)[0] for u in ug}; rng = np.random.default_rng(0)
        boots = [np.concatenate([members[u] for u in rng.choice(ug, len(ug))]) for _ in range(500)]
        out[sname] = {"n_test": int(te.sum()), "n_test_digital": int((y[te] == 0).sum())}
        for name, X in encs.items():
            sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tr]), y[tr])
            s = m.decision_function(sc.transform(X[te])); yt = y[te]
            vals = [roc_auc_score(yt[b], s[b]) for b in boots]
            out[sname][name] = {"auroc": float(roc_auc_score(yt, s)), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
            print(f"{sname:38s} {name:34s} AUROC {out[sname][name]['auroc']:.3f} [{out[sname][name]['ci'][0]:.3f},{out[sname][name]['ci'][1]:.3f}]  n={te.sum()} digital={(yt == 0).sum()}", flush=True)
    (REPORTS / "encoders_controls.json").write_text(json.dumps(out, indent=1))



def _content_only(im: Image.Image) -> Image.Image:
    """Grey, shrunk to 48 px on the long side and enlarged again: subject, layout and era survive; grain, fine texture,
    colour rendering and most tonal detail do not."""
    g = im.convert("L")
    s = 48 / max(g.size)
    small = g.resize((max(4, round(g.width * s)), max(4, round(g.height * s))), Image.BILINEAR)
    return small.resize(g.size, Image.BICUBIC).convert("RGB")


def _filmify(im: Image.Image, lut, seed: int) -> Image.Image:
    x = lut.apply_u8(np.asarray(im))
    rng = np.random.default_rng(seed)
    f = x.astype(np.float32) / 255 + rng.normal(0, 0.02, x.shape[:2])[..., None] * (1 - 0.6 * x.mean(-1, keepdims=True) / 255)
    return Image.fromarray(np.clip(f * 255, 0, 255).round().astype(np.uint8))


def cmd_probes(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from research.c08.common import all_luts
    from research.c08.pairs import DinoLayers, dino_crop
    from research.photographers.embed import hf_encoder
    rs = rows()
    y = np.array([r["label"] for r in rs]); tr = np.array([r["split"] == "train" for r in rs]); colour = np.array([bool(r.get("colour")) for r in rs])
    ds = np.array([r["dataset"] for r in rs])
    keep = colour & ((y == 1) | np.isin(ds, ["commoncatalog_digital_camera", "commoncatalog_digital_filtered"]))
    imgs = [_img(r["path"]) for r in rs]
    dino, clip = DinoLayers(), hf_encoder("clip_b32", "cuda")
    def embed_all(images):
        D = dino(np.stack([dino_crop(np.asarray(i)) for i in images])).astype(np.float32)
        C = np.concatenate([clip(images[i:i + 64]) for i in range(0, len(images), 64)])
        return {"dinov2_small_L4+L8+L12": D, "prod_L12": D[:, 1536:2304], "clip_b32": C}
    content = embed_all([_content_only(i) for i in imgs])
    full = {"dinov2_small_L4+L8+L12": np.load(OUT / "dinov2_small_layers.npy").astype(np.float32), "clip_b32": np.load(OUT / "clip_b32.npy")}
    full["prod_L12"] = full["dinov2_small_L4+L8+L12"][:, 1536:2304]
    out = {"content_only": {}, "rendering_sensitivity": {}}
    tri, tei = keep & tr, keep & ~tr
    presets = [x for x in all_luts() if x["source"] == "owner_presets"][:13]
    test_digital = np.where(tei & (y == 0))[0]
    film_versions = [_filmify(imgs[i], presets[k % len(presets)]["lut"], int(i)) for i in test_digital for k in range(3)]
    filmed = embed_all(film_versions)
    for name in ("prod_L12", "dinov2_small_L4+L8+L12", "clip_b32"):
        # content-only: train and test on degraded images
        X = content[name]; sc = StandardScaler().fit(X[tri]); m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tri]), y[tri])
        out["content_only"][name] = float(roc_auc_score(y[tei], m.decision_function(sc.transform(X[tei]))))
        # rendering sensitivity: the model trained on full images scores digital test photos before and after a film look + grain
        X = full[name]; sc = StandardScaler().fit(X[tri]); m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=4000).fit(sc.transform(X[tri]), y[tri])
        before = np.repeat(m.decision_function(sc.transform(X[test_digital])), 3)
        after = m.decision_function(sc.transform(filmed[name]))
        film_scores = m.decision_function(sc.transform(X[tei & (y == 1)]))
        out["rendering_sensitivity"][name] = {"mean_logit_shift": float((after - before).mean()), "share_increased": float((after > before).mean()),
                                              "share_now_above_median_film": float((after > np.median(film_scores)).mean()),
                                              "share_before_above_median_film": float((before > np.median(film_scores)).mean())}
        print(f"{name:26s} content-only AUROC {out['content_only'][name]:.3f} | film look+grain on digital: logit {out['rendering_sensitivity'][name]['mean_logit_shift']:+.2f}, "
              f"up {out['rendering_sensitivity'][name]['share_increased']:.2f}, above median film {out['rendering_sensitivity'][name]['share_before_above_median_film']:.2f} -> {out['rendering_sensitivity'][name]['share_now_above_median_film']:.2f}", flush=True)
    (REPORTS / "encoders_probes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
