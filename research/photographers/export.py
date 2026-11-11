"""Export the serving index for the website: vectors and per-work provenance, no images.

  python -m research.photographers.export [--seed 0] [--min-works 40] [--out dist/refmatch]

Contents of dist/refmatch/<version>/:
  vectors.f16.npy   N x 768 float16: the production encoder's [CLS, mean patch] halves, each L2-normalised, the whole
                    L2-normalised again (cosine on it = mean of the two halves' cosines).
  works.jsonl       one row per vector: work_id, photographer_id, title, year, source_url, licence, credit, rights_class.
  calibration.json  top-m aggregation size and score thresholds, all chosen on the validation split of the benchmark
                    (bank = bank split only, so they are conservative for the full index), the benchmark headline, and
                    the reading of 500 modern CC-BY digital photographs against the same bank.
  manifest.json     encoder file hash, preprocessing string, counts, sha256 of every file.
  siglip.f16.npy    (when data/embeddings/photographers/siglip_base_onnx_int8.npz exists) N x 768 float16: the optional
                    style encoder (SigLIP-B/16 int8 ONNX, C19), L2-normalised; calibration.json -> fusion_siglip holds
                    its z-scored fusion (weights chosen on validation, as for `fusion`), manifest.json -> style_encoder
                    the model files' sha256 and the preprocessing. `fusion` and every other file are unchanged by it.

The index holds every eligible work of photographers with at least --min-works works (bank, val and test splits:
the benchmark is finished before this runs). Only public-domain / CC0 works can be in it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml.refdb import manifest  # noqa: E402
from research.photographers import evaluate as E  # noqa: E402

STYLE = {"name": "google/siglip-base-patch16-224 image tower, int8 dynamic (MatMul, per-channel), ONNX with external weights",
         "licence": "Apache-2.0", "files": ("siglip_base_int8_ext.onnx", "siglip_base_int8_ext.weights"),
         "preprocessing": "RGB, resize to 224 x 224 bicubic (no crop), (x / 255 - 0.5) / 0.5; output L2-normalised pooled embedding (768)"}
PREPROCESSING = "dinov2_pixels: short side 256 bicubic, centre crop 224, ImageNet normalisation; ONNX fp32 [CLS, mean patch]"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-works", type=int, default=40)
    ap.add_argument("--out", default="dist/refmatch")
    ap.add_argument("--version", default="", help="reuse this version name (only when the base files are byte-identical to it)")
    a = ap.parse_args()
    rows = E.load_rows(); ids = [r["work_id"] for r in rows]
    X = E.load_vectors("prod", ids)
    pid = np.array([r["photographer_id"] for r in rows])
    counts = Counter(pid.tolist())
    keep = np.array([counts[p] >= a.min_works for p in pid])
    # thresholds from the benchmark protocol (same seed, same groups)
    groups, _ = E.leakage_groups(rows, E.l2(X[:, :384])); split = E.split_groups(rows, groups, a.seed)
    classes = sorted(p for p, n in counts.items() if n >= a.min_works and p not in E.HELD_OUT)
    cidx = {p: i for i, p in enumerate(classes)}; y = np.array([cidx.get(p, -1) for p in pid]); known = y >= 0
    b, v, t = known & (split == "bank"), known & (split == "val"), known & (split == "test")
    Sv, St = X[v] @ X[b].T, X[t] @ X[b].T
    m = max([1, 2, 3, 5, 8, 12], key=lambda q: E.class_metrics(E.topm_scores(Sv, y[b], len(classes), q), y[v], len(classes))["top1"])
    Pv, Pt = E.topm_scores(Sv, y[b], len(classes), m), E.topm_scores(St, y[b], len(classes), m)
    cal = {"aggregation": "mean cosine of the m closest works per photographer", "m": int(m),
           "thresholds": {"weak": round(float(np.percentile(Pv.max(1), 5)), 4), "moderate": round(float(np.percentile(Pv.max(1), 50)), 4),
                          "strong": round(float(np.percentile(Pv.max(1), 85)), 4),
                          "basis": "5th / 50th / 85th percentile of the best photographer score of validation photographs by bank photographers"},
           "benchmark": {"photographers": len(classes), "n_test": int(t.sum()), "top1": round(float((Pt.argmax(1) == y[t]).mean()), 4),
                         "top3": round(float(np.mean([y[t][i] in np.argsort(-Pt[i])[:3] for i in range(int(t.sum()))])), 4),
                         "uniform_chance": round(1 / len(classes), 4), "known_test_above_weak": round(float((Pt.max(1) >= np.percentile(Pv.max(1), 5)).mean()), 4)}}
    held = np.isin(pid, E.HELD_OUT)
    if held.any():
        Ph = E.topm_scores(X[held] @ X[b].T, y[b], len(classes), m).max(1)
        cal["benchmark"]["unknown_photographers_above_weak"] = round(float((Ph >= cal["thresholds"]["weak"]).mean()), 4)
    mod = E.EMB / "modern_prod.npz"
    if mod.exists():
        Pm = E.topm_scores(E.l2(np.load(mod)["X"].astype(np.float32)) @ X[b].T, y[b], len(classes), m).max(1)
        cal["modern_digital"] = {"n": int(len(Pm)), "above_weak": round(float((Pm >= cal["thresholds"]["weak"]).mean()), 4),
                                 "above_moderate": round(float((Pm >= cal["thresholds"]["moderate"]).mean()), 4),
                                 "above_strong": round(float((Pm >= cal["thresholds"]["strong"]).mean()), 4), "p50": round(float(np.percentile(Pm, 50)), 4)}
    # Style-aware reranking: measured colour / tone (205 values) and the 17 structural axes, fused with the embedding
    # score exactly as research.photographers.evaluate does (weights and scales from the validation split).
    fusion, extra = None, {}
    desc = E.EMB / "descriptors.npz"
    if desc.exists():
        d = np.load(desc, allow_pickle=False); pos = {w: i for i, w in enumerate(d["work_ids"].tolist())}
        if all(w in pos for w in ids):
            order = [pos[w] for w in ids]
            ax = np.nan_to_num(d["axes"][order], nan=0.5); f205 = d["film205"][order]
            ax_mu = ax[b].mean(0); f_mu, f_sd = f205[b].mean(0), f205[b].std(0) + 1e-6
            spaces = {"axes17": E.l2(ax - ax_mu), "film205": E.l2((f205 - f_mu) / f_sd)}
            scores = {"prod": (Pv, Pt, int(m))}
            for name, Z in spaces.items():
                Zv, Zt = Z[v] @ Z[b].T, Z[t] @ Z[b].T
                mm = max([1, 2, 3, 5, 8, 12], key=lambda q: E.class_metrics(E.topm_scores(Zv, y[b], len(classes), q), y[v], len(classes))["top1"])
                scores[name] = (E.topm_scores(Zv, y[b], len(classes), mm), E.topm_scores(Zt, y[b], len(classes), mm), int(mm))
            zs = lambda A, ref: (A - ref.mean()) / (ref.std() + 1e-9)  # noqa: E731
            grid, best = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], None
            for w1 in grid:
                for w2 in grid:
                    acc = float(((zs(scores["prod"][0], scores["prod"][0]) + w1 * zs(scores["film205"][0], scores["film205"][0]) + w2 * zs(scores["axes17"][0], scores["axes17"][0])).argmax(1) == y[v]).mean())
                    if best is None or acc > best[0] + 1e-9:
                        best = (acc, w1, w2)
            fused = zs(scores["prod"][1], scores["prod"][0]) + best[1] * zs(scores["film205"][1], scores["film205"][0]) + best[2] * zs(scores["axes17"][1], scores["axes17"][0])
            fusion = {"weights": {"prod": 1.0, "film205": best[1], "axes17": best[2]},
                      "scale": {k: {"mean": float(scores[k][0].mean()), "std": float(scores[k][0].std()), "m": scores[k][2]} for k in scores},
                      "axes17": {"mean": ax_mu.tolist(), "names": d["axis_names"].tolist()}, "film205": {"mean": f_mu.tolist(), "std": f_sd.tolist()},
                      "benchmark": {"top1": round(float((fused.argmax(1) == y[t]).mean()), 4), "top1_embedding_only": cal["benchmark"]["top1"], "val_top1": round(best[0], 4)},
                      "rule": "photographers are admitted by the embedding score (thresholds above); those admitted are ordered by the fused score"}
            extra = {"axes17.f16.npy": spaces["axes17"], "film205.f16.npy": spaces["film205"]}
    cal["fusion"] = fusion
    # Optional style encoder (C19): its own fusion, the shipped one stays as it is. Weights on validation, test read once.
    sig = E.load_vectors("siglip_base_onnx_int8", ids)
    if fusion is not None and sig is not None:
        Gv, Gt = sig[v] @ sig[b].T, sig[t] @ sig[b].T
        ms = max([1, 2, 3, 5, 8, 12], key=lambda q: E.class_metrics(E.topm_scores(Gv, y[b], len(classes), q), y[v], len(classes))["top1"])
        scores["siglip"] = (E.topm_scores(Gv, y[b], len(classes), ms), E.topm_scores(Gt, y[b], len(classes), ms), int(ms))
        Z = {k: (zs(scores[k][0], scores[k][0]), zs(scores[k][1], scores[k][0])) for k in scores}
        sgrid, best_s = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], None
        for ws in sgrid:
            for w1 in grid:
                for w2 in grid:
                    acc = float(((Z["prod"][0] + w1 * Z["film205"][0] + w2 * Z["axes17"][0] + ws * Z["siglip"][0]).argmax(1) == y[v]).mean())
                    if best_s is None or acc > best_s[0] + 1e-9:
                        best_s = (acc, w1, w2, ws)
        fused_s = Z["prod"][1] + best_s[1] * Z["film205"][1] + best_s[2] * Z["axes17"][1] + best_s[3] * Z["siglip"][1]
        cal["fusion_siglip"] = {
            "weights": {"prod": 1.0, "film205": best_s[1], "axes17": best_s[2], "siglip": best_s[3]},
            "scale": {k: {"mean": float(scores[k][0].mean()), "std": float(scores[k][0].std()), "m": scores[k][2]} for k in scores},
            "weight_grid": {"siglip": sgrid, "film205": grid, "axes17": grid},
            "benchmark": {"top1": round(float((fused_s.argmax(1) == y[t]).mean()), 4),
                          "top3": round(float(np.mean([y[t][i] in np.argsort(-fused_s[i])[:3] for i in range(int(t.sum()))])), 4),
                          "top1_without_style": fusion["benchmark"]["top1"], "val_top1": round(best_s[0], 4)},
            "rule": "as fusion, with the style encoder's top-m score added; used only when the website's style encoder is on"}
        extra["siglip.f16.npy"] = sig
    with manifest.connect() as c:
        meta = {r["work_id"]: dict(r) for r in c.execute(
            "SELECT work_id, photographer_id, title, year, source_url, license, license_evidence_url, credit, rights_class, rights_status FROM works")}
    sel = [i for i in range(len(rows)) if keep[i]]
    assert all(meta[ids[i]]["rights_status"] == "eligible" and meta[ids[i]]["rights_class"] in ("public_domain", "cc0") for i in sel)
    version = a.version or time.strftime("refmatch-%Y%m%d-") + hashlib.sha256("".join(ids[i] for i in sel).encode()).hexdigest()[:8]
    target = ROOT / a.out / version
    reuse = bool(a.version) and (target / "manifest.json").exists()
    out = target.with_name(version + ".new") if reuse else target  # a reused name is replaced only after the check below
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.f16.npy", X[sel].astype(np.float16))
    for name, Z in extra.items():
        np.save(out / name, Z[sel].astype(np.float16))
    with (out / "works.jsonl").open("w") as f:
        for i in sel:
            mrow = meta[ids[i]]
            f.write(json.dumps({k: mrow[k] for k in ("work_id", "photographer_id", "title", "year", "source_url", "license", "credit", "rights_class")}, ensure_ascii=False) + "\n")
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    enc = ROOT / "checkpoints" / "refdb" / "dinov2_small_cls_patch.onnx"
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name != "manifest.json"}
    if reuse:  # a reused version name must keep every base file byte-identical
        before = json.loads((target / "manifest.json").read_text())["files"]
        changed = [n for n, h in before.items() if n != "calibration.json" and files.get(n) != h]
        old_cal = {k: v for k, v in json.loads((target / "calibration.json").read_text()).items() if k != "fusion_siglip"}
        if changed or json.dumps({k: v for k, v in cal.items() if k != "fusion_siglip"}) != json.dumps(old_cal):
            raise SystemExit(f"--version {version}: base files differ ({changed or 'calibration'}); kept {target}, new files in {out}")
    style = None
    if "siglip.f16.npy" in files:
        enc_dir = ROOT / "checkpoints" / "c19"
        style = {**STYLE, "files": {n: hashlib.sha256((enc_dir / n).read_bytes()).hexdigest() for n in STYLE["files"]}, "dim": int(sig.shape[1]),
                 "vectors": "siglip.f16.npy", "parity": "vectors computed with the int8 ONNX graph (same weights as the shipped file) and this preprocessing"}
    (out / "manifest.json").write_text(json.dumps({
        "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "encoder": "facebook/dinov2-small (ONNX fp32)",
        "encoder_sha256": hashlib.sha256(enc.read_bytes()).hexdigest(), "preprocessing": PREPROCESSING, "dim": 768, "metric": "cosine (dot product of L2-normalised vectors)",
        "works": len(sel), "photographers": dict(Counter(pid[sel].tolist())), "rights": dict(Counter(meta[ids[i]]["rights_class"] for i in sel)), "files": files,
        **({"style_encoder": style} if style else {})}, indent=1))
    if reuse:
        import shutil
        shutil.rmtree(target); out.rename(target); out = target
    print(version, len(sel), "works", len(set(pid[sel].tolist())), "photographers"); print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
