#!/usr/bin/env python
"""E49: invariance to geometry and compression, sensitivity to rendering edits, on real photographs the student never trained on.

  python scripts/e49_robustness.py   # ~1 GPU-minute for the DINOv2 reference rows -> reports/e49_robustness.json

Data: the commoncatalog_ccby VALIDATION originals (real film scans, digital and smartphone photographs; never a training
input of any student) and their eight on-disk edit variants from the manifest's own edit pipeline (film_emulation,
bw_film_emulation, fake_grain_filter, hdr_overprocessed, oversharpened, posterized, digital_clean_sharp,
bw_harsh_digital), a pipeline no student or teacher was trained on. Perturbations computed here on the originals:
centre crop 80 %, top-left crop 70 % (content changes; reported only), horizontal flip, JPEG q50, 0.5x rescale.
Representations: the three E43 students' pooled PCA-64, the ImageNet-initialised pooled PCA-64, frozen DINOv2-Small
CLS++mean and L4+L8 PCA-64, the 28 explicit statistics (standardised); every reduction fitted on the CC-BY train originals.
Per representation:
  invariance    own-image rank-1 of a perturbed image among the originals, and mean cosine to its own original
  edit signal   edit-type p@5 among originals + edited images with the image's own family excluded (9 classes; chance = prior)
  content       own-source rank-1 of an edited image among the originals (chance 1/n)
  displacement  (1 - cosine) for each edit, relative to the centre-crop displacement
Pre-registered rule (research/backlog.yaml E49): supported = 1 iff for every seed the student's PCA-64 has own-image
rank-1 >= 0.9 under centre crop, flip, JPEG q50 and rescale, edit-type p@5 >= 0.25 (more than twice chance) and edited
own-source rank-1 <= 0.5. Validation split only; no test split, no BlueNeg.
"""
import io
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import synth_experiment as X  # noqa: E402
import synth_experiment_v3 as V3  # noqa: E402
from e29_audit import retrieval  # noqa: E402
from e43_penultimate_pca import PCA64  # noqa: E402
from e48_frozen_vs_trained import INIT, REF, fit_embed  # noqa: E402
from style_benchmark import unit  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402

FIT = P.EMBEDDINGS / "ccby_train_originals_features_v1"
PERTURB = ["crop80", "crop70_corner", "flip", "jpeg50", "rescale0.5"]
INVARIANT = ["crop80", "flip", "jpeg50", "rescale0.5"]


def perturb(a, kind):
    h, w = a.shape[:2]
    if kind == "crop80":
        y, x = int(0.1 * h), int(0.1 * w); return a[y:h - y, x:w - x]
    if kind == "crop70_corner":
        return a[: int(0.7 * h), : int(0.7 * w)]
    if kind == "flip":
        return a[:, ::-1]
    if kind == "jpeg50":
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=50); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    im = Image.fromarray(a); return np.asarray(im.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR))


def student_embed(tag, arrays, output="features_576"):
    import onnxruntime as ort
    from filmlook_infer.compact import preprocess
    ck = P.CHECKPOINTS / f"student_{tag}"; sess = ort.InferenceSession(str(ck / "model.onnx"), providers=["CPUExecutionProvider"]); oi = json.loads((ck / "meta.json").read_text())["outputs"].index(output)
    acc = [sess.run(None, {"pixel_values": np.concatenate([preprocess(Image.fromarray(x)) for x in arrays[i:i + 64]])})[oi] for i in range(0, len(arrays), 64)]
    return np.concatenate(acc).astype(np.float32)


def main():
    import torch
    t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = [r for r in read_jsonl(P.DATA / "manifests" / "manifest.jsonl") if r.get("dataset_id") == "commoncatalog_ccby"]
    fit_paths = [r["path"] for r in man if r["split"] == "train" and r["edit_type"] == "original"]
    orig = sorted([r for r in man if r["split"] == "val" and r["edit_type"] == "original"], key=lambda r: r["image_id"]); fam = [r["family_id"] for r in orig]
    edits = sorted([r for r in man if r["split"] == "val" and r["edit_type"] != "original"], key=lambda r: (r["edit_type"], r["image_id"]))
    assert all(r["family_id"] in set(fam) for r in edits)
    with Pool(16) as pool:
        A0 = pool.map(V3._load, [(r["path"], r["image_id"], False, True) for r in orig], chunksize=4)
        AE = pool.map(V3._load, [(r["path"], r["image_id"], False, True) for r in edits], chunksize=4)
    AP = {k: [perturb(a, k) for a in A0] for k in PERTURB}
    arrays = [*A0, *AE, *[a for k in PERTURB for a in AP[k]]]
    n0, ne = len(A0), len(AE); idx = {"orig": np.arange(n0), "edit": np.arange(n0, n0 + ne)}
    for j, k in enumerate(PERTURB):
        idx[k] = np.arange(n0 + ne + j * n0, n0 + ne + (j + 1) * n0)
    fam_all = np.array([*fam, *[r["family_id"] for r in edits]]); etype = np.array(["original"] * n0 + [r["edit_type"] for r in edits]); medium = np.array([r.get("medium") for r in orig])
    Fit = dict(np.load(FIT.with_suffix(".npz")))
    # embeddings of every array under every representation
    reps = {}
    for tag in (*REF, INIT):
        reps[f"{tag}:pca64"] = PCA64(fit_embed(tag, fit_paths))(student_embed(tag, arrays))
    with Pool(16) as pool:
        ex = np.stack(pool.map(X.explicit_of, [Image.fromarray(a) for a in arrays], chunksize=8))
    mu, sd = Fit["explicit"].mean(0), Fit["explicit"].std(0) + 1e-6; reps["explicit28_std"] = (ex - mu) / sd
    dg, ds = X.Dino(dev)([Image.fromarray(a) for a in arrays]); reps["dino_global:pca64"] = PCA64(Fit["dino_global"])(dg); reps["dino_style:pca64"] = PCA64(Fit["dino_style"])(ds)
    out = {"experiment": "E49_robustness", "validation_data_only": True, "n_originals": n0, "n_edits": ne, "media": {m: int((medium == m).sum()) for m in sorted(set(medium))}, "representations": {}}
    for name, Z in reps.items():
        E = unit(np.nan_to_num(Z.astype(np.float32))); Eo = E[idx["orig"]]; r = {"invariance": {}, "displacement": {}}
        for k in PERTURB:
            Ep = E[idx[k]]; S = Ep @ Eo.T; own = np.einsum("ij,ij->i", Ep, Eo)
            r["invariance"][k] = {"own_rank1": round(float(np.mean(S.argmax(1) == np.arange(n0))), 4), "mean_cos": round(float(own.mean()), 4), "own_rank1_by_medium": {m: round(float(np.mean(S.argmax(1)[medium == m] == np.where(medium == m)[0])), 3) for m in sorted(set(medium))}}
        base = 1 - r["invariance"]["crop80"]["mean_cos"] + 1e-6
        Ee = E[idx["edit"]]; S = Ee @ Eo.T; fi = {f: i for i, f in enumerate(fam)}; own_idx = np.array([fi[r_["family_id"]] for r_ in edits])
        r["content"] = {"edited_own_source_rank1": round(float(np.mean(S.argmax(1) == own_idx)), 4), "chance": round(1 / n0, 4)}
        own = np.einsum("ij,ij->i", Ee, Eo[own_idx])
        for et in sorted(set(etype) - {"original"}):
            m = etype[idx["edit"]] == et; r["displacement"][et] = {"one_minus_cos": round(float((1 - own[m]).mean()), 4), "relative_to_crop80": round(float((1 - own[m]).mean() / base), 2)}
        Eall = E[np.r_[idx["orig"], idx["edit"]]]; r["edit_type_p5"] = retrieval(Eall, etype, fam_all)
        out["representations"][name] = r
        print(f"{name:34s} crop80 {r['invariance']['crop80']['own_rank1']:.3f} flip {r['invariance']['flip']['own_rank1']:.3f} jpeg {r['invariance']['jpeg50']['own_rank1']:.3f} rescale {r['invariance']['rescale0.5']['own_rank1']:.3f} corner {r['invariance']['crop70_corner']['own_rank1']:.3f} | "
              f"edit p@5 {r['edit_type_p5']['precision@5']:.3f} ch {r['edit_type_p5']['chance']:.3f} | edited own-source {r['content']['edited_own_source_rank1']:.3f} | film_emulation disp x{r['displacement']['film_emulation']['relative_to_crop80']:.1f}", flush=True)
    checks = {}
    for tag in REF:
        r = out["representations"][f"{tag}:pca64"]
        checks[tag] = {"invariant_rank1_ge_0.9": min(r["invariance"][k]["own_rank1"] for k in INVARIANT) >= 0.9, "edit_p5_ge_0.25": r["edit_type_p5"]["precision@5"] >= 0.25, "edited_own_source_le_0.5": r["content"]["edited_own_source_rank1"] <= 0.5}
    ok = all(all(c.values()) for c in checks.values()); mean = lambda f: round(float(np.mean([f(out["representations"][f"{t}:pca64"]) for t in REF])), 4)  # noqa: E731
    out["checks"] = checks
    out["summary"] = {"supported": int(ok), "seeds_passing": int(sum(all(c.values()) for c in checks.values())), "student_min_invariant_rank1": round(float(min(out["representations"][f"{t}:pca64"]["invariance"][k]["own_rank1"] for t in REF for k in INVARIANT)), 4),
                      "student_corner_rank1_mean": mean(lambda r: r["invariance"]["crop70_corner"]["own_rank1"]), "student_edit_p5_mean": mean(lambda r: r["edit_type_p5"]["precision@5"]), "edit_p5_chance": out["representations"][f"{REF[0]}:pca64"]["edit_type_p5"]["chance"],
                      "student_edited_own_source_mean": mean(lambda r: r["content"]["edited_own_source_rank1"]), "student_film_emulation_disp_rel_mean": mean(lambda r: r["displacement"]["film_emulation"]["relative_to_crop80"]),
                      **{f"{n}_edit_p5": out["representations"][n]["edit_type_p5"]["precision@5"] for n in (f"{INIT}:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std")},
                      **{f"{n}_edited_own_source": out["representations"][n]["content"]["edited_own_source_rank1"] for n in (f"{INIT}:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std")},
                      **{f"{n}_crop80_rank1": out["representations"][n]["invariance"]["crop80"]["own_rank1"] for n in (f"{INIT}:pca64", "dino_global:pca64", "dino_style:pca64", "explicit28_std")}, "seconds": round(time.time() - t0, 1)}
    (P.REPORTS / "e49_robustness.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
