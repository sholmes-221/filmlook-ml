#!/usr/bin/env python
"""Generate the controlled synthetic film-aesthetic dataset (film-synth-v1).

  python scripts/synth_generate.py --sources 100 --variants 8

- Sources: eligible, non-quarantined, COLOUR originals (monochrome sources would make colour families ill-defined).
  The existing uploader-grouped splits are kept (70 / 15 / 15 sources from train / val / test), so all variants of
  one source stay in one split.
- Per source: identity_v1 plus (variants - 1) distinct film families, chosen per source with a seeded RNG.
- Every output, identity included, goes through the same pipeline: bounded loader -> random crop 0-3 % -> resize to a
  384 px short side with a random resampler -> random exposure +/-0.25 EV -> render -> JPEG quality 80-95. The
  nuisance draws are independent of the family.
- Output (git-ignored): data/synthetic/film_synth_v1/{images/*.jpg, annotations.jsonl, summary.json}.
  Original photographs are never modified.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from filmlook_infer.imageio import load_image
from filmlook_ml import paths as P
from filmlook_ml.annotation.workflow import discover
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.filmsynth import SyntheticAnnotation, load_presets, sample_nuisance, sample_params, variant_seed
from filmlook_ml.filmsynth.engine import encode_jpeg, prepare, render

OUT = P.DATA / "synthetic" / "film_synth_v1"


def _make(job):
    r, fam, k, cfg = job
    version = cfg["version"]
    seed = variant_seed(r["image_id"], fam, k, version)
    rng = np.random.default_rng(seed)
    params = sample_params(cfg["families"][fam], rng)
    nz = sample_nuisance(cfg, rng)
    img = load_image(r["path"], max_side=1280)
    arr = prepare(img, cfg["working_short_side"], nz["crop_fraction"], nz["resample"], rng)
    out = render(arr, params, seed, exposure_ev=nz["exposure_ev"])
    data = encode_jpeg(out, nz["jpeg_quality"])
    vid = f"{r['family_id']}__{fam}__{k:02d}"
    path = OUT / "images" / f"{vid}.jpg"
    path.write_bytes(data)
    return SyntheticAnnotation(
        source_image_id=r["image_id"], variant_id=vid, augmentation_family=fam, augmentation_version=version,
        parameters=params, nuisance=nz, seed=seed, approximate_references=cfg["families"][fam]["approximate_references"],
        split=r["split"], group_id=r["group_id"], source_license=r.get("license"), source_dataset=r["dataset_id"],
        source_attribution=r.get("attribution"), path=str(path.relative_to(P.ROOT)),
        sha256=hashlib.sha256(data).hexdigest()).model_dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=100)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_presets()
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    feats = json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text())
    colour = [r for r in eligible if not feats[r["sha256"]]["is_mono"]]
    rng = np.random.default_rng(a.seed)
    quota = {"train": round(a.sources * 0.7), "val": round(a.sources * 0.15)}
    quota["test"] = a.sources - quota["train"] - quota["val"]
    chosen = []
    for s, k in quota.items():
        pool = [r for r in colour if r["split"] == s]
        idx = rng.choice(len(pool), size=k, replace=False)
        chosen += [pool[i] for i in sorted(idx)]
    fams = [f for f in cfg["families"] if f != "identity_v1"]
    jobs = []
    for r in chosen:
        srng = np.random.default_rng(variant_seed(r["image_id"], "choose", 0, cfg["version"]))
        pick = ["identity_v1", *srng.choice(fams, size=a.variants - 1, replace=False).tolist()]
        jobs += [(r, f, k, cfg) for k, f in enumerate(pick)]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with Pool(16) as pool:
        anns = pool.map(_make, jobs, chunksize=4)
    with open(OUT / "annotations.jsonl", "w") as fh:
        for x in anns:
            fh.write(json.dumps(x) + "\n")
    summ = {"version": cfg["version"], "sources": len(chosen), "variants": len(anns),
            "sources_by_split": dict(Counter(r["split"] for r in chosen)),
            "families": dict(Counter(x["augmentation_family"] for x in anns)),
            "source_media": dict(Counter(r.get("medium") or "pd12m" for r in chosen))}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
