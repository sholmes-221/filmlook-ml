#!/usr/bin/env python
"""E14: is diffusion<->diffusion transfer a shared VAE/diffusion artifact rather than film treatment?

  python scripts/artifact_probe.py    # -> reports/artifact_probe.json

Probe A (treatment): train SDXL treated vs SDXL VAE control (train sources); test on Z-Image treated vs Z-Image VAE
control (test sources) = the pairwise LOGO sdxl->zimage.
Probe B (artifact): same training; test on Z-Image VAE CONTROL vs raw standardised sources (test sources). If the
probe scores the VAE control as "treated" (AUROC >> 0.5), the learned direction is "went through a diffusion VAE",
not a film treatment. Also: raw sources vs SDXL VAE control (train on filmsynth treated vs identity, to see whether
the filmsynth direction reacts to VAE smoothing at all). All with source bootstrap CIs; hybrid and explicit features.
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import synth_experiment_v3 as V3  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.annotation.workflow import discover  # noqa: E402
from filmlook_ml.datasets.preprocess import read_jsonl  # noqa: E402
from filmlook_ml.datasets.quarantine import load_quarantine  # noqa: E402
from filmlook_ml.datasets.registry import Registry  # noqa: E402


def main():
    t0 = time.time()
    dev = torch.device("cuda")
    lcfg = P.load_yaml("labeling.yaml")
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, _ = discover(man, Registry.load(), set(lcfg["claude"]["consented_datasets"]), load_quarantine())
    by_id = {r["image_id"]: r for r in eligible}
    fs = read_jsonl(P.DATA / "synthetic" / "film_synth_v2" / "annotations.jsonl")
    sdx = read_jsonl(P.DATA / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl")
    zim = read_jsonl(P.DATA / "synthetic" / "zimage_i2i_v1" / "annotations.jsonl")
    srcs = sorted({a["source_image_id"] for a in fs} & {a["source_image_id"] for a in sdx} & {a["source_image_id"] for a in zim})
    sp = {s: by_id[s]["split"] for s in srcs}
    pick = lambda rows, cond: [a for a in rows if a["source_image_id"] in sp and cond(a)]  # noqa: E731
    sets = {"fs_t": pick(fs, lambda a: a["augmentation_family"] != "identity_v2"), "fs_c": pick(fs, lambda a: a["augmentation_family"] == "identity_v2"),
            "sdx_t": pick(sdx, lambda a: a.get("treated", True)), "sdx_c": pick(sdx, lambda a: a.get("treated", True) is False),
            "zim_t": pick(zim, lambda a: a.get("treated", True)), "zim_c": pick(zim, lambda a: a.get("treated", True) is False)}
    feats = V3.Feats(dev)
    F = {k: feats([str(P.ROOT / a["path"]) for a in v], [a["variant_id"] for a in v], True) for k, v in sets.items()}
    F["raw"] = feats([by_id[s]["path"] for s in srcs], srcs, True, real=True)
    S = {k: np.array([a["source_image_id"] for a in v]) for k, v in sets.items()}; S["raw"] = np.array(srcs)
    SP = {k: np.array([sp[s] for s in v]) for k, v in S.items()}
    rep = {"n": {k: len(v) for k, v in sets.items()}, "probes": {}}
    for name in ("hybrid_style", "explicit"):
        def xy(pos, neg, split):
            X = np.vstack([F[pos][name][SP[pos] == split], F[neg][name][SP[neg] == split]])
            y = np.r_[np.ones((SP[pos] == split).sum()), np.zeros((SP[neg] == split).sum())].astype(int)
            s = np.r_[S[pos][SP[pos] == split], S[neg][SP[neg] == split]]
            return X, y, s
        out = {}
        Xtr, ytr, _ = xy("sdx_t", "sdx_c", "train")
        out["A_sdxl_treated_vs_vae__on_zimage_treated_vs_vae"], _ = V3.probe_auroc(Xtr, ytr, *xy("zim_t", "zim_c", "test"))
        out["B_sdxl_treated_vs_vae__on_zimage_VAE_control_vs_raw"], _ = V3.probe_auroc(Xtr, ytr, *xy("zim_c", "raw", "test"))
        out["B2_sdxl_treated_vs_vae__on_sdxl_VAE_control_vs_raw"], _ = V3.probe_auroc(Xtr, ytr, *xy("sdx_c", "raw", "test"))
        Xtr2, ytr2, _ = xy("zim_t", "zim_c", "train")
        out["C_zimage_treated_vs_vae__on_sdxl_VAE_control_vs_raw"], _ = V3.probe_auroc(Xtr2, ytr2, *xy("sdx_c", "raw", "test"))
        Xtr3, ytr3, _ = xy("fs_t", "fs_c", "train")
        out["D_filmsynth_treated_vs_identity__on_zimage_VAE_control_vs_raw"], _ = V3.probe_auroc(Xtr3, ytr3, *xy("zim_c", "raw", "test"))
        out["E_raw_vs_VAE_controls_separable"], _ = V3.probe_auroc(*xy("sdx_c", "raw", "train")[:2], *xy("zim_c", "raw", "test"))
        rep["probes"][name] = out
        print(name, json.dumps({k: (v["auroc"], v["ci95"]) for k, v in out.items()}), file=sys.stderr)
    h = rep["probes"]["hybrid_style"]
    rep["reading"] = ("B >> 0.5 means the SDXL 'treated' direction fires on a VAE round-trip with no treatment: artifact. "
                      "Compare A (true treatment transfer) with B; if B >= A the diffusion<->diffusion transfer is mostly artifact.")
    rep["summary"] = {"A_treatment_transfer": h["A_sdxl_treated_vs_vae__on_zimage_treated_vs_vae"]["auroc"],
                      "B_artifact_fires_on_vae_control": h["B_sdxl_treated_vs_vae__on_zimage_VAE_control_vs_raw"]["auroc"],
                      "E_vae_vs_raw_separable": h["E_raw_vs_VAE_controls_separable"]["auroc"]}
    rep["seconds"] = round(time.time() - t0, 1)
    (P.REPORTS / "artifact_probe.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["summary"], indent=1))


if __name__ == "__main__":
    main()
