"""LUT feature database: build (once, offline) and load (at inference; nothing is recomputed per upload).

Build (needs the research code and the owner's presets; writes only derived numbers, never the XMP files):
    python -m research.lut.engine.lut_db build --presets data/lut/presets.json [--cube-dir DIR] --out research/lut/db

Files written to the db directory:
  lut_db.json         per style: id, display name, source (xmp|cube), approximation note, descriptors (RGB transfer,
                      CIELAB transformation, hue shift per region, saturation / luminance response, shadow / midtone /
                      highlight colour bias, midtone contrast, channel interactions, tone curves, gamut compression,
                      skin-tone shift, clipping), explanation phrases, grain parameters (not representable by a LUT);
                      reference-scene list (attributions), feature names, build metadata, validation reports
  lut_db_arrays.npz   lattices [S,33,33,33,3] float16, inverse lattices + residual (for plausibility scoring),
                      reference bank [R, S+1, D] (descriptors of R licensed digital reference scenes rendered through
                      every style and untouched), bank_neutral [R, S+1, D] (the same renders after the engine's
                      neutralisation, for scene-conditioned delta matching), natural-pixel-cloud signatures [S+1, D],
                      Fisher weights [D], bank mean / std [D]
Reference grid: the 33^3 lattice over sRGB [0, 1] (input domain), descriptors evaluated on a 17^3 sub-grid, greys on
a 33-step ramp, skin tones on a fixed CIELAB patch set. All colour maths in sRGB (D65) -> CIELAB (D50, Bradford).
Precision: float32 internally, float16 storage for lattices (max abs error 2.4e-4 in [0, 1], below 1/255).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.lut.engine.descriptors import COLOUR_IDX, DIM, NAMES, TONE_IDX, describe, describe_pixels, neutralise, srgb_to_lab
from research.lut.engine.lut3d import LUT3D

LATTICE = 33
MATCH_IDX = np.concatenate([COLOUR_IDX, TONE_IDX])

APPROX_NOTE_XMP = ("Approximate re-implementation of a Lightroom / Camera Raw preset (process version 15.4): point curves, HSL, "
                   "colour grading, point colour and dehaze as documented in research/lut/forward_model.py; not fitted to Lightroom "
                   "output; grain, clarity, texture and sharpening are not colour transforms and are not in the LUT.")
APPROX_NOTE_CUBE = "Exact 3D LUT lattice read from a .cube file (trilinear interpolation between lattice points)."


# ---------------------------------------------------------------------------------------------------------------
# style descriptors from a lattice
# ---------------------------------------------------------------------------------------------------------------
def _grid(n):
    g = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def _hue(lab):
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def _dhue(h_out, h_in):
    return ((h_out - h_in + 180) % 360) - 180


def style_descriptors(lut: LUT3D, resid: np.ndarray | None = None) -> dict:
    g17 = _grid(17)
    out17 = lut.apply(g17)
    lab_in, lab_out = srgb_to_lab(g17), srgb_to_lab(out17)
    c_in, c_out = np.hypot(lab_in[:, 1], lab_in[:, 2]), np.hypot(lab_out[:, 1], lab_out[:, 2])
    ramp_in = np.linspace(0, 1, LATTICE, dtype=np.float32)
    ramp = lut.grey_ramp(LATTICE)
    lab_ramp = srgb_to_lab(ramp)
    lab_grey_in = srgb_to_lab(np.stack([ramp_in] * 3, -1))
    d = {}
    d["rgb_transfer"] = {"grey_input": ramp_in.round(4).tolist(), "grey_output_rgb": ramp.round(4).tolist(),
                         "channel_gain_at_mid": (ramp[LATTICE // 2] / max(ramp_in[LATTICE // 2], 1e-6)).round(4).tolist(),
                         "monotone": bool((np.diff(ramp, axis=0) >= -1e-4).all())}
    bands = [(0, 35), (35, 65), (65, 101)]
    d["lab_transformation"] = {"mean_dL": float((lab_out[:, 0] - lab_in[:, 0]).mean()), "mean_da": float((lab_out[:, 1] - lab_in[:, 1]).mean()),
                               "mean_db": float((lab_out[:, 2] - lab_in[:, 2]).mean()),
                               "by_lightness_band": [{"L_in": list(b), "dL": float((lab_out[m, 0] - lab_in[m, 0]).mean()), "da": float((lab_out[m, 1] - lab_in[m, 1]).mean()),
                                                      "db": float((lab_out[m, 2] - lab_in[m, 2]).mean()), "chroma_ratio": float(c_out[m].sum() / max(c_in[m].sum(), 1e-6))}
                                                     for b in bands for m in [(lab_in[:, 0] >= b[0]) & (lab_in[:, 0] < b[1])]]}
    chrom = (c_in > 15) & (lab_in[:, 0] > 15) & (lab_in[:, 0] < 90)
    h_in, h_out = _hue(lab_in), _hue(lab_out)
    regions = []
    for k in range(12):
        m = chrom & (h_in >= 30 * k) & (h_in < 30 * (k + 1))
        regions.append({"hue_in_deg": [30 * k, 30 * (k + 1)], "n": int(m.sum()),
                        "mean_dhue_deg": float(_dhue(h_out[m], h_in[m]).mean()) if m.any() else 0.0,
                        "chroma_ratio": float(c_out[m].sum() / max(c_in[m].sum(), 1e-6)) if m.any() else 1.0,
                        "dL": float((lab_out[m, 0] - lab_in[m, 0]).mean()) if m.any() else 0.0})
    d["hue_shift_by_region"] = regions
    edges = [0, 10, 25, 45, 70, 200]
    d["saturation_response"] = [{"chroma_in": [edges[i], edges[i + 1]], "chroma_ratio": float(c_out[m].sum() / max(c_in[m].sum(), 1e-6)) if m.any() else 1.0}
                                for i in range(5) for m in [(c_in >= edges[i]) & (c_in < edges[i + 1]) & (lab_in[:, 0] > 10) & (lab_in[:, 0] < 95)]]
    Lg_in, Lg_out = lab_grey_in[:, 0], lab_ramp[:, 0]
    d["luminance_response"] = {"L_in": Lg_in.round(2).tolist(), "L_out": Lg_out.round(2).tolist(),
                               "midtone_contrast": float(np.interp(55, Lg_in, Lg_out) - np.interp(45, Lg_in, Lg_out)) / 10.0,
                               "shadow_slope": float(np.interp(25, Lg_in, Lg_out) - np.interp(10, Lg_in, Lg_out)) / 15.0,
                               "highlight_slope": float(np.interp(95, Lg_in, Lg_out) - np.interp(80, Lg_in, Lg_out)) / 15.0}
    for key, (lo, hi) in (("shadow_colour_bias", (3, 30)), ("midtone_colour_bias", (30, 70)), ("highlight_colour_bias", (70, 98))):
        m = (Lg_in >= lo) & (Lg_in < hi)
        ab = lab_ramp[m, 1:].mean(0) if m.any() else np.zeros(2)
        d[key] = {"a": float(ab[0]), "b": float(ab[1]), "chroma": float(np.hypot(*ab)), "hue_deg": float(np.degrees(np.arctan2(ab[1], ab[0])) % 360)}
    X = np.concatenate([g17, np.ones((len(g17), 1), np.float32)], 1)
    A, *_ = np.linalg.lstsq(X, out17, rcond=None)
    M = A[:3].T
    d["channel_interaction"] = {"linear_fit_matrix": M.round(4).tolist(), "offset": A[3].round(4).tolist(),
                                "offdiag_ratio": float(np.abs(M - np.diag(np.diag(M))).sum() / max(np.abs(np.diag(M)).sum(), 1e-6)),
                                "fit_rms": float(np.sqrt(((X @ A - out17) ** 2).mean()))}
    d["gamut_compression"] = {"chroma_p95_ratio": float(np.percentile(c_out, 95) / max(np.percentile(c_in, 95), 1e-6)),
                              "chroma_mean_ratio": float(c_out.mean() / max(c_in.mean(), 1e-6)),
                              "L_range_ratio": float((lab_out[:, 0].max() - lab_out[:, 0].min()) / 100)}
    if resid is not None:
        d["gamut_compression"]["unreachable_cube_fraction_de2"] = float((resid > 2).mean())
    skin = []
    for L in (45, 55, 65, 75):
        for h in (25, 40, 55):
            for c in (18, 30):
                skin.append([L, c * np.cos(np.radians(h)), c * np.sin(np.radians(h))])
    skin = np.array(skin, np.float32)
    from research.lut.engine.descriptors import _M_SRGB_TO_XYZ_D50, _WHITE_D50
    fy = (skin[:, 0] + 16) / 116
    f = np.stack([fy + skin[:, 1] / 500, fy, fy - skin[:, 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * _WHITE_D50
    lin = np.clip(xyz @ np.linalg.inv(_M_SRGB_TO_XYZ_D50).T, 0, 1)
    skin_rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)
    s_in, s_out = srgb_to_lab(skin_rgb), srgb_to_lab(lut.apply(skin_rgb))
    d["skin_tone_shift"] = {"n_patches": int(len(skin)), "mean_dhue_deg": float(_dhue(_hue(s_out), _hue(s_in)).mean()),
                            "chroma_ratio": float(np.hypot(s_out[:, 1], s_out[:, 2]).mean() / max(np.hypot(s_in[:, 1], s_in[:, 2]).mean(), 1e-6)),
                            "dL": float((s_out[:, 0] - s_in[:, 0]).mean()), "da": float((s_out[:, 1] - s_in[:, 1]).mean()), "db": float((s_out[:, 2] - s_in[:, 2]).mean())}
    d["clipping"] = {"black_output_rgb": ramp[0].round(4).tolist(), "white_output_rgb": ramp[-1].round(4).tolist(),
                     "black_lift_8bit": float(ramp[0].mean() * 255), "white_8bit": float(ramp[-1].mean() * 255),
                     "grid_fraction_clipped_low": float((out17 <= 1e-4).any(1).mean()), "grid_fraction_clipped_high": float((out17 >= 1 - 1e-4).any(1).mean())}
    return d


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------
def reference_scenes(n_ref: int) -> list[dict]:
    """Deterministic licensed digital reference scenes from the TRAIN split of the synthetic generator (held-out scenes stay out)."""
    from research.lut.synthetic import base_rows, split_of
    rows = [r for r in base_rows() if split_of(r["group_id"]) == "train"]
    rows.sort(key=lambda r: hashlib.sha1(r["source_id"].encode()).hexdigest())
    return rows[:n_ref]


class _LatticePreset:
    """Preset-like wrapper (``.apply``) around a lattice, for the grid inverse of .cube / HaldCLUT styles."""

    def __init__(self, lut: LUT3D):
        self.lut = lut

    def apply(self, x):
        return self.lut.apply(x).astype(np.float64)


def build_from_styles(styles: list[dict], luts: list[LUT3D], out_dir: str, n_ref: int = 32, with_inverse: bool = True,
                      natural_pixels="data/lut/natural_pixels.npy", presets_by_id: dict | None = None, extra_meta: dict | None = None,
                      lattice_store: int = LATTICE) -> dict:
    """Generic database builder: `styles` (metadata dicts with id/name/source/approximation/...) and matching `luts`.
    Writes lut_db.json + lut_db_arrays.npz to out_dir; returns the metadata. Inverse lattices (approach B) are optional;
    `lattice_store` controls the stored lattice resolution (descriptors always use the 33^3 working lattice)."""
    from research.lut.inverse_analysis import GridInverse
    from research.lut.synthetic import load_base
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    presets_by_id = presets_by_id or {}
    validations = [l.validate() for l in luts]
    keep = [i for i, v in enumerate(validations) if v["ok"]]
    styles = [styles[i] for i in keep]
    luts = [luts[i] for i in keep]
    S = len(styles)
    print(f"{S} valid styles ({len(validations) - S} rejected by validation)")
    cube = _grid(LATTICE).astype(np.float64)
    inv_lat = np.zeros((S, LATTICE, LATTICE, LATTICE, 3), np.float32) if with_inverse else None
    inv_res = np.zeros((S, LATTICE, LATTICE, LATTICE), np.float32) if with_inverse else None
    for i, (st, lut) in enumerate(zip(styles, luts)):
        resid = None
        if with_inverse:
            p = presets_by_id.get(st["id"]) or _LatticePreset(lut)
            x, r = GridInverse(p)(cube)
            inv_lat[i] = x.reshape(LATTICE, LATTICE, LATTICE, 3)
            inv_res[i] = r.reshape(LATTICE, LATTICE, LATTICE)
            resid = inv_res[i]
        st["descriptors"] = style_descriptors(lut, resid)
        if i % 25 == 0 or i == S - 1:
            print(f"  descriptors {i + 1}/{S} ({time.time() - t0:.0f}s)")
    nat = np.load(natural_pixels)
    cloud = np.stack([describe_pixels(l.apply(nat)) for l in luts] + [describe_pixels(nat)])
    refs = reference_scenes(n_ref)
    bank = np.zeros((len(refs), S + 1, DIM), np.float32)
    bank_neutral = np.zeros_like(bank)        # descriptors of neutralise(render): the same estimate a photo gets (approach E)
    for k, r in enumerate(refs):
        base = load_base(r["path"], short=384)
        renders = [l.apply_u8(base) for l in luts] + [base]
        for i, im in enumerate(renders):
            bank[k, i] = describe(im)
            bank_neutral[k, i] = describe(neutralise(im), with_texture=False)
        if k % 8 == 0:
            print(f"  reference scene {k + 1}/{len(refs)} ({time.time() - t0:.0f}s)")
    mean = bank.reshape(-1, DIM).mean(0)
    std = bank.reshape(-1, DIM).std(0) + 1e-6
    Z = (bank - mean) / std
    between = Z.mean(0).var(0)              # variance of style means (over scenes)
    within = Z.var(0).mean(0) + 1e-6        # within-style variance across scenes
    fisher = between / within
    fisher[np.setdiff1d(np.arange(DIM), MATCH_IDX)] = 0.0
    stored = [l if lattice_store == LATTICE else l.resample(lattice_store) for l in luts]
    arrays = {"lattices": np.stack([l.lattice for l in stored]).astype(np.float16), "bank": bank, "bank_neutral": bank_neutral,
              "cloud_signatures": cloud.astype(np.float32), "fisher_weights": fisher.astype(np.float32),
              "bank_mean": mean.astype(np.float32), "bank_std": std.astype(np.float32)}
    if with_inverse:
        arrays["inverse_lattices"] = inv_lat.astype(np.float16)
        arrays["inverse_residual"] = inv_res.astype(np.float16)
    np.savez(out / "lut_db_arrays.npz", **arrays)
    meta = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "lattice_size": LATTICE, "stored_lattice_size": lattice_store, "has_inverse": with_inverse,
            "descriptor_dim": DIM, "descriptor_names": NAMES, "match_feature_indices": MATCH_IDX.tolist(), "n_styles": S, "styles": styles,
            "validation": [validations[i] for i in keep], "rejected": [v for i, v in enumerate(validations) if i not in keep],
            "reference_scenes": [{"source_id": r["source_id"], "attribution": r["attribution"], "license": r["license"], "dataset": r["dataset_id"]} for r in refs],
            "natural_pixels": {"file": natural_pixels, "n": int(len(nat))},
            "colour_space": "input/output sRGB (IEC 61966-2-1, D65) in [0,1]; CIELAB with D50 white after Bradford adaptation",
            "storage_precision": "float16 lattices (max abs error 2.4e-4), float32 descriptors", "build_seconds": round(time.time() - t0, 1),
            **(extra_meta or {})}
    (out / "lut_db.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"wrote {out}/lut_db.json and lut_db_arrays.npz in {time.time() - t0:.0f}s; sizes: json {(out / 'lut_db.json').stat().st_size / 1e6:.2f} MB, "
          f"npz {(out / 'lut_db_arrays.npz').stat().st_size / 1e6:.2f} MB")
    return meta


def preset_styles(presets_json: str):
    """The owner's 13 XMP presets as (styles, luts, presets_by_id)."""
    from research.lut.explain import describe as explain_preset
    from research.lut.forward_model import load_presets
    styles, luts, presets_by_id = [], [], {}
    for name, p in load_presets(presets_json).items():
        sid = name.replace("'", "").lower()
        luts.append(LUT3D(p.lattice(LATTICE), name, notes="xmp"))
        presets_by_id[sid] = p
        styles.append({"id": sid, "name": name.replace("_", " "), "source": "xmp", "license": "owner's private preset (not redistributed)",
                       "approximation": APPROX_NOTE_XMP, "label_basis": "LUT-derived aesthetic",
                       "tone_curves": {"main": p.main_curve, "red": p.rgb_curves["Red"], "green": p.rgb_curves["Green"], "blue": p.rgb_curves["Blue"]},
                       "grain_parameters_not_in_lut": p.grain, "explanation_phrases": explain_preset(p)})
    return styles, luts, presets_by_id


def build(presets_json: str, out_dir: str, cube_dir: str | None = None, n_ref: int = 32, natural_pixels="data/lut/natural_pixels.npy") -> None:
    """The vendored 13-preset database (plus any .cube files found in cube_dir)."""
    styles, luts, presets_by_id = preset_styles(presets_json)
    if cube_dir and Path(cube_dir).exists():
        for f in sorted(Path(cube_dir).glob("*.cube")):
            try:
                lut = LUT3D.from_cube(f).resample(LATTICE)
            except ValueError as e:
                print("skip", f, e)
                continue
            luts.append(lut)
            styles.append({"id": f.stem.lower().replace(" ", "_"), "name": f.stem, "source": "cube", "approximation": APPROX_NOTE_CUBE,
                           "label_basis": "LUT-derived aesthetic", "tone_curves": None, "grain_parameters_not_in_lut": {}, "explanation_phrases": []})
    build_from_styles(styles, luts, out_dir, n_ref=n_ref, with_inverse=True, natural_pixels=natural_pixels, presets_by_id=presets_by_id)


# ---------------------------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------------------------
class LutDB:
    def __init__(self, db_dir):
        self.dir = Path(db_dir)
        self.meta = json.loads((self.dir / "lut_db.json").read_text())
        z = np.load(self.dir / "lut_db_arrays.npz")
        self.lattices = z["lattices"]
        self.stored_lattice_size = int(self.lattices.shape[1])
        self.has_inverse = "inverse_lattices" in z.files
        self.inverse_lattices = z["inverse_lattices"] if self.has_inverse else None
        self.inverse_residual = z["inverse_residual"] if self.has_inverse else None
        self.bank = z["bank"]
        self.bank_neutral = z["bank_neutral"]
        self.cloud = z["cloud_signatures"]
        self.fisher = z["fisher_weights"]
        self.mean, self.std = z["bank_mean"], z["bank_std"]
        self.styles = self.meta["styles"]
        self.ids = [s["id"] for s in self.styles] + ["none"]
        self.match_idx = np.array(self.meta["match_feature_indices"])
        self._luts: dict[int, LUT3D] = {}

    @property
    def n_styles(self) -> int:
        return len(self.styles)

    def lut(self, i: int) -> LUT3D:
        if i not in self._luts:
            self._luts[i] = LUT3D(self.lattices[i].astype(np.float32), self.styles[i]["name"])
        return self._luts[i]

    def inverse(self, i: int) -> tuple[LUT3D, np.ndarray]:
        if not self.has_inverse:
            raise ValueError("this database was built without inverse lattices (approach B unavailable)")
        return LUT3D(self.inverse_lattices[i].astype(np.float32)), self.inverse_residual[i].astype(np.float32)

    def style(self, sid: str) -> dict:
        return next(s for s in self.styles if s["id"] == sid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--presets", default="data/lut/presets.json")
    b.add_argument("--cube-dir", default="/home/shanmo/misc/lut")
    b.add_argument("--out", default="research/lut/db")
    b.add_argument("--n-ref", type=int, default=32)
    s = sub.add_parser("show")
    s.add_argument("--db", default="research/lut/db")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.presets, a.out, a.cube_dir, a.n_ref)
    else:
        db = LutDB(a.db)
        for st in db.styles:
            d = st["descriptors"]
            print(f"{st['id']:22} black {d['clipping']['black_lift_8bit']:5.1f} white {d['clipping']['white_8bit']:5.1f} midC {d['luminance_response']['midtone_contrast']:.2f} "
                  f"shadow hue {d['shadow_colour_bias']['hue_deg']:5.0f} c {d['shadow_colour_bias']['chroma']:4.1f} | mid hue {d['midtone_colour_bias']['hue_deg']:5.0f} c {d['midtone_colour_bias']['chroma']:4.1f} "
                  f"| skin dhue {d['skin_tone_shift']['mean_dhue_deg']:+5.1f} chroma x{d['skin_tone_shift']['chroma_ratio']:.2f} | sat x{d['gamut_compression']['chroma_mean_ratio']:.2f} "
                  f"| unreachable {d['gamut_compression']['unreachable_cube_fraction_de2']:.2f}")


if __name__ == "__main__":
    main()
