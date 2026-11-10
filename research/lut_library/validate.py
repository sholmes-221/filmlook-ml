"""Small validator for 3D LUT files in the LUT library (.cube, HaldCLUT .png/.tif).

Checks (all recorded per file, nothing is "fixed"):
  - parse: header / dimensions, domain min/max, comment lines;
  - size: 3D grid edge (N for N^3) or 1D length; HaldCLUT level L -> N = L^2;
  - range: NaN / inf, values outside [domain_min, domain_max], share of clipped entries;
  - monotonic identity check: along the grey diagonal the output luminance (mean of RGB) must be
    non-decreasing (tolerance), and each output channel along its own input axis must on
    average increase (a LUT that inverts or heavily folds the tone curve is flagged);
  - neutrality: output on the grey axis is reported as mean saturation (abs channel spread),
    so a strongly tinted or B&W LUT is visible without being an error;
  - identity distance: RMS distance from the identity LUT (0 means "does nothing").

Usage:
    python -I research/lut_library/validate.py path/to/lut.cube [more files]
    python -I research/lut_library/validate.py --json out.json file...

Only numpy and Pillow are needed. The parser is deliberately strict about what it needs and
lenient about vendor comment lines; anything it does not understand is reported, not guessed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:  # Pillow is optional: without it HaldCLUT images are reported as "not parsed"
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass
class LutReport:
    path: str
    format: str = ""              # cube_3d | cube_1d | haldclut_png | haldclut_tif | unknown
    parsed: bool = False
    error: str | None = None
    title: str | None = None
    size: int | None = None       # grid edge N (3D) or table length (1D)
    hald_level: int | None = None
    domain_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    domain_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    n_entries: int = 0
    n_nan: int = 0
    n_out_of_domain: int = 0
    clip_fraction: float = 0.0    # share of entries at or beyond the domain bounds (after tolerance)
    value_min: float | None = None
    value_max: float | None = None
    grey_luma_monotonic: bool | None = None
    grey_luma_min_step: float | None = None     # most negative luminance step along the grey diagonal
    axis_monotonic: bool | None = None          # each channel increases along its own axis on average
    grey_mean_saturation: float | None = None   # mean |max-min| of RGB along the grey axis
    grey_black: list[float] | None = None
    grey_white: list[float] | None = None
    identity_rms: float | None = None
    is_near_identity: bool | None = None
    header_comments: list[str] = field(default_factory=list)
    ok: bool = False
    warnings: list[str] = field(default_factory=list)


def _parse_cube(path: Path, rep: LutReport) -> np.ndarray | None:
    size_3d = None
    size_1d = None
    data: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if len(rep.header_comments) < 20:
                    rep.header_comments.append(line[1:].strip()[:200])
                continue
            up = line.upper()
            if up.startswith("TITLE"):
                rep.title = line[5:].strip().strip('"')
            elif up.startswith("LUT_3D_SIZE"):
                size_3d = int(line.split()[1])
            elif up.startswith("LUT_1D_SIZE"):
                size_1d = int(line.split()[1])
            elif up.startswith("DOMAIN_MIN"):
                rep.domain_min = [float(x) for x in line.split()[1:4]]
            elif up.startswith("DOMAIN_MAX"):
                rep.domain_max = [float(x) for x in line.split()[1:4]]
            elif up.startswith("LUT_3D_INPUT_RANGE") or up.startswith("LUT_1D_INPUT_RANGE"):
                lo, hi = (float(x) for x in line.split()[1:3])
                rep.domain_min, rep.domain_max = [lo] * 3, [hi] * 3
            else:
                parts = line.replace(",", " ").split()
                try:
                    vals = [float(p) for p in parts[:3]]
                except ValueError:
                    rep.warnings.append(f"unparsed line: {line[:60]}")
                    continue
                if len(vals) == 3:
                    data.append(vals)
    arr = np.asarray(data, dtype=np.float64)
    if size_3d:
        rep.format, rep.size = "cube_3d", size_3d
        if arr.shape[0] != size_3d ** 3:
            rep.error = f"expected {size_3d ** 3} entries, found {arr.shape[0]}"
            return None
        # .cube order: red varies fastest, then green, then blue
        return arr.reshape(size_3d, size_3d, size_3d, 3)  # index [b, g, r]
    if size_1d:
        rep.format, rep.size = "cube_1d", size_1d
        if arr.shape[0] != size_1d:
            rep.error = f"expected {size_1d} entries, found {arr.shape[0]}"
            return None
        return arr
    rep.error = "no LUT_3D_SIZE / LUT_1D_SIZE header"
    return None


def _parse_hald(path: Path, rep: LutReport) -> np.ndarray | None:
    if Image is None:
        rep.error = "Pillow not available"
        return None
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        w, h = im.size
        if w != h:
            rep.error = f"HaldCLUT must be square, got {w}x{h}"
            return None
        level = round(w ** (1.0 / 3.0))
        if level ** 3 != w:
            # some tools pad; try nearby integers
            for cand in (level - 1, level + 1):
                if cand > 0 and cand ** 3 == w:
                    level = cand
                    break
            else:
                rep.error = f"width {w} is not a perfect cube (not a HaldCLUT)"
                return None
        n = level * level
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float64)
            if arr.ndim == 2:  # greyscale 16-bit
                arr = np.repeat(arr[..., None], 3, axis=-1)
            scale = 65535.0
        else:
            rgb = im.convert("RGB") if mode != "RGB" else im
            arr = np.asarray(rgb, dtype=np.float64)
            scale = 255.0
            if mode not in ("RGB", "RGBA", "L", "P"):
                rep.warnings.append(f"image mode {mode} converted to RGB")
        rep.format = "haldclut_" + (path.suffix.lower().lstrip(".") or "img")
        rep.hald_level, rep.size = level, n
        flat = arr.reshape(-1, 3)[:, :3] / scale
        # HaldCLUT order: red fastest, then green, then blue (same as .cube)
        return flat.reshape(n, n, n, 3)


def _analyse(lut: np.ndarray, rep: LutReport) -> None:
    dmin = np.asarray(rep.domain_min, dtype=np.float64)
    dmax = np.asarray(rep.domain_max, dtype=np.float64)
    flat = lut.reshape(-1, 3)
    rep.n_entries = int(flat.shape[0])
    finite = np.isfinite(flat)
    rep.n_nan = int((~finite).sum())
    vals = np.where(finite, flat, 0.0)
    rep.value_min = float(vals.min())
    rep.value_max = float(vals.max())
    tol = 1e-6
    below = vals < (dmin - tol)
    above = vals > (dmax + tol)
    rep.n_out_of_domain = int((below | above).sum())
    at_bounds = (vals <= dmin + 1e-4) | (vals >= dmax - 1e-4)
    rep.clip_fraction = float(at_bounds.mean())

    if lut.ndim == 4:
        n = lut.shape[0]
        idx = np.arange(n)
        grey = lut[idx, idx, idx, :]                       # [b=g=r=i]
        luma = grey.mean(axis=1)
        steps = np.diff(luma)
        rep.grey_luma_min_step = float(steps.min()) if steps.size else 0.0
        # tolerance: 8-bit HaldCLUTs carry +-1 code of quantisation noise (1/255 = 0.0039); a dip larger
        # than ~2.5 codes (0.01 of the range) is a real fold in the tone curve, smaller dips are noise
        rep.grey_luma_monotonic = bool((steps >= -1e-2 * (dmax - dmin).mean()).all())
        rep.grey_mean_saturation = float((grey.max(axis=1) - grey.min(axis=1)).mean())
        rep.grey_black = [float(x) for x in grey[0]]
        rep.grey_white = [float(x) for x in grey[-1]]
        # average slope of each output channel along its own input axis
        r_slope = np.diff(lut[..., 0], axis=2).mean()
        g_slope = np.diff(lut[..., 1], axis=1).mean()
        b_slope = np.diff(lut[..., 2], axis=0).mean()
        rep.axis_monotonic = bool(min(r_slope, g_slope, b_slope) > 0)
        # identity distance
        ramp = dmin[None] + (dmax - dmin)[None] * (idx / (n - 1))[:, None]
        ident = np.stack(np.meshgrid(ramp[:, 2], ramp[:, 1], ramp[:, 0], indexing="ij"), axis=-1)[..., ::-1]
        rep.identity_rms = float(np.sqrt(((lut - ident) ** 2).mean()))
        rep.is_near_identity = rep.identity_rms < 1e-3
    else:  # 1D
        n = lut.shape[0]
        steps = np.diff(lut.mean(axis=1))
        rep.grey_luma_min_step = float(steps.min()) if steps.size else 0.0
        rep.grey_luma_monotonic = bool((steps >= -1e-3).all())
        rep.axis_monotonic = bool((np.diff(lut, axis=0).mean(axis=0) > 0).all())
        rep.grey_mean_saturation = float((lut.max(axis=1) - lut.min(axis=1)).mean())
        rep.grey_black = [float(x) for x in lut[0]]
        rep.grey_white = [float(x) for x in lut[-1]]
        ramp = np.linspace(0, 1, n)[:, None] * (dmax - dmin)[None] + dmin[None]
        rep.identity_rms = float(np.sqrt(((lut - ramp) ** 2).mean()))
        rep.is_near_identity = rep.identity_rms < 1e-3

    if rep.n_nan:
        rep.warnings.append(f"{rep.n_nan} non-finite entries")
    if rep.n_out_of_domain:
        rep.warnings.append(f"{rep.n_out_of_domain} entries outside the declared domain")
    if rep.clip_fraction > 0.10:
        rep.warnings.append(f"{rep.clip_fraction:.1%} of entries sit at the domain bounds (heavy clipping)")
    if rep.grey_luma_monotonic is False:
        rep.warnings.append(f"grey-axis luminance not monotonic (min step {rep.grey_luma_min_step:.4f})")
    if rep.axis_monotonic is False:
        rep.warnings.append("a channel decreases on average along its own input axis")
    if rep.is_near_identity:
        rep.warnings.append("LUT is (near) identity")
    rep.ok = rep.n_nan == 0 and rep.n_out_of_domain == 0 and bool(rep.grey_luma_monotonic) and bool(rep.axis_monotonic)


def validate(path: str | Path) -> LutReport:
    p = Path(path)
    rep = LutReport(path=str(p))
    try:
        suf = p.suffix.lower()
        if suf == ".cube":
            lut = _parse_cube(p, rep)
        elif suf in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            lut = _parse_hald(p, rep)
        else:
            rep.format = "unknown"
            rep.error = f"unsupported extension {suf}"
            lut = None
        if lut is not None:
            rep.parsed = True
            _analyse(lut, rep)
    except Exception as exc:  # report, never raise: the manifest records the failure
        rep.error = f"{type(exc).__name__}: {exc}"
        rep.parsed = False
        rep.ok = False
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", help="write all reports to this JSON file")
    args = ap.parse_args(argv)
    reports = [validate(f) for f in args.files]
    for r in reports:
        status = "OK " if r.ok else ("ERR" if not r.parsed else "WARN")
        extra = r.error or "; ".join(r.warnings)
        size = f"{r.size}^3" if r.format.startswith(("cube_3d", "haldclut")) else str(r.size)
        print(f"{status} {r.format:13s} {size:6s} id_rms={r.identity_rms if r.identity_rms is not None else float('nan'):.4f} {r.path} {extra}")
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in reports], indent=1))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
