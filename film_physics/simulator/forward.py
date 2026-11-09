"""A minimal, independent forward film model on spektrafilm's CC BY-SA profile data (no spektrafilm code).

    I = F(S, θ_film, θ_process, θ_scan)

scene sRGB → smooth spectrum (3 Gaussian bases fitted to CIE XYZ) → film exposure through the stock's measured spectral
sensitivities under a 5500 K blackbody → characteristic (H&D) curves → dye transmission spectrum (channel dye densities +
base) → negatives: a three-band scanner and a per-channel log inversion (a lab scan), slides: the transmission viewed
under D50-ish light → sRGB.

Approximations, all deliberate and stated:
- spectral upsampling by three smooth bases (not a measured scene spectrum); negative coefficients clipped;
- illuminants are blackbodies, not the CIE daylight series; CMFs are the Wyman, Sloan & Shirley (2013) analytic fit;
- no grain, halation, couplers (inter-layer effects) or print paper; the scan is a generic three-band scanner with an
  automatic per-channel inversion anchored on mid grey, which is what makes lab scans of one negative look alike;
- exposure is relative: an 18 % grey reflector maps to log H = 0 (+ ev · log10 2).
Measured inputs per stock: `log_sensitivity`, `density_curves` over `log_exposure`, `channel_density`, `base_density`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WL = np.arange(380.0, 781.0, 5.0)  # The profiles' 81 wavelengths.
PROFILES = Path(__file__).resolve().parents[1] / "profiles" / "spektrafilm"


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(wl: np.ndarray = WL) -> np.ndarray:
    """CIE 1931 2° colour matching functions, Wyman-Sloan-Shirley (2013) multi-lobe fit. Shape (n, 3)."""
    x = 1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7) - 0.065 * _g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=1)


def blackbody(t: float, wl: np.ndarray = WL) -> np.ndarray:
    lam = wl * 1e-9
    b = 1.0 / (lam ** 5 * (np.exp(1.4388e-2 / (lam * t)) - 1.0))
    return b / b[np.argmin(np.abs(wl - 560))]


SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
BASES = np.stack([np.exp(-0.5 * ((WL - mu) / s) ** 2) for mu, s in ((455, 22), (545, 30), (615, 32))], axis=1)


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


@dataclass
class Stock:
    name: str
    positive: bool
    sens: np.ndarray  # (81, 3) linear spectral sensitivity
    log_exposure: np.ndarray  # (k,)
    curves: np.ndarray  # (k, 3) density above base
    dyes: np.ndarray  # (81, 3) spectral density of each dye layer at unit density
    base: np.ndarray  # (81,) base density

    @classmethod
    def load(cls, stock: str) -> "Stock":
        d = json.loads((PROFILES / f"{stock}.json").read_text())["data"]
        nan0 = lambda a: np.nan_to_num(np.array(a, dtype=float), nan=0.0)  # noqa: E731
        sens = 10 ** np.array([[(-20 if v is None else v) for v in row] for row in d["log_sensitivity"]], dtype=float)
        dyes = nan0([[(0.0 if v is None else v) for v in row] for row in d["channel_density"]])
        base = nan0([(0.0 if v is None else v) for v in (d.get("base_density") or [0.0] * len(WL))])
        curves = np.array(d["density_curves"], dtype=float)
        positive = curves[-1].mean() < curves[0].mean()
        return cls(stock, bool(positive), sens, np.array(d["log_exposure"], dtype=float), curves, dyes, base)


_ILLUM = blackbody(5500.0)
_VIEW = blackbody(5000.0)
_CMF = cmf()
_BASIS_XYZ = (_CMF * _VIEW[:, None]).T @ BASES  # XYZ of each basis spectrum as a reflectance under the viewing light.
_BASIS_XYZ /= (_CMF[:, 1] * _VIEW).sum()
_SCANNER = np.stack([np.exp(-0.5 * ((WL - mu) / 18) ** 2) for mu in (640, 545, 455)], axis=1)  # R, G, B: the cyan, magenta, yellow layers


def scene_spectra(rgb: np.ndarray) -> np.ndarray:
    """sRGB (..., 3) in [0, 1] → reflectance-like spectra (..., 81) reproducing the colour under the viewing light."""
    xyz = srgb_to_linear(rgb) @ SRGB_TO_XYZ.T
    coeff = xyz @ np.linalg.inv(_BASIS_XYZ).T
    return np.clip(coeff, 0, None) @ BASES.T


def simulate(rgb: np.ndarray, stock: Stock, ev: float = 0.0) -> np.ndarray:
    """One photograph through one stock. rgb: (h, w, 3) floats in [0, 1]; returns sRGB floats in [0, 1]."""
    spectra = scene_spectra(rgb)
    grey = np.full(len(WL), 0.184)
    exposure = (spectra * _ILLUM) @ stock.sens  # (h, w, 3) relative exposure per layer
    ref = (grey * _ILLUM) @ stock.sens
    log_h = np.log10(np.maximum(exposure / ref, 1e-6)) + ev * np.log10(2.0)
    dens = np.stack([np.interp(log_h[..., c], stock.log_exposure, stock.curves[:, c]) for c in range(3)], axis=-1)
    spectral_density = dens @ stock.dyes.T + stock.base  # (h, w, 81)
    trans = 10 ** (-spectral_density)
    if stock.positive:
        xyz = (trans * _VIEW) @ _CMF / (_CMF[:, 1] * _VIEW).sum()
        clear = (10 ** (-stock.base) * _VIEW) @ _CMF / (_CMF[:, 1] * _VIEW).sum()
        return linear_to_srgb((xyz / max(clear[1], 1e-6)) @ XYZ_TO_SRGB.T)
    # Negative: a three-band scan, then a per-channel log inversion anchored on the grey patch (what a lab scan does).
    scan = (trans * _VIEW) @ _SCANNER  # (h, w, 3), channel c reads the dye of layer c
    grey_d = np.stack([np.interp(np.log10(1.0) + ev * np.log10(2.0), stock.log_exposure, stock.curves[:, c]) for c in range(3)])
    grey_scan = ((10 ** (-(grey_d @ stock.dyes.T + stock.base))) * _VIEW) @ _SCANNER
    log_ratio = np.log10(np.maximum(scan, 1e-9)) - np.log10(np.maximum(grey_scan, 1e-9))
    slope = np.array([np.gradient(stock.curves[:, c], stock.log_exposure)[np.argmin(np.abs(stock.log_exposure))] for c in range(3)])
    positive_log = -log_ratio / np.maximum(slope, 1e-3) * 0.6  # Profiles order layers R (cyan dye), G (magenta), B (yellow).
    lin = 0.184 * 10 ** positive_log
    return linear_to_srgb(lin)


def available() -> list[str]:
    return sorted(p.stem for p in PROFILES.glob("*.json") if json.loads(p.read_text())["info"].get("support") == "film")
