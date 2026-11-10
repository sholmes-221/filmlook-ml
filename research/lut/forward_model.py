"""Approximate forward rendering of a parsed Lightroom / Camera Raw preset (numpy, vectorised).

    from research.lut.forward_model import load_presets, Preset
    presets = load_presets("data/lut/presets.json")
    out = presets["Kodak_Portra_400"].apply(rgb)        # rgb: float array (..., 3), sRGB in [0, 1]

HONEST STATEMENT OF WHAT THIS IS
Lightroom's rendering pipeline is proprietary. The functions below reproduce the *documented behaviour* of each
control with simple, stated formulas. They were not fitted to Lightroom output (no Lightroom is available here) and
they operate on an sRGB JPEG, not on demosaiced raw data with a camera profile. Treat every render as "a plausible
look in the direction the preset describes", not as what Lightroom would produce. Every approximation:

 1. Working space. Lightroom applies the point tone curve in a ProPhoto-primaries space with an sRGB-like transfer
    curve ("Melissa RGB"). We convert sRGB -> linear -> ProPhoto (Bradford D65->D50), encode with the sRGB TRC, apply
    the curves on 0..255, decode and come back to sRGB. Input values outside ProPhoto (none for sRGB) are clipped.
 2. Point curves (ToneCurvePV2012, *Red/Green/Blue) are monotone cubic (PCHIP) interpolants through the control points;
    Lightroom uses its own smooth spline. Order: RGB channel curves then the main curve (both act on the same encoded
    values; for the single preset with channel curves the difference in order is small). Parametric curve sliders are
    all zero in these presets and are ignored; `CurveRefineSaturation=100` is the default (curve applied per channel).
 3. HSL. Eight bands with centres at HSV hue 0 (red), 30 (orange), 60 (yellow), 120 (green), 180 (aqua), 240 (blue),
    280 (purple), 320 (magenta) degrees, triangular weights between neighbouring centres. Hue slider: +/-100 rotates
    the HSV hue by +/-30 degrees. Saturation slider: HSV saturation times (1 + v/100) for v<0 and (1 + 0.6 v/100) for
    v>0. Luminance slider: RGB scaled by (1 + 0.5 v/100), both weighted by a saturation mask (smoothstep of HSV S
    between 0.08 and 0.35) so neutrals are untouched. Lightroom's real HSL model is unpublished.
 4. Colour grading / split toning. Shadows (SplitToningShadow*), midtones (ColorGradeMidtone*) and highlights
    (SplitToningHighlight*) add a CIELAB chroma offset of 24 * sat/100 units in the direction of the wheel hue (the
    Lab hue of the sRGB colour with that HSV hue), weighted by luminance windows: shadows 1-smoothstep(0.1,0.6,L),
    highlights smoothstep(0.4,0.9,L), midtones 1-|2L-1| scaled by Blending/50; both ends fade to zero within 0.06 of
    black and white so pure black/white stay neutral. Balance and the Global/luminance wheels are zero in all presets.
 5. Point Color (Kodak Gold 200 has 3 entries, two presets 1): parsed as (source hue in radians, source saturation,
    source luminance, hue/sat/lum deltas, range, three trapezoids). Applied as HSV hue rotation by the hue delta,
    saturation and value scaled by (1 + delta) under a mask that is a product of a hue window (full within 30 degrees,
    zero beyond 60, scaled by `range`/0.5) and the saturation and luminance trapezoids. Small effects.
 6. Dehaze (Phoenix 200: -14) is approximated as a veil: out = out*(1-k) + k*0.5 with k = 0.5*|v|/100 for v<0;
    positive dehaze is ignored. Clarity, Texture, Sharpening and Grain are spatial (texture) operations and are not
    colour transforms; they are omitted here (Grain is simulated separately in `synthetic.py` as an option).
 7. Camera profile: none of the presets sets one (no `CameraProfile`, no `LookName`, no embedded LookTable/RGBTable:
    `RequiresRGBTables="False"` everywhere), so the preset is "Adobe Color on top of whatever the photo already is".
 8. Output is clipped to [0, 1] sRGB; the inverse analysis treats clipped pixels as lost information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

BAND_CENTRES = {"Red": 0.0, "Orange": 30.0, "Yellow": 60.0, "Green": 120.0, "Aqua": 180.0, "Blue": 240.0,
                "Purple": 280.0, "Magenta": 320.0}
BANDS = list(BAND_CENTRES)

# --- colour-space helpers -------------------------------------------------------------------------------------
M_SRGB_TO_XYZ_D65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
M_BRADFORD_D65_D50 = np.array([[1.0478112, 0.0228866, -0.0501270],
                               [0.0295424, 0.9904844, -0.0170491],
                               [-0.0092345, 0.0150436, 0.7521316]])
M_XYZ_D50_TO_PROPHOTO = np.array([[1.3459433, -0.2556075, -0.0511118],
                                  [-0.5445989, 1.5081673, 0.0205351],
                                  [0.0000000, 0.0000000, 1.2118128]])
M_SRGB_TO_PROPHOTO = M_XYZ_D50_TO_PROPHOTO @ M_BRADFORD_D65_D50 @ M_SRGB_TO_XYZ_D65
M_PROPHOTO_TO_SRGB = np.linalg.inv(M_SRGB_TO_PROPHOTO)
M_SRGB_TO_XYZ_D50 = M_BRADFORD_D65_D50 @ M_SRGB_TO_XYZ_D65
M_XYZ_D50_TO_SRGB = np.linalg.inv(M_SRGB_TO_XYZ_D50)
WHITE_D50 = np.array([0.96422, 1.0, 0.82521])


def srgb_decode(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((np.abs(v) + 0.055) / 1.055) ** 2.4 * np.sign(v))


def srgb_encode(v):
    v = np.asarray(v, dtype=np.float64)
    a = np.abs(v)
    return np.sign(v) * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def xyz_to_lab(xyz):
    r = xyz / WHITE_D50
    f = np.where(r > 216 / 24389, np.cbrt(np.maximum(r, 0)), (24389 / 27 * r + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    r = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389)
    return r * WHITE_D50


def srgb_to_lab(rgb):
    return xyz_to_lab(srgb_decode(rgb) @ M_SRGB_TO_XYZ_D50.T)


def lab_to_srgb(lab):
    return srgb_encode(np.clip(lab_to_xyz(lab) @ M_XYZ_D50_TO_SRGB.T, 0, None))


def rgb_to_hsv(rgb):
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    s = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    dd = np.where(d > 1e-10, d, 1.0)
    h = np.where(mx == r, (g - b) / dd, np.where(mx == g, 2 + (b - r) / dd, 4 + (r - g) / dd))
    h = np.where(d > 1e-10, (h * 60) % 360, 0.0)
    return h, s, mx


def hsv_to_rgb(h, s, v):
    h6 = (h % 360) / 60
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def wheel_hue_to_lab_direction(hue_deg: float) -> np.ndarray:
    """Unit (a, b) direction of the sRGB colour with HSV hue `hue_deg`, S=1, V=1 (saturated wheel colour)."""
    rgb = hsv_to_rgb(np.array([hue_deg]), np.array([1.0]), np.array([1.0]))
    lab = srgb_to_lab(rgb)[0]
    ab = lab[1:]
    return ab / max(np.linalg.norm(ab), 1e-9)


def band_weights(h):
    """Triangular weights of the 8 HSL bands for HSV hue h (degrees). Rows sum to 1."""
    centres = np.array(list(BAND_CENTRES.values()))
    w = np.zeros(h.shape + (len(centres),))
    for i, c in enumerate(centres):
        left = centres[i - 1] if i > 0 else centres[-1] - 360
        right = centres[(i + 1) % len(centres)] + (360 if i == len(centres) - 1 else 0)
        d = ((h - c + 180) % 360) - 180          # signed distance to centre
        wl = np.clip(1 + d / (c - left), 0, 1)   # d<0 side
        wr = np.clip(1 - d / (right - c), 0, 1)  # d>0 side
        w[..., i] = np.where(d < 0, wl, wr)
    return w / np.maximum(w.sum(-1, keepdims=True), 1e-9)


# --- preset ------------------------------------------------------------------------------------------------------
@dataclass
class Preset:
    name: str
    main_curve: list
    rgb_curves: dict
    hsl: dict                      # band -> {hue, saturation, luminance}
    grading: dict
    point_colors: list = field(default_factory=list)
    dehaze: float = 0.0
    grain: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_parsed(p: dict) -> "Preset":
        return Preset(name=p["name"], main_curve=p["curves"].get("ToneCurvePV2012", [[0, 0], [255, 255]]),
                      rgb_curves={c: p["curves"].get("ToneCurvePV2012" + c, [[0, 0], [255, 255]]) for c in ("Red", "Green", "Blue")},
                      hsl=p["hsl"], grading=p["grading"], point_colors=p.get("point_colors", []),
                      dehaze=p["basic"].get("Dehaze", 0.0) or 0.0, grain=p.get("grain", {}), raw=p)

    # -- curves
    @staticmethod
    def _interp(points):
        pts = sorted({float(x): float(y) for x, y in points}.items())
        xs = np.array([x for x, _ in pts]) / 255
        ys = np.array([y for _, y in pts]) / 255
        if len(xs) < 2:
            return lambda v: v
        if len(xs) == 2:
            return lambda v, xs=xs, ys=ys: np.interp(v, xs, ys)
        f = PchipInterpolator(xs, ys, extrapolate=True)
        return lambda v, f=f: f(np.clip(v, 0, 1))

    def curve_lut(self, n: int = 256) -> np.ndarray:
        """(n, 3) table: encoded ProPhoto value in -> value out, per channel (channel curve then main)."""
        x = np.linspace(0, 1, n)
        main = self._interp(self.main_curve)
        out = np.zeros((n, 3))
        for i, c in enumerate(("Red", "Green", "Blue")):
            out[:, i] = np.clip(main(np.clip(self._interp(self.rgb_curves[c])(x), 0, 1)), 0, 1)
        return out

    def apply_curves(self, rgb):
        pp = srgb_decode(rgb) @ M_SRGB_TO_PROPHOTO.T
        enc = srgb_encode(np.clip(pp, 0, 1))
        main = self._interp(self.main_curve)
        out = np.empty_like(enc)
        for i, c in enumerate(("Red", "Green", "Blue")):
            out[..., i] = main(np.clip(self._interp(self.rgb_curves[c])(enc[..., i]), 0, 1))
        lin = srgb_decode(np.clip(out, 0, 1)) @ M_PROPHOTO_TO_SRGB.T
        return np.clip(srgb_encode(np.clip(lin, 0, None)), 0, 1)

    # -- HSL
    def apply_hsl(self, rgb):
        h, s, v = rgb_to_hsv(rgb)
        w = band_weights(h)
        hue = np.array([self.hsl[b].get("hue", 0.0) or 0.0 for b in BANDS])
        sat = np.array([self.hsl[b].get("saturation", 0.0) or 0.0 for b in BANDS])
        lum = np.array([self.hsl[b].get("luminance", 0.0) or 0.0 for b in BANDS])
        if not (np.any(hue) or np.any(sat) or np.any(lum)):
            return rgb
        mask = smoothstep(0.08, 0.35, s)
        dh = (w @ hue) / 100 * 30.0
        ds = w @ sat
        sfac = np.where(ds < 0, 1 + ds / 100, 1 + 0.6 * ds / 100)
        dl = (w @ lum) / 100 * 0.5
        h2 = h + dh * mask
        s2 = np.clip(s * (1 + (sfac - 1) * mask), 0, 1)
        v2 = np.clip(v * (1 + dl * mask), 0, 1)
        return hsv_to_rgb(h2, s2, v2)

    # -- point colour
    def apply_point_colors(self, rgb):
        if not self.point_colors:
            return rgb
        h, s, v = rgb_to_hsv(rgb)
        for pc in self.point_colors:
            src_h = np.degrees(pc["src_hue_rad"]) % 360
            d = np.abs(((h - src_h + 180) % 360) - 180)
            scale = max(pc.get("range", 0.5), 1e-3) / 0.5
            wh = 1 - smoothstep(30 * scale, 60 * scale, d)
            ws = self._trap(s, pc["sat_trap"])
            wl = self._trap(v, pc["lum_trap"])
            m = wh * ws * wl
            h = h + np.degrees(pc["d_hue_rad"]) * m
            s = np.clip(s * (1 + pc["d_sat"] * m), 0, 1)
            v = np.clip(v * (1 + pc["d_lum"] * m), 0, 1)
        return hsv_to_rgb(h, s, v)

    @staticmethod
    def _trap(x, t):
        a, b, c, d = t
        up = np.clip((x - a) / max(b - a, 1e-6), 0, 1) if b > a else (x >= a).astype(float)
        down = np.clip((d - x) / max(d - c, 1e-6), 0, 1) if d > c else (x <= d).astype(float)
        return np.minimum(up, down)

    # -- colour grading
    def grading_terms(self):
        g = self.grading
        terms = []
        for zone, hk, sk in (("shadow", "SplitToningShadowHue", "SplitToningShadowSaturation"),
                             ("highlight", "SplitToningHighlightHue", "SplitToningHighlightSaturation"),
                             ("midtone", "ColorGradeMidtoneHue", "ColorGradeMidtoneSat")):
            sat = g.get(sk, 0.0) or 0.0
            if sat:
                terms.append((zone, g.get(hk, 0.0) or 0.0, sat))
        return terms

    def apply_grading(self, rgb):
        terms = self.grading_terms()
        if not terms:
            return rgb
        lab = srgb_to_lab(rgb)
        L = lab[..., 0] / 100
        blend = (self.grading.get("ColorGradeBlending", 50.0) or 50.0) / 50.0
        edge = smoothstep(0.0, 0.06, L) * smoothstep(0.0, 0.06, 1 - L)
        for zone, hue, sat in terms:
            if zone == "shadow":
                w = 1 - smoothstep(0.1, 0.6, L)
            elif zone == "highlight":
                w = smoothstep(0.4, 0.9, L)
            else:
                w = (1 - np.abs(2 * L - 1)) * blend
            d = wheel_hue_to_lab_direction(hue)
            lab[..., 1:] += (24.0 * sat / 100 * w * edge)[..., None] * d
        return np.clip(lab_to_srgb(lab), 0, 1)

    def apply_dehaze(self, rgb):
        if self.dehaze >= 0:
            return rgb
        k = 0.5 * abs(self.dehaze) / 100
        return rgb * (1 - k) + k * 0.5

    def apply(self, rgb) -> np.ndarray:
        """sRGB float (..., 3) in [0, 1] -> sRGB float in [0, 1]. Order: curves, HSL, point colour, grading, dehaze."""
        x = np.clip(np.asarray(rgb, dtype=np.float64), 0, 1)
        x = self.apply_curves(x)
        x = self.apply_hsl(x)
        x = self.apply_point_colors(x)
        x = self.apply_grading(x)
        x = self.apply_dehaze(x)
        return np.clip(x, 0, 1)

    def apply_image(self, img_u8: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> uint8 HxWx3 via a 33^3 lattice + trilinear interpolation (fast path for images)."""
        lut = self.lattice(33)
        return apply_lattice(lut, img_u8)

    def lattice(self, n: int = 33) -> np.ndarray:
        g = np.linspace(0, 1, n)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        return self.apply(grid).reshape(n, n, n, 3)


def apply_lattice(lut: np.ndarray, img_u8: np.ndarray) -> np.ndarray:
    """Trilinear 3D-LUT application (lut indexed [r, g, b])."""
    n = lut.shape[0]
    x = img_u8.astype(np.float64) / 255 * (n - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, n - 2)
    f = x - i0
    out = np.zeros(img_u8.shape[:-1] + (3,))
    for dr in (0, 1):
        wr = f[..., 0] if dr else 1 - f[..., 0]
        for dg in (0, 1):
            wg = f[..., 1] if dg else 1 - f[..., 1]
            for db in (0, 1):
                wb = f[..., 2] if db else 1 - f[..., 2]
                out += (wr * wg * wb)[..., None] * lut[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


def load_presets(path="data/lut/presets.json") -> dict[str, Preset]:
    data = json.loads(Path(path).read_text())
    return {p["name"]: Preset.from_parsed(p) for p in data}


def identity_preset() -> Preset:
    return Preset(name="none", main_curve=[[0, 0], [255, 255]], rgb_curves={c: [[0, 0], [255, 255]] for c in ("Red", "Green", "Blue")},
                  hsl={b: {} for b in BANDS}, grading={})


if __name__ == "__main__":
    import sys
    presets = load_presets(sys.argv[1] if len(sys.argv) > 1 else "data/lut/presets.json")
    greys = np.stack([np.linspace(0, 1, 9)] * 3, -1)
    for name, p in presets.items():
        out = p.apply(greys)
        print(name.ljust(22), "grey ramp ->", np.round(out * 255).astype(int).tolist()[:3], "...", np.round(out[-1] * 255).astype(int).tolist())
