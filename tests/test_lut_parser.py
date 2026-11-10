"""XMP preset parsing, forward model and 3D-LUT container (no owner files needed: a minimal XMP is written here)."""
from pathlib import Path

import numpy as np
import pytest

from research.lut.engine.lut3d import LUT3D
from research.lut.forward_model import Preset, identity_preset, lab_to_srgb, srgb_to_lab
from research.lut.xmp_parser import parse_xmp

XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Normal" crs:Version="17.2" crs:ProcessVersion="15.4" crs:RequiresRGBTables="False"
   crs:HueAdjustmentRed="+6" crs:HueAdjustmentOrange="+50" crs:SaturationAdjustmentRed="-35" crs:LuminanceAdjustmentBlue="-28"
   crs:SplitToningShadowHue="42" crs:SplitToningShadowSaturation="15" crs:ColorGradeMidtoneHue="28" crs:ColorGradeMidtoneSat="28"
   crs:ColorGradeBlending="50" crs:GrainAmount="31" crs:GrainSize="32" crs:Dehaze="-10" crs:Clarity2012="-5" crs:HasSettings="True">
   <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">Test_Preset</rdf:li></rdf:Alt></crs:Name>
   <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 13</rdf:li><rdf:li>128, 128</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
   <crs:ToneCurvePV2012Red><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012Red>
   <crs:ToneCurvePV2012Green><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012Green>
   <crs:ToneCurvePV2012Blue><rdf:Seq><rdf:li>13, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012Blue>
   <crs:PointColors><rdf:Seq><rdf:li>0.515105, 0.621142, 0.075477, 0.000000, -0.030000, 0.160000, 0.500000, 0.040000, 0.440000, 0.770000, 1.000000, 0.000000, 0.440000, 0.800000, 1.000000, 0.000000, 0.120000, 0.480000, 1.000000</rdf:li></rdf:Seq></crs:PointColors>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


@pytest.fixture
def parsed(tmp_path: Path):
    f = tmp_path / "Test_Preset parvec.xmp"
    f.write_text(XMP)
    return parse_xmp(f)


def test_parse_fields(parsed):
    assert parsed["name"] == "Test_Preset"
    assert parsed["header"]["ProcessVersion"] == "15.4"
    assert parsed["curves"]["ToneCurvePV2012"] == [[0, 13], [128, 128], [255, 255]]
    assert parsed["curves"]["ToneCurvePV2012Blue"][0] == [13, 0]
    assert parsed["hsl"]["Orange"]["hue"] == 50 and parsed["hsl"]["Red"]["saturation"] == -35
    assert parsed["hsl"]["Blue"]["luminance"] == -28
    assert parsed["grading"]["ColorGradeMidtoneSat"] == 28 and parsed["grading"]["SplitToningShadowHue"] == 42
    assert parsed["grain"]["GrainAmount"] == 31 and parsed["basic"]["Dehaze"] == -10
    assert parsed["lut"]["embedded_3d_lut"] is False
    assert len(parsed["point_colors"]) == 1 and abs(parsed["point_colors"][0]["src_hue_rad"] - 0.515105) < 1e-6


def test_forward_model_properties(parsed):
    p = Preset.from_parsed(parsed)
    greys = np.stack([np.linspace(0, 1, 17)] * 3, -1)
    out = p.apply(greys)
    assert out.shape == (17, 3) and np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
    assert out[0].mean() * 255 > 8          # lifted blacks from the 0 -> 13 point
    assert np.all(np.diff(out.mean(1)) >= -1e-6)   # grey ramp monotone
    assert np.allclose(identity_preset().apply(greys), greys, atol=1e-6)
    # lattice path agrees with the exact path within 8-bit tolerance
    img = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    exact = np.round(p.apply(img / 255) * 255)
    fast = p.apply_image(img).astype(float)
    assert np.abs(exact - fast).mean() < 1.0 and np.abs(exact - fast).max() <= 6


def test_lab_roundtrip():
    x = np.random.default_rng(1).random((500, 3))
    assert np.abs(lab_to_srgb(srgb_to_lab(x)) - x).max() < 1e-9


def test_lut3d_cube_roundtrip(tmp_path: Path, parsed):
    p = Preset.from_parsed(parsed)
    lut = LUT3D(p.lattice(9), "t")
    rep = lut.validate()
    assert rep["ok"] and rep["grey_monotone"]
    f = tmp_path / "t.cube"
    lut.to_cube(f)
    back = LUT3D.from_cube(f)
    assert back.size == 9 and np.allclose(back.lattice, lut.lattice, atol=1e-5)
    x = np.random.default_rng(2).random((50, 3)).astype(np.float32)
    assert np.abs(back.apply(x) - lut.apply(x)).max() < 1e-4
    # red varies fastest in the file: the second row is lattice[1, 0, 0]
    rows = [l for l in f.read_text().splitlines() if l and l[0].isdigit()]
    assert np.allclose([float(v) for v in rows[1].split()], lut.lattice[1, 0, 0], atol=1e-5)
    with pytest.raises(ValueError):
        LUT3D(np.zeros((3, 3, 2, 3)))
