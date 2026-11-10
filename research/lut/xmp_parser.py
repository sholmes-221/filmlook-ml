"""Parse Adobe Camera Raw / Lightroom preset XMP files into plain parameter dictionaries.

Usage:
    python -m research.lut.xmp_parser /home/shanmo/misc/lut --out data/lut/presets.json

What the parser reads (namespace crs = http://ns.adobe.com/camera-raw-settings/1.0/):
  * header: Name, Group, ProcessVersion, Version, UUID, SupportsAmount, RequiresRGBTables, PresetType
  * tone curves: ToneCurvePV2012 (main) and ToneCurvePV2012Red/Green/Blue as lists of (x, y) in 0..255
  * HSL: HueAdjustment*, SaturationAdjustment*, LuminanceAdjustment* for the 8 bands
  * split toning / colour grading: SplitToning{Shadow,Highlight}{Hue,Saturation}, SplitToningBalance,
    ColorGrade{Shadow,Midtone,Highlight,Global}{Hue,Sat,Lum}, ColorGradeBlending
  * parametric tone sliders (Parametric*), basic sliders (Exposure2012, Contrast2012, ... when present)
  * calibration (ShadowTint, Red/Green/BlueHue, Red/Green/BlueSaturation when present)
  * grain (GrainAmount/Size/Frequency), sharpening, Clarity2012, Dehaze
  * LUT indicators: LookTable / RGBTable / Look (base64) -- reported as absent/present, data never copied
Everything else is kept under `other` (string values) so nothing is silently dropped.
The XMP files themselves are the owner's private assets; only the parsed numbers are written out.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
CURVES = ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue")
LUT_KEYS = ("LookTable", "RGBTable", "RGBTableAmount", "Look")
NUMERIC_PREFIXES = ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment", "SplitToning", "ColorGrade",
                    "Parametric", "Grain", "Sharpen", "Sharpness", "Clarity2012", "Dehaze", "Exposure2012",
                    "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012", "Texture",
                    "Vibrance", "Saturation", "Temperature", "Tint", "ShadowTint", "RedHue", "GreenHue", "BlueHue",
                    "RedSaturation", "GreenSaturation", "BlueSaturation", "CurveRefineSaturation",
                    "LuminanceSmoothing", "ColorNoiseReduction")


def _num(s: str):
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def parse_xmp(path: Path) -> dict:
    root = ET.parse(path).getroot()
    desc = root.find(f".//{{{RDF}}}Description")
    if desc is None:
        raise ValueError(f"{path}: no rdf:Description")
    out: dict = {"file": path.name, "header": {}, "curves": {}, "hsl": {b: {} for b in BANDS}, "grading": {},
                 "parametric": {}, "basic": {}, "calibration": {}, "grain": {}, "sharpen": {}, "lut": {},
                 "other": {}}
    attrs = {k.split("}")[1]: v for k, v in desc.attrib.items() if k.startswith("{" + CRS)}
    # child elements with text (the curves, names, RGB tables)
    children = {}
    for el in desc:
        if not el.tag.startswith("{" + CRS):
            continue
        key = el.tag.split("}")[1]
        seq = el.find(f".//{{{RDF}}}Seq")
        alt = el.find(f".//{{{RDF}}}Alt")
        if seq is not None:
            children[key] = [li.text or "" for li in seq.findall(f"{{{RDF}}}li")]
        elif alt is not None:
            li = alt.find(f"{{{RDF}}}li")
            children[key] = li.text if li is not None else ""
        else:
            children[key] = (el.text or "").strip()
    for k in ("Name", "Group", "ProcessVersion", "Version", "UUID", "SupportsAmount", "SupportsAmount2",
              "RequiresRGBTables", "PresetType", "CameraProfile", "LookName", "HasSettings", "ToneCurveName2012",
              "CompatibleVersion"):
        v = attrs.get(k, children.get(k))
        if v is not None:
            out["header"][k] = v
    for c in CURVES:
        if c in children:
            pts = []
            for li in children[c]:
                m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", li)
                if m:
                    pts.append([float(m.group(1)), float(m.group(2))])
            out["curves"][c] = pts
    # Point Color (Lightroom 13+): 19 numbers per entry; a row of -1 means "no point colour".
    pcs = []
    for li in children.get("PointColors", []):
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", li)]
        if len(vals) == 19 and not all(v == -1.0 for v in vals):
            pcs.append({"src_hue_rad": vals[0], "src_sat": vals[1], "src_lum": vals[2], "d_hue_rad": vals[3],
                        "d_sat": vals[4], "d_lum": vals[5], "range": vals[6], "hue_trap": vals[7:11],
                        "sat_trap": vals[11:15], "lum_trap": vals[15:19]})
    out["point_colors"] = pcs
    for k in LUT_KEYS:
        if k in attrs or k in children:
            v = attrs.get(k, children.get(k))
            out["lut"][k] = {"present": True, "length": len(v) if isinstance(v, str) else len(json.dumps(v))}
    out["lut"]["embedded_3d_lut"] = any(k in children or (k in attrs and attrs[k] not in ("", "False"))
                                       for k in ("LookTable", "RGBTable", "Look"))
    out["lut"]["RequiresRGBTables"] = attrs.get("RequiresRGBTables")
    for k, v in attrs.items():
        if k in out["header"] or k in LUT_KEYS:
            continue
        n = _num(v)
        placed = False
        for b in BANDS:
            for kind in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
                if k == kind + b:
                    out["hsl"][b][kind.replace("Adjustment", "").lower()] = n
                    placed = True
        if placed:
            continue
        if k.startswith("SplitToning") or k.startswith("ColorGrade"):
            out["grading"][k] = n
        elif k.startswith("Parametric"):
            out["parametric"][k] = n
        elif k.startswith("Grain"):
            out["grain"][k] = n
        elif k.startswith("Sharpen") or k == "Sharpness":
            out["sharpen"][k] = n
        elif k in ("ShadowTint", "RedHue", "GreenHue", "BlueHue", "RedSaturation", "GreenSaturation", "BlueSaturation"):
            out["calibration"][k] = n
        elif n is not None and k.startswith(NUMERIC_PREFIXES):
            out["basic"][k] = n
        else:
            out["other"][k] = v
    for k, v in children.items():
        if k not in CURVES and k not in out["header"] and k not in LUT_KEYS and k != "PointColors":
            out["other"][k] = v if isinstance(v, str) else (f"<seq len={len(v)}>" if v is not None else "")
    out["name"] = out["header"].get("Name") or path.stem.replace(" parvec", "")
    return out


def is_identity_curve(pts) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in pts) if pts else True


def summarise(p: dict) -> dict:
    s = {"name": p["name"], "process_version": p["header"].get("ProcessVersion"),
         "main_curve": p["curves"].get("ToneCurvePV2012"),
         "rgb_curves_non_identity": [c for c in CURVES[1:] if not is_identity_curve(p["curves"].get(c, []))],
         "hsl_nonzero": sum(1 for b in BANDS for v in p["hsl"][b].values() if v),
         "grading_nonzero": {k: v for k, v in p["grading"].items() if v and k not in ("ColorGradeBlending",)},
         "parametric_nonzero": {k: v for k, v in p["parametric"].items() if v and not k.endswith("Split")},
         "basic": p["basic"], "calibration_nonzero": {k: v for k, v in p["calibration"].items() if v},
         "grain": p["grain"], "embedded_3d_lut": p["lut"]["embedded_3d_lut"], "point_colors": len(p["point_colors"]),
         "camera_profile": p["header"].get("CameraProfile"), "look_name": p["header"].get("LookName")}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/lut/presets.json"))
    a = ap.parse_args()
    presets = [parse_xmp(f) for f in sorted(a.folder.glob("*.xmp"))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(presets, indent=1))
    for p in presets:
        print(json.dumps(summarise(p)))
    print(f"wrote {len(presets)} presets to {a.out}")


if __name__ == "__main__":
    main()
