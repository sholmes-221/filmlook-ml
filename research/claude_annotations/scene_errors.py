"""Which kinds of scene does the look-ranking engine fail on? Engine predictions on every synthetic test render (variant
0: 81 scenes x 14 looks), joined with the Haiku annotation of each scene's untouched render (`lutscene`).

  python -m research.claude_annotations.scene_errors   -> reports/claude/scene_errors.json

Reports top-1 accuracy of the live engine (method D) by scene field (subject, setting, light source, key, palette,
saturation, dominant colours), each with its count, so that the fields that move accuracy most stand out. Labels are
the applied preset (known by construction); the annotation only describes the scene and is never a label.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.lut.engine.engine import FilmStyleEngine as Engine  # noqa: E402

OUT = ROOT / "reports" / "claude" / "scene_errors.json"
FIELDS = ("subject_primary", "setting", "light_source", "light_quality", "key", "contrast", "palette", "saturation", "colour_temperature", "depth", "people_count")


def main() -> None:
    scenes = P._synth_test(); ann = P.annotated("lutscene", P.cfg()["models"]["cheap"])
    eng = Engine(ROOT / "research/lut/db")
    per_scene: dict[str, list[bool]] = defaultdict(list); n = 0
    for scene, by in sorted(scenes.items()):
        for preset, path in by.items():
            out = eng.analyze(path)
            pred = out["styles"][0]["id"] if out.get("styles") and not out.get("uncertain") else "none"
            truth = "none" if preset == "none" else preset.lower().replace("'", "").replace(" ", "_")
            per_scene[scene].append(pred == truth or (preset == "none" and out.get("uncertain"))); n += 1
    acc = {s: float(np.mean(v)) for s, v in per_scene.items()}
    report = {"renders": n, "scenes": len(acc), "overall_top1": round(float(np.mean(list(acc.values()))), 3), "by_field": {}}
    for f in FIELDS:
        groups = defaultdict(list)
        for s, a in acc.items():
            k = f"lutscene:{s}"
            if k in ann:
                v = ann[k]["annotation"][f]; groups[v if not isinstance(v, list) else ",".join(v)].append(a)
        report["by_field"][f] = {v: {"scenes": len(x), "top1": round(float(np.mean(x)), 3)} for v, x in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    groups = defaultdict(list)
    for s, a in acc.items():
        k = f"lutscene:{s}"
        if k in ann:
            for col in ann[k]["annotation"]["dominant_colours"][:1]: groups[col].append(a)
    report["by_field"]["first_dominant_colour"] = {v: {"scenes": len(x), "top1": round(float(np.mean(x)), 3)} for v, x in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)[:3500])


if __name__ == "__main__":
    main()
