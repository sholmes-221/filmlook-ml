"""Score the look-description check: for each (scene, applied preset) item, did the reader pick the description that
was written for the applied preset? Ground truth is the pipeline's own record of which preset was applied; the model
never saw it. -> reports/claude/lutdesc_eval.json

Readings: accuracy against chance (0.25), per preset, confusion between presets (which descriptions get mistaken for
which), accuracy by the reader's own "visible_change" and "confidence", and the Sonnet reading where it exists.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402

OUT = ROOT / "reports" / "claude" / "lutdesc_eval.json"


def main() -> None:
    c = P.cfg(); items = {it["item_id"]: it for it in P.lutdesc_items()}
    report = {}
    for tier, model in (("cheap", c["models"]["cheap"]), ("check", c["models"]["check"])):
        ann = P.annotated("lutdesc", model)
        if not ann:
            continue
        per = defaultdict(lambda: [0, 0]); conf = Counter(); by_vis = defaultdict(lambda: [0, 0]); by_conf = defaultdict(lambda: [0, 0]); right = n = 0
        for k, r in ann.items():
            it = items.get(k)
            if it is None:
                continue
            a = r["annotation"]; truth = it["meta"]["answer"]; preset = it["group"]
            ok = a["choice"] == truth; n += 1; right += ok
            per[preset][0] += ok; per[preset][1] += 1
            by_vis[a["visible_change"]][0] += ok; by_vis[a["visible_change"]][1] += 1
            by_conf[a["confidence"]][0] += ok; by_conf[a["confidence"]][1] += 1
            if not ok:
                conf[(preset, it["meta"]["options"][int(a["choice"]) - 1])] += 1
        report[model] = {"n": n, "accuracy": round(right / max(1, n), 3), "chance": 0.25,
                         "per_preset": {p: round(v[0] / v[1], 2) for p, v in sorted(per.items())},
                         "by_visible_change": {k: {"n": v[1], "acc": round(v[0] / v[1], 2)} for k, v in by_vis.items()},
                         "by_confidence": {k: {"n": v[1], "acc": round(v[0] / v[1], 2)} for k, v in by_conf.items()},
                         "most_confused": [f"{a} read as {b}: {k}" for (a, b), k in conf.most_common(8)]}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
