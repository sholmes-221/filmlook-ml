"""Machine-readable summary of an experiment plus a short EXPERIMENT_LOG paragraph."""
from __future__ import annotations

import json
from pathlib import Path

from research import evaluator
from research import experiment_registry as R

SUMMARIES = R.ROOT / "reports" / "experiments"


def summarize(exp_id: str) -> dict:
    exp = R.get(exp_id)
    cmp_ = evaluator.compare(exp_id)
    s = {"id": exp_id, "hypothesis": exp["hypothesis"], "status": exp["status"], "metrics": exp.get("metrics", {}),
         "comparison": cmp_, "resources": {k: exp.get(k) for k in ("wall_seconds", "peak_gpu_mb_total", "disk_bytes")},
         "labels": exp.get("labels", "")}
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    (SUMMARIES / f"{exp_id}.summary.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def log_paragraph(exp_id: str, learned: str, next_step: str) -> str:
    s = summarize(exp_id)
    lines = [f"### {exp_id} — {s['hypothesis']}", f"- Status: {s['status']}; wall {s['resources']['wall_seconds']} s; "
             f"peak GPU {s['resources']['peak_gpu_mb_total']} MB (total on card)."]
    for name, e in s["comparison"]["per_metric"].items():
        base = f" (baseline {e['baseline']}, {e['verdict']})" if "baseline" in e else ""
        ci = f" CI {e['ci95']}" if e.get("ci95") else ""
        lines.append(f"- {name} [{e['kind']}]: {e['value']}{ci}{base}")
    lines += [f"- Decision rule: {s['comparison']['decision']}", f"- Learned: {learned}", f"- Next: {next_step}"]
    return "\n".join(lines)
