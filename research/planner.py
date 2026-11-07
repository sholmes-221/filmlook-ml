"""Pick the next experiment from research/backlog.yaml by expected value / cost within the remaining budget."""
from __future__ import annotations

from pathlib import Path

import yaml

from research import experiment_registry as R
from research.experiment_runner import remaining

BACKLOG = R.ROOT / "research" / "backlog.yaml"


def load_backlog(path: Path = BACKLOG) -> list[dict]:
    return (yaml.safe_load(Path(path).read_text()) or {}).get("experiments", [])


def pending(path: Path = BACKLOG) -> list[dict]:
    done = {r["id"] for r in R.load() if r.get("status") in ("done", "failed", "rejected", "blocked")}
    return [e for e in load_backlog(path) if e["id"] not in done and e.get("status", "pending") == "pending"]


def select_next(path: Path = BACKLOG) -> dict | None:
    rem = remaining()
    cands = [e for e in pending(path) if e.get("gpu_minutes", 0) * 60 <= rem["gpu_seconds"]
             and e.get("disk_gb", 0) * 1024 ** 3 <= rem["disk_bytes"] and not e.get("blocked_by")]
    if not cands:
        return None
    return max(cands, key=lambda e: e.get("value", 1) / max(e.get("gpu_minutes", 0) + e.get("cpu_minutes", 1), 1))


def render_markdown(path: Path = BACKLOG) -> str:
    rows = load_backlog(path)
    done = {r["id"]: r.get("status") for r in R.load()}
    out = ["# Research backlog (generated from research/backlog.yaml; edit the YAML)", "",
           "| id | hypothesis | value (1-5) | GPU min | disk GB | metric | falsified if | status |", "|---|---|---|---|---|---|---|---|"]
    for e in rows:
        st = done.get(e["id"], e.get("status", "pending"))
        out.append(f"| {e['id']} | {e['hypothesis']} | {e.get('value', '')} | {e.get('gpu_minutes', 0)} | {e.get('disk_gb', 0)} | "
                   f"{e.get('metric', '')} | {e.get('falsified_if', '')} | {st} |")
    return "\n".join(out) + "\n"
