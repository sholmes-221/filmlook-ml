"""Append-only experiment registry (research/registry.jsonl)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "registry.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path | None = None) -> list[dict]:
    path = path or REGISTRY
    if not Path(path).exists():
        return []
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    latest: dict[str, dict] = {}
    for r in rows:                                   # last record per id wins (status updates are appended)
        latest[r["id"]] = {**latest.get(r["id"], {}), **r}
    return list(latest.values())


def get(exp_id: str, path: Path | None = None) -> dict | None:
    path = path or REGISTRY
    return next((r for r in load(path) if r["id"] == exp_id), None)


def _append(row: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def register(exp_id: str, hypothesis: str, command: list[str], baseline_id: str | None = None,
             metrics_of_interest: dict | None = None, labels: str = "", path: Path | None = None, **extra) -> dict:
    path = path or REGISTRY
    """metrics_of_interest: {name: {"path": "a.b.c", "higher_is_better": bool, "kind": "synthetic|real|proxy"}}"""
    if get(exp_id, path) is not None:
        raise ValueError(f"experiment {exp_id} already registered")
    row = {"id": exp_id, "hypothesis": hypothesis, "command": command, "baseline_id": baseline_id,
           "metrics_of_interest": metrics_of_interest or {}, "labels": labels, "status": "registered",
           "registered_at": _now(), **extra}
    _append(row, path)
    return row


def update(exp_id: str, path: Path | None = None, **fields) -> dict:
    path = path or REGISTRY
    if get(exp_id, path) is None:
        raise KeyError(exp_id)
    row = {"id": exp_id, "updated_at": _now(), **fields}
    _append(row, path)
    return get(exp_id, path)


def dig(obj, dotted: str):
    cur = obj
    for k in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur
