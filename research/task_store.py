"""Persistent orchestrator task state (research/tasks.json) with a file lock for concurrent safety.

Task: {id, pipeline: data|research|eval, command: [...], depends_on: ["task:<id>" | "dataset:<id>"], state, attempts,
max_attempts, gpu_mb, wall_minutes_max, retry_backoff_s, next_attempt_at, pid, log, exit_code, last_error, created,
started, finished, env: {}}.
States: pending -> running -> done | retryable (re-queued after a backoff) | failed | blocked (dependency failed).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from research import experiment_registry as R

TASKS = R.ROOT / "research" / "tasks.json"
STATES = ("pending", "running", "done", "retryable", "failed", "blocked")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@contextmanager
def locked(path: Path | None = None):
    path = Path(path or TASKS); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _load(path: Path | None = None) -> dict:
    path = Path(path or TASKS)
    return json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}


def _save(d: dict, path: Path | None = None) -> None:
    path = Path(path or TASKS); tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(d, indent=1)); os.replace(tmp, path)


def all_tasks(path: Path | None = None) -> dict:
    with locked(path):
        return _load(path)


def get(task_id: str, path: Path | None = None) -> dict | None:
    return all_tasks(path).get(task_id)


def add(task_id: str, pipeline: str, command: list[str], depends_on: list[str] | None = None, gpu_mb: float = 0, max_attempts: int = 2,
        wall_minutes_max: float = 240, retry_backoff_s: float = 1800, env: dict | None = None, replace: bool = False, path: Path | None = None, **extra) -> dict:
    assert pipeline in ("data", "research", "eval"), pipeline
    with locked(path):
        d = _load(path)
        if task_id in d and not replace:
            return d[task_id]
        d[task_id] = {"id": task_id, "pipeline": pipeline, "command": command, "depends_on": depends_on or [], "state": "pending", "attempts": 0,
                      "max_attempts": max_attempts, "gpu_mb": gpu_mb, "wall_minutes_max": wall_minutes_max, "retry_backoff_s": retry_backoff_s,
                      "next_attempt_at": 0, "pid": None, "log": None, "exit_code": None, "last_error": None, "created": _now(), "started": None,
                      "finished": None, "env": env or {}, **extra}
        _save(d, path)
        return d[task_id]


def update(task_id: str, path: Path | None = None, **fields) -> dict:
    with locked(path):
        d = _load(path)
        if task_id not in d:
            raise KeyError(task_id)
        if "state" in fields:
            assert fields["state"] in STATES, fields["state"]
        d[task_id].update(fields); d[task_id]["updated"] = _now(); _save(d, path)
        return d[task_id]


def finish(task_id: str, exit_code: int, path: Path | None = None, error: str | None = None) -> dict:
    """Record a task's outcome: 0 -> done; 3 (paused/resumable) or any other failure -> retryable while attempts remain,
    then failed."""
    with locked(path):
        d = _load(path); t = d[task_id]
        t["exit_code"], t["finished"], t["pid"] = exit_code, _now(), None
        if exit_code == 0:
            t["state"], t["last_error"] = "done", None
        else:
            t["last_error"] = error or f"exit code {exit_code}"
            if t["attempts"] < t["max_attempts"]:
                t["state"] = "retryable"; t["next_attempt_at"] = time.time() + float(t.get("retry_backoff_s", 0)) * (2 ** max(0, t["attempts"] - 1))
            else:
                t["state"] = "failed"
        t["updated"] = _now(); _save(d, path)
        return t


def deps_status(task: dict, tasks: dict, dataset_ready) -> tuple[str, str]:
    """-> ('ok'|'wait'|'blocked', reason). dataset_ready(id) -> state string."""
    for dep in task.get("depends_on", []):
        kind, _, name = dep.partition(":")
        if kind == "task":
            st = tasks.get(name, {}).get("state", "missing")
            if st == "done":
                continue
            return ("blocked", f"task {name} {st}") if st in ("failed", "blocked", "missing") else ("wait", f"task {name} {st}")
        if kind == "dataset":
            st = dataset_ready(name)
            if st == "ready":
                continue
            return ("blocked", f"dataset {name} {st}") if st in ("permanent_failure", "quarantined") else ("wait", f"dataset {name} {st}")
    return "ok", "dependencies satisfied"


def runnable(tasks: dict, dataset_ready, now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    out = []
    for t in tasks.values():
        if t["state"] == "pending" or (t["state"] == "retryable" and now >= t.get("next_attempt_at", 0)):
            st, _ = deps_status(t, tasks, dataset_ready)
            if st == "ok":
                out.append(t)
    return sorted(out, key=lambda t: ({"data": 0, "eval": 1, "research": 2}[t["pipeline"]], t["created"]))


def recover(pid_alive, path: Path | None = None) -> list[str]:
    """Crash recovery: tasks marked running whose process is gone are re-queued (or failed when attempts are exhausted)."""
    changed = []
    with locked(path):
        d = _load(path)
        for t in d.values():
            if t["state"] == "running" and not pid_alive(t.get("pid")):
                t["pid"] = None; t["last_error"] = "process disappeared (crash or reboot)"; t["finished"] = _now()
                t["state"] = "retryable" if t["attempts"] < t["max_attempts"] else "failed"; t["next_attempt_at"] = 0; changed.append(t["id"])
        if changed:
            _save(d, path)
    return changed


def mark_blocked(tasks_path: Path | None, dataset_ready) -> list[str]:
    """Tasks whose dependency failed permanently become blocked (reported, never silently dropped)."""
    changed = []
    with locked(tasks_path):
        d = _load(tasks_path)
        for t in d.values():
            if t["state"] in ("pending", "retryable"):
                st, why = deps_status(t, d, dataset_ready)
                if st == "blocked":
                    t["state"], t["last_error"] = "blocked", why; changed.append(t["id"])
        if changed:
            _save(d, tasks_path)
    return changed
