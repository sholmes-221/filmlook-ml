"""Runs one orchestrator task in its own process and records the outcome in the task store.

  python -m research.task_wrapper <task_id> [--tasks path]

Holds the GPU lock for GPU tasks (exit 3 = retryable when it is busy), enforces the task's wall-clock cap (the process
group is killed; outcome retryable), streams output to reports/orchestrator/<task_id>.log.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from research import experiment_registry as R
from research import task_store as T
from research.resources import GpuLock

LOGS = R.ROOT / "reports" / "orchestrator"


def run_task(task_id: str, tasks_path: Path | None = None) -> int:
    t = T.get(task_id, tasks_path)
    if t is None:
        print(f"unknown task {task_id}", file=sys.stderr); return 2
    LOGS.mkdir(parents=True, exist_ok=True); log = LOGS / f"{task_id}.log"
    lock = GpuLock()
    if t.get("gpu_mb", 0) > 0 and not lock.acquire(task_id):
        T.finish(task_id, 3, tasks_path, error="GPU lock busy"); return 3
    T.update(task_id, tasks_path, state="running", pid=os.getpid(), started=time.strftime("%Y-%m-%dT%H:%M:%S"), attempts=t["attempts"] + 1, log=str(log))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(t.get("env") or {})}
    rc, err = 1, None
    with open(log, "a") as fh:
        fh.write(f"\n=== {task_id} attempt {t['attempts'] + 1} {time.strftime('%Y-%m-%dT%H:%M:%S')}: {' '.join(t['command'])}\n"); fh.flush()
        proc = subprocess.Popen(t["command"], cwd=t.get("cwd") or R.ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=float(t.get("wall_minutes_max", 240)) * 60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            rc, err = 124, f"wall-clock cap {t.get('wall_minutes_max')} min exceeded"
            fh.write(f"=== {err}\n")
    lock.release()
    T.finish(task_id, rc, tasks_path, error=err)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("task_id"); ap.add_argument("--tasks", default=None)
    a = ap.parse_args()
    raise SystemExit(run_task(a.task_id, Path(a.tasks) if a.tasks else None))
