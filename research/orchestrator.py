"""Local-first orchestration of the three pipelines (data, research, evaluation).

  python -m research.orchestrator seed                 # register the standard tasks (idempotent)
  python -m research.orchestrator status
  python -m research.orchestrator tick                 # one scheduling pass: recover, launch what is runnable, report
  python -m research.orchestrator loop --interval 900 [--max-hours 20] [--started-at 2026-11-07T21:47:57 | --resume]
      # periodic ticks for a bounded runtime (default 20 h = 72,000 s from the start time); SIGTERM/SIGINT stop it
      # gracefully after the current tick; detached tasks keep running; state in reports/orchestrator/loop_state.json
  python -m research.orchestrator add <id> <pipeline> --command "..." [--depends dataset:commons_film] [--gpu-mb 6000]

Each tick: crash recovery (running tasks with dead processes are re-queued), dataset refresh, dependency check
(task:<id> done, dataset:<id> ready), resource check (GPU free memory from nvidia-smi, one GPU task at a time via the
GPU lock, at most `max_parallel` tasks), launch through research.task_wrapper (detached; outcome recorded by the
wrapper). Data tasks never block research or evaluation tasks whose dependencies are met. Events are appended to
reports/orchestrator/events.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from research import data_pipeline as DP
from research import experiment_registry as R
from research import task_store as T
from research.resources import GpuLock, gpu_can_launch, gpu_mem_mb, pid_alive

EVENTS = R.ROOT / "reports" / "orchestrator" / "events.jsonl"
PY = sys.executable


def event(kind: str, **fields) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **fields}) + "\n")


def dataset_state(name: str) -> str:
    return DP.status(name).get("state", "pending") if name in DP.DATASETS else "missing"


def seed(auto_research: bool = False) -> list[str]:
    """Standard tasks. Data: Commons acquisition (resumable, retried with backoff). Eval: real-film evaluation of the
    eligible candidates on the analysis half once commons_film is ready. Research: one research-loop cycle (optional)."""
    made = []
    made.append(T.add("data_commons_film_acquire", "data", [PY, "-m", "research.data_pipeline", "acquire", "commons_film"], max_attempts=12,
                      wall_minutes_max=300, retry_backoff_s=1800)["id"])
    made.append(T.add("data_validate_existing", "data", [PY, "-m", "research.data_pipeline", "validate-all"], max_attempts=2, wall_minutes_max=30)["id"])
    made.append(T.add("eval_commons_film_analysis", "eval", [PY, "-m", "research.eval_pipeline", "run", "--dataset", "commons_film", "--split", "analysis"],
                      depends_on=["dataset:commons_film"], gpu_mb=3000, max_attempts=3, wall_minutes_max=60, retry_backoff_s=600)["id"])
    # second real-film source (CommonCatalog film scans with uploader stock tags): metadata scan, then resumable acquisition, then evaluation
    made.append(T.add("data_commoncatalog_film_scan", "data", [PY, "scripts/ingest_commoncatalog_film.py", "--scan", "--shards", "300"], max_attempts=3, wall_minutes_max=120, retry_backoff_s=900)["id"])
    made.append(T.add("data_commoncatalog_film_acquire", "data", [PY, "-m", "research.data_pipeline", "acquire", "commoncatalog_film"], depends_on=["task:data_commoncatalog_film_scan"],
                      max_attempts=8, wall_minutes_max=130, retry_backoff_s=600)["id"])
    made.append(T.add("eval_commoncatalog_film_analysis", "eval", [PY, "-m", "research.eval_pipeline", "run", "--dataset", "commoncatalog_film", "--split", "analysis"],
                      depends_on=["dataset:commoncatalog_film"], gpu_mb=3000, max_attempts=3, wall_minutes_max=60, retry_backoff_s=600)["id"])
    # ARTeFACT (figshare, CC BY 4.0, class A) and the curated CC0 museum photographs (Art Institute of Chicago)
    made.append(T.add("data_artefact_film_acquire", "data", [PY, "-m", "research.data_pipeline", "acquire", "artefact_film"], max_attempts=4, wall_minutes_max=70, retry_backoff_s=900)["id"])
    made.append(T.add("data_aic_photographs_acquire", "data", [PY, "-m", "research.data_pipeline", "acquire", "aic_photographs"], max_attempts=4, wall_minutes_max=100, retry_backoff_s=900)["id"])
    made.append(T.add("eval_aic_photographs_analysis", "eval", [PY, "scripts/eval_curated_photos.py", "--features", "--split", "analysis"],
                      depends_on=["dataset:aic_photographs"], gpu_mb=3000, max_attempts=3, wall_minutes_max=45, retry_backoff_s=600)["id"])
    if auto_research:
        made.append(T.add("research_cycle", "research", [PY, "-m", "research.research_loop", "--once"], gpu_mb=6000, max_attempts=1, wall_minutes_max=120)["id"])
    return made


def launch(task: dict, tasks_path: Path | None = None) -> int:
    log_dir = R.ROOT / "reports" / "orchestrator"; log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "research.task_wrapper", task["id"]] + (["--tasks", str(tasks_path)] if tasks_path else [])
    p = subprocess.Popen(cmd, cwd=R.ROOT, stdout=open(log_dir / f"{task['id']}.wrapper.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    T.update(task["id"], tasks_path, state="running", pid=p.pid, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    event("launch", task=task["id"], pid=p.pid, pipeline=task["pipeline"])
    return p.pid


def tick(max_parallel: int = 3, tasks_path: Path | None = None, dry_run: bool = False, min_free_mb_margin: float = 500) -> dict:
    recovered = T.recover(pid_alive, tasks_path)
    for tid in recovered:
        event("recovered", task=tid)
    DP.refresh()
    blocked = T.mark_blocked(tasks_path, dataset_state)
    for tid in blocked:
        event("blocked", task=tid, reason=T.get(tid, tasks_path)["last_error"])
    tasks = T.all_tasks(tasks_path)
    running = [t for t in tasks.values() if t["state"] == "running"]
    gpu_busy = any(t.get("gpu_mb", 0) > 0 for t in running) or GpuLock().holder() is not None
    mem = gpu_mem_mb(); launched, skipped = [], []
    for t in T.runnable(tasks, dataset_state):
        if len(running) + len(launched) >= max_parallel:
            skipped.append((t["id"], "max_parallel")); continue
        if t.get("gpu_mb", 0) > 0:
            if gpu_busy:
                skipped.append((t["id"], "GPU task already running")); continue
            ok, why = gpu_can_launch(t["gpu_mb"] + min_free_mb_margin, mem)
            if not ok:
                skipped.append((t["id"], why)); continue
            gpu_busy = True
        if dry_run:
            launched.append(t["id"]); continue
        launch(t, tasks_path); launched.append(t["id"])
    rep = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "recovered": recovered, "blocked": blocked, "running": [t["id"] for t in running],
           "launched": launched, "skipped": skipped, "gpu": mem, "datasets": {d: s.get("state") for d, s in DP.status().items()}}
    event("tick", **{k: rep[k] for k in ("recovered", "blocked", "running", "launched", "skipped")})
    return rep


def status_text(tasks_path: Path | None = None) -> str:
    tasks = T.all_tasks(tasks_path); lines = ["TASKS"]
    for t in sorted(tasks.values(), key=lambda t: (t["pipeline"], t["created"])):
        nxt = f" next {time.strftime('%H:%M', time.localtime(t['next_attempt_at']))}" if t["state"] == "retryable" else ""
        lines.append(f"  {t['pipeline']:8s} {t['id']:32s} {t['state']:10s} attempts {t['attempts']}/{t['max_attempts']}{nxt} {t.get('last_error') or ''}")
    lines.append("DATASETS")
    for d, s in DP.status().items():
        lines.append(f"  {d:22s} {s.get('state', 'pending'):18s} rows={s.get('rows', '-')} {s.get('last_error') or ''}")
    mem = gpu_mem_mb(); lines.append(f"GPU: {mem} lock holder: {GpuLock().holder()}")
    return "\n".join(lines)


LOOP_STATE = R.ROOT / "reports" / "orchestrator" / "loop_state.json"
DEFAULT_MAX_HOURS = 20.0


def loop_deadline(started_at: float, max_hours: float = DEFAULT_MAX_HOURS) -> float:
    """Absolute stop time (epoch seconds) for a loop that started at `started_at` (default budget 20 h = 72,000 s)."""
    return started_at + max_hours * 3600.0


def run_loop(interval: float = 900, max_parallel: int = 3, max_hours: float = DEFAULT_MAX_HOURS, max_ticks: int = 0,
             started_at: float | None = None, tasks_path: Path | None = None, state_path: Path | None = None,
             clock=time.time, sleep=time.sleep, tick_fn=None, install_signals: bool = True, poll_seconds: float = 5.0) -> dict:
    """Periodic ticks until the runtime budget (`max_hours` after `started_at`, default now) is spent, `max_ticks` is
    reached, or SIGTERM/SIGINT arrives (graceful: the current tick finishes, state is written, tasks keep running
    detached). A tick that raises is logged and the loop continues; the loop state (pid, start, deadline, ticks) is
    persisted to reports/orchestrator/loop_state.json so a restarted loop can keep the original start time
    (`--started-at` / `--resume`)."""
    import os
    import signal
    state_path = state_path or LOOP_STATE; tick_fn = tick_fn or (lambda: tick(max_parallel, tasks_path))
    started = float(started_at) if started_at is not None else clock(); deadline = loop_deadline(started, max_hours)
    stop = {"reason": None}

    def _write(**extra):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        st = {"pid": os.getpid(), "started_at": started, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)), "max_hours": max_hours,
              "deadline_at": deadline, "deadline": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(deadline)), "interval": interval, "ticks": n,
              "last_tick": last, "stop_reason": stop["reason"], "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(clock())), **extra}
        tmp = state_path.with_suffix(".json.part"); tmp.write_text(json.dumps(st, indent=1, default=str)); tmp.replace(state_path)
        return st

    def _on_signal(signum, _frame):
        stop["reason"] = stop["reason"] or f"signal {signum}"

    if install_signals:
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, _on_signal)
    n, last, errors = 0, None, 0
    _write(); event("loop_start", pid=os.getpid(), started=started, deadline=deadline, max_hours=max_hours)
    while stop["reason"] is None:
        if clock() >= deadline:
            stop["reason"] = "deadline"; break
        try:
            rep = tick_fn(); last = rep.get("ts")
            print(f"{rep['ts']} launched={rep['launched']} running={rep['running']} skipped={rep['skipped']} datasets={rep['datasets']}", flush=True)
        except Exception as ex:                                            # a failing tick must not end the loop
            errors += 1; last = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(clock())); event("tick_error", error=repr(ex)); print(f"{last} tick error: {ex!r}", flush=True)
        n += 1; _write(tick_errors=errors)
        if max_ticks and n >= max_ticks:
            stop["reason"] = "max_ticks"; break
        wake = min(clock() + interval, deadline)
        while stop["reason"] is None and clock() < wake:                 # interruptible sleep (graceful shutdown)
            sleep(min(poll_seconds, max(wake - clock(), 0)))
    st = _write(tick_errors=errors); event("loop_stop", reason=stop["reason"], ticks=n)
    print(f"loop stopped: {stop['reason']} after {n} ticks (deadline {st['deadline']})", flush=True)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seed", "status", "tick", "loop", "add"])
    ap.add_argument("task_id", nargs="?"); ap.add_argument("pipeline", nargs="?")
    ap.add_argument("--command", default=None); ap.add_argument("--depends", default=""); ap.add_argument("--gpu-mb", type=float, default=0)
    ap.add_argument("--interval", type=float, default=900); ap.add_argument("--max-parallel", type=int, default=3); ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS, help="runtime budget of `loop` from its start time (default 20 h)")
    ap.add_argument("--started-at", default=None, help="epoch seconds or ISO local time of the original start (keeps the budget of a restarted loop)")
    ap.add_argument("--resume", action="store_true", help="take the start time from reports/orchestrator/loop_state.json")
    ap.add_argument("--auto-research", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "seed":
        print("tasks:", seed(a.auto_research))
    elif a.cmd == "status":
        print(status_text())
    elif a.cmd == "add":
        import shlex
        print(json.dumps(T.add(a.task_id, a.pipeline, shlex.split(a.command), [d for d in a.depends.split(",") if d], gpu_mb=a.gpu_mb), indent=1))
    elif a.cmd == "tick":
        print(json.dumps(tick(a.max_parallel, dry_run=a.dry_run), indent=1, default=str))
    elif a.cmd == "loop":
        started = None
        if a.resume and LOOP_STATE.exists():
            started = json.loads(LOOP_STATE.read_text()).get("started_at")
        if a.started_at:
            try:
                started = float(a.started_at)
            except ValueError:
                started = time.mktime(time.strptime(a.started_at, "%Y-%m-%dT%H:%M:%S"))
        run_loop(a.interval, a.max_parallel, a.max_hours, a.max_ticks, started)


if __name__ == "__main__":
    main()
