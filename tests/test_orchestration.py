"""Orchestration: task store (dependencies, retries/backoff, blocked), concurrent task safety, crash recovery, wrapper
exit codes and timeouts, GPU limits in the tick, evaluation triggering on dataset readiness, research-loop selection
and falsification rules, registry/backlog consistency."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from research import data_pipeline as DP
from research import experiment_registry as R
from research import orchestrator as O
from research import research_loop as RL
from research import task_store as T
from research import task_wrapper as W
from research.resources import GpuLock, estimate, gpu_can_launch

PY = sys.executable


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = tmp_path / "tasks.json"
    monkeypatch.setattr(T, "TASKS", p)
    monkeypatch.setattr(W, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(O, "EVENTS", tmp_path / "events.jsonl")
    monkeypatch.setattr(DP, "STATE_PATH", tmp_path / "dataset_state.json")
    import research.resources as RES
    monkeypatch.setattr(RES, "GPU_LOCK", tmp_path / "gpu.lock")
    return p


def test_dependencies_retries_backoff_and_blocking(store):
    T.add("a", "data", ["true"], max_attempts=2, retry_backoff_s=100)
    T.add("b", "eval", ["true"], depends_on=["task:a"])
    T.add("c", "eval", ["true"], depends_on=["dataset:dsX"])
    ready = {"dsX": "pending"}
    assert [t["id"] for t in T.runnable(T.all_tasks(), ready.get)] == ["a"]
    T.update("a", state="running", attempts=1); t = T.finish("a", 3)
    assert t["state"] == "retryable" and t["next_attempt_at"] > time.time() + 50          # backoff applied
    assert T.runnable(T.all_tasks(), ready.get, now=time.time()) == []
    assert [t["id"] for t in T.runnable(T.all_tasks(), ready.get, now=time.time() + 101)] == ["a"]
    T.update("a", attempts=2); assert T.finish("a", 1)["state"] == "failed"               # attempts exhausted
    assert T.mark_blocked(None, ready.get) == ["b"] and T.get("b")["state"] == "blocked"
    ready["dsX"] = "permanent_failure"; assert T.mark_blocked(None, ready.get) == ["c"]
    ready["dsX"] = "ready"; T.update("c", state="pending")
    assert [t["id"] for t in T.runnable(T.all_tasks(), ready.get)] == ["c"]


def test_concurrent_adds_and_updates_are_lock_safe(store):
    def worker(i):
        for j in range(20):
            T.add(f"t{i}_{j}", "research", ["true"]); T.update(f"t{i}_{j}", state="done")
    th = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in th]; [t.join() for t in th]
    d = T.all_tasks(); assert len(d) == 120 and all(t["state"] == "done" for t in d.values())
    assert json.loads(store.read_text()) == d                                             # file intact (atomic writes)


def test_crash_recovery_requeues_running_tasks_with_dead_processes(store):
    T.add("dead", "research", ["true"], max_attempts=2); T.update("dead", state="running", pid=999999, attempts=1)
    T.add("alive", "research", ["true"]); T.update("alive", state="running", pid=os.getpid(), attempts=1)
    T.add("spent", "research", ["true"], max_attempts=1); T.update("spent", state="running", pid=999999, attempts=1)
    changed = T.recover(lambda pid: pid == os.getpid())
    assert set(changed) == {"dead", "spent"} and T.get("dead")["state"] == "retryable" and T.get("spent")["state"] == "failed" and T.get("alive")["state"] == "running"
    DP._set("dsY", "downloading", pid=999999); DP.DATASETS["dsY"] = {"manifest": store}   # dataset acquisition that died
    try:
        assert DP.refresh()["dsY"]["state"] == "retryable_failure"
    finally:
        DP.DATASETS.pop("dsY")


def test_wrapper_records_exit_codes_and_enforces_wall_clock_cap(store, tmp_path):
    T.add("ok", "data", [PY, "-c", "print('hi')"]); assert W.run_task("ok", store) == 0 and T.get("ok")["state"] == "done"
    T.add("paused", "data", [PY, "-c", "import sys; sys.exit(3)"], max_attempts=3); assert W.run_task("paused", store) == 3
    assert T.get("paused")["state"] == "retryable" and T.get("paused")["exit_code"] == 3
    T.add("slow", "research", [PY, "-c", "import time; time.sleep(30)"], wall_minutes_max=0.02, max_attempts=1)
    t0 = time.time(); rc = W.run_task("slow", store)
    assert rc == 124 and time.time() - t0 < 25 and T.get("slow")["state"] == "failed" and "wall-clock" in T.get("slow")["last_error"]
    assert (tmp_path / "logs" / "ok.log").read_text().strip().endswith("hi")


def test_gpu_lock_blocks_second_gpu_task(store, tmp_path):
    lk = GpuLock(tmp_path / "gpu.lock"); assert lk.acquire("other")
    try:
        T.add("g", "research", [PY, "-c", "print(1)"], gpu_mb=1000); assert W.run_task("g", store) == 3 and T.get("g")["last_error"] == "GPU lock busy"
        assert GpuLock(tmp_path / "gpu.lock").holder()["owner"] == "other"
    finally:
        lk.release()
    assert GpuLock(tmp_path / "gpu.lock").holder() is None


def test_tick_respects_gpu_memory_parallelism_and_dataset_readiness(store, monkeypatch):
    monkeypatch.setattr(O, "gpu_mem_mb", lambda: {"used": 20000.0, "total": 24000.0, "free": 4000.0})
    monkeypatch.setattr(O, "dataset_state", lambda name: {"dsA": "retryable_failure", "dsB": "ready"}.get(name, "missing"))
    T.add("cpu1", "data", ["true"]); T.add("cpu2", "data", ["true"]); T.add("cpu3", "data", ["true"])
    T.add("gpu_big", "research", ["true"], gpu_mb=6000); T.add("gpu_small", "research", ["true"], gpu_mb=1000)
    T.add("eval_A", "eval", ["true"], depends_on=["dataset:dsA"]); T.add("eval_B", "eval", ["true"], depends_on=["dataset:dsB"], gpu_mb=500)
    rep = O.tick(max_parallel=4, dry_run=True)
    assert "eval_A" not in rep["launched"] and "eval_B" in rep["launched"]                 # evaluation triggers on readiness only
    assert "gpu_big" not in rep["launched"] and any(s[0] == "gpu_big" for s in rep["skipped"])
    assert sum(1 for t in rep["launched"] if t in ("gpu_small", "eval_B")) == 1            # one GPU task per tick
    assert len(rep["launched"]) <= 4
    T.update("eval_B", state="done"); T.update("gpu_small", state="done")
    rep = O.tick(max_parallel=4, dry_run=True)                                              # memory is now the only barrier
    assert any(s[0] == "gpu_big" and "GPU free" in s[1] for s in rep["skipped"])
    monkeypatch.setattr(O, "gpu_mem_mb", lambda: None)
    rep = O.tick(max_parallel=10, dry_run=True)
    assert all(t not in rep["launched"] for t in ("gpu_big", "gpu_small", "eval_B"))       # no GPU: GPU tasks wait


def test_tick_launches_through_wrapper_and_records_outcome(store, monkeypatch):
    monkeypatch.setattr(O, "gpu_mem_mb", lambda: None)
    T.add("real", "data", [PY, "-c", "print('done')"])
    rep = O.tick(max_parallel=1, tasks_path=store)
    assert rep["launched"] == ["real"]
    for _ in range(100):
        if T.get("real", store)["state"] == "done":
            break
        time.sleep(0.2)
    assert T.get("real", store)["state"] == "done" and T.get("real", store)["exit_code"] == 0
    ev = [json.loads(l)["event"] for l in O.EVENTS.read_text().splitlines()]; assert "launch" in ev and "tick" in ev


def test_estimate_and_gpu_checks(monkeypatch):
    e = estimate({"gpu_minutes": 8}); assert e["gpu_mb"] == 6000 and e["wall_minutes_max"] >= 16
    import research.resources as RES
    monkeypatch.setattr(RES, "gpu_mem_mb", lambda: None)
    assert gpu_can_launch(0, None)[0] and not gpu_can_launch(100, None)[0]                 # no GPU detected
    assert gpu_can_launch(1000, {"free": 1500})[0] and not gpu_can_launch(2000, {"free": 1500})[0]


@pytest.fixture
def reg(tmp_path, monkeypatch):
    from research import experiment_runner as RR
    monkeypatch.setattr(R, "REGISTRY", tmp_path / "registry.jsonl"); monkeypatch.setattr(RR, "LEDGER", tmp_path / "ledger.json"); monkeypatch.setattr(RR, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(RL, "OUT", tmp_path / "rl"); monkeypatch.setattr(RL, "LOG_MD", tmp_path / "LOG.md"); monkeypatch.setattr(DP, "STATE_PATH", tmp_path / "ds.json")
    return tmp_path


def test_research_loop_selects_only_runnable_specs_and_applies_falsification(reg, tmp_path, monkeypatch):
    out = tmp_path / "res.json"
    bl = tmp_path / "backlog.yaml"
    bl.write_text(f"""experiments:
  - id: no_spec
    hypothesis: needs a script
    value: 5
    gpu_minutes: 1
  - id: waits_data
    hypothesis: needs commons
    value: 5
    gpu_minutes: 0
    datasets: [commons_film]
    command: [true]
    result_json: x.json
    metrics_of_interest: {{m: {{path: m, kind: real}}}}
  - id: runnable
    hypothesis: cheap and runnable
    value: 3
    gpu_minutes: 0
    cpu_minutes: 1
    command: ["{PY}", "-c", "import json; json.dump({{'m': 0.3, 'beats': 0}}, open(r'{out}', 'w'))"]
    result_json: {out}
    metrics_of_interest: {{m: {{path: m, kind: real}}, beats: {{path: beats, kind: proxy}}}}
    falsify: {{metric: beats, below: 0.5}}
""")
    monkeypatch.setattr(RL.planner, "BACKLOG", bl)
    (tmp_path / "LOG.md").write_text("# log\n\n## Decisions taken from these runs\n")
    sel, needs = RL.select(bl)
    assert sel["id"] == "runnable" and {n["id"] for n in needs} == {"no_spec", "waits_data"}
    dry = RL.cycle(dry_run=True, path=bl); assert dry["selected"] == "runnable" and dry["estimate"]["gpu_mb"] == 0
    rep = RL.cycle(path=bl)
    assert rep["decision"] == "rejected" and "falsified" in rep["reason"] and R.get("runnable")["decision"] == "rejected"
    import yaml
    st = {e["id"]: e.get("status") for e in yaml.safe_load(bl.read_text())["experiments"]}; assert st["runnable"] == "rejected"
    assert "### runnable" in (tmp_path / "LOG.md").read_text() and rep["log_appended"]
    assert RL.cycle(path=bl)["selected"] is None                                           # not re-run once decided
    assert RL.apply_falsify({"metric": "x", "above": 0.5}, {"x": 0.7}) == (True, "x=0.7000 > 0.5")


def test_registry_and_backlog_consistency_in_repo():
    """Every backlog entry with a terminal status that also appears in the registry agrees on the outcome kind."""
    from research import planner
    reg = {r["id"]: r for r in R.load()}
    for e in planner.load_backlog():
        r = reg.get(e["id"])
        if r and e.get("status") in ("rejected", "supported") and r.get("decision"):
            assert r["decision"] == e["status"], e["id"]
        if r and r.get("status") == "running":
            pytest.fail(f"{e['id']} marked running in the registry (crash without recovery?)")
