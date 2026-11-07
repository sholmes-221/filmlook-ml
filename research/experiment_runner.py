"""Run a registered experiment as a subprocess, meter it, extract metrics, enforce the session resource budget."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from research import experiment_registry as R

ROOT = R.ROOT
LEDGER = ROOT / "research" / "ledger.json"
LOGS = ROOT / "reports" / "experiments"
LIMITS = {"gpu_seconds": 4 * 3600, "disk_bytes": 20 * 1024 ** 3}


def _ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"gpu_seconds": 0.0, "disk_bytes": 0, "runs": []}


def _save(l: dict) -> None:
    LEDGER.write_text(json.dumps(l, indent=1))


def remaining() -> dict:
    l = _ledger()
    return {k: LIMITS[k] - l[k] for k in LIMITS}


def _gpu_mem_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except Exception:  # noqa: BLE001
        return None


def _du(paths: list[str]) -> int:
    total = 0
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            total += pp.stat().st_size
        elif pp.is_dir():
            total += sum(f.stat().st_size for f in pp.rglob("*") if f.is_file())
    return total


def run(exp_id: str, result_json: str, uses_gpu: bool = True, watched_paths: list[str] | None = None,
        expected_gpu_seconds: float = 0.0, timeout: int = 4 * 3600) -> dict:
    """Execute the registered command; record wall time, peak GPU memory, disk delta on watched paths, metrics."""
    exp = R.get(exp_id)
    if exp is None:
        raise KeyError(exp_id)
    rem = remaining()
    if uses_gpu and expected_gpu_seconds > rem["gpu_seconds"]:
        R.update(exp_id, status="blocked", reason=f"GPU budget: need {expected_gpu_seconds:.0f}s, {rem['gpu_seconds']:.0f}s left")
        return R.get(exp_id)
    LOGS.mkdir(parents=True, exist_ok=True)
    before_disk = _du(watched_paths or [])
    base_mem = _gpu_mem_mb() or 0.0
    peak = {"mb": base_mem}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            m = _gpu_mem_mb()
            if m is not None:
                peak["mb"] = max(peak["mb"], m)
            stop.wait(2.0)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    R.update(exp_id, status="running", started_at=R._now())
    log = LOGS / f"{exp_id}.log"
    with open(log, "w") as fh:
        proc = subprocess.run(exp["command"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    wall = time.time() - t0
    stop.set(); th.join(timeout=5)
    disk_delta = max(0, _du(watched_paths or []) - before_disk)
    l = _ledger()
    l["gpu_seconds"] += wall if uses_gpu else 0.0
    l["disk_bytes"] += disk_delta
    l["runs"].append({"id": exp_id, "wall_seconds": round(wall, 1), "gpu": uses_gpu, "disk_bytes": disk_delta,
                      "peak_gpu_mb_total": peak["mb"], "returncode": proc.returncode})
    _save(l)
    metrics, missing = {}, []
    if proc.returncode == 0 and Path(result_json).exists():
        res = json.loads(Path(result_json).read_text())
        for name, spec in exp.get("metrics_of_interest", {}).items():
            try:
                metrics[name] = R.dig(res, spec["path"])
            except (KeyError, IndexError, TypeError):
                missing.append(name)
    status = "done" if proc.returncode == 0 else "failed"
    return R.update(exp_id, status=status, returncode=proc.returncode, wall_seconds=round(wall, 1),
                    peak_gpu_mb_total=peak["mb"], disk_bytes=disk_delta, metrics=metrics, metrics_missing=missing,
                    result_json=str(result_json), log=str(log), finished_at=R._now())
