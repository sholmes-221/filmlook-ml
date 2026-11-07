"""Resource awareness for the orchestrator: GPU free memory, a single-GPU-job lock, cost estimates, process liveness."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from research import experiment_registry as R

GPU_LOCK = R.ROOT / "research" / ".gpu.lock"
MIN_FREE_MB_DEFAULT = 4000


def gpu_mem_mb() -> dict | None:
    """{'used': .., 'total': .., 'free': ..} in MB from nvidia-smi, or None without a GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return {"used": used, "total": total, "free": total - used}
    except Exception:  # noqa: BLE001
        return None


_DETECT = object()


def gpu_can_launch(required_mb: float, mem=_DETECT) -> tuple[bool, str]:
    """mem: a gpu_mem_mb() result (None = no GPU) or omitted to detect now."""
    if required_mb <= 0:
        return True, "cpu task"
    mem = gpu_mem_mb() if mem is _DETECT else mem
    if mem is None:
        return False, "no GPU detected"
    if mem["free"] < required_mb:
        return False, f"GPU free {mem['free']:.0f} MB < required {required_mb:.0f} MB"
    return True, f"GPU free {mem['free']:.0f} MB"


class GpuLock:
    """Advisory file lock so two GPU-heavy tasks are never launched together (held for the task's lifetime)."""

    def __init__(self, path: Path | None = None):
        self.path, self.fh = Path(path or GPU_LOCK), None        # GPU_LOCK read at call time (tests redirect it)

    def acquire(self, owner: str) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close(); self.fh = None
            return False
        self.fh.seek(0); self.fh.truncate(); self.fh.write(json.dumps({"owner": owner, "pid": os.getpid(), "since": time.time()})); self.fh.flush()
        return True

    def holder(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path) as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fh, fcntl.LOCK_UN)
                    return None                                   # nobody holds it
                except OSError:
                    fh.seek(0); return json.loads(fh.read() or "{}")
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close(); self.fh = None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                          # zombie check
        with open(f"/proc/{pid}/status") as fh:
            return "State:\tZ" not in fh.read()
    except OSError:
        return True


def estimate(spec: dict) -> dict:
    """Cost estimate from a backlog/task spec; conservative defaults for unknown fields."""
    gpu_min = float(spec.get("gpu_minutes", 0) or 0); cpu_min = float(spec.get("cpu_minutes", 0) or 0)
    mem = float(spec.get("gpu_mb", 0) or (6000 if gpu_min > 0 else 0))
    return {"gpu_minutes": gpu_min, "cpu_minutes": cpu_min, "gpu_mb": mem, "disk_gb": float(spec.get("disk_gb", 0) or 0),
            "wall_minutes_max": float(spec.get("max_minutes", 0) or max(2 * (gpu_min + cpu_min), 10))}
