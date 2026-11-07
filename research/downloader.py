"""Resumable, rate-limit-aware file downloader for the data pipeline.

Per-item state is persisted in an append-only JSONL file (last record per id wins), so an interrupted run resumes
without re-downloading anything that was validated. Files are written atomically (``<dst>.part`` then rename), decoded
and checked by a caller-supplied validator before they count, and recorded with their sha256.

Item states (``STATES``): pending -> downloading -> validated; failures are ``retryable_failure`` (network error,
5xx, rate limit after the bounded wait), ``permanent_failure`` (4xx other than 429, exhausted attempts, no usable
URL) and ``quarantined`` (the bytes arrived but did not decode as a valid image).

Loop control distinguishes three things that look alike from the outside:
  * progress          a file was validated (``last_success``)
  * alive but slow    HTTP responses keep arriving, including 429 (``last_response``); bounded waits, never a stall
  * stalled           no HTTP response at all for ``stall_seconds`` (connection/read timeouts) -> exit ``stalled``
A ``max_seconds`` wall-clock cap exits with ``paused`` (resumable). A heartbeat line is logged every
``heartbeat_seconds`` even when nothing completes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable

STATES = ("pending", "downloading", "validated", "retryable_failure", "permanent_failure", "quarantined")
RUN_STATUSES = ("complete", "paused", "stalled", "rate_limited")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadState:
    """Append-only per-item state (JSONL). ``get(id)`` returns the merged latest record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["id"]] = {**self.rows.get(r["id"], {}), **r}

    def get(self, item_id: str) -> dict:
        return self.rows.get(item_id, {"id": item_id, "status": "pending", "attempts": 0})

    def set(self, item_id: str, status: str, **fields) -> dict:
        assert status in STATES, status
        row = {**self.get(item_id), **fields, "id": item_id, "status": status, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.rows[item_id] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def counts(self) -> dict:
        return dict(Counter(r["status"] for r in self.rows.values()))


class Downloader:
    def __init__(self, state_path: Path, *, session=None, headers: dict | None = None, connect_timeout: float = 10, read_timeout: float = 90,
                 max_attempts: int = 4, spacing: float = 0.5, max_retry_after: float = 120, max_rate_limit_waits: int = 20,
                 stall_seconds: float = 1200, max_seconds: float | None = None, heartbeat_seconds: float = 60, log_path: Path | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time, checkpoint_every: int = 100,
                 on_checkpoint: Callable[[], None] | None = None):
        if session is None:
            import requests
            session = requests.Session()
        self.s, self.headers, self.state = session, headers or {}, DownloadState(state_path)
        self.timeout = (connect_timeout, read_timeout)
        self.max_attempts, self.spacing, self.max_retry_after, self.max_rl_waits = max_attempts, spacing, max_retry_after, max_rate_limit_waits
        self.stall_seconds, self.max_seconds, self.heartbeat_seconds = stall_seconds, max_seconds, heartbeat_seconds
        self.log_path, self.sleep, self.clock = log_path, sleep, clock
        self.checkpoint_every, self.on_checkpoint = checkpoint_every, on_checkpoint
        self.stats = Counter()

    # ---------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(line + "\n")

    # ---------------------------------------------------------------- one request
    def fetch(self, url: str) -> tuple[str, bytes | None, float | None]:
        """-> (kind, content, retry_after). kind: ok | rate_limited | server_error | client_error | network_error."""
        try:
            r = self.s.get(url, headers=self.headers, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001  (connect/read timeout, connection reset, DNS)
            self.stats["network_error"] += 1
            return "network_error", None, None
        self.last_response = self.clock()
        if r.status_code == 200:
            return "ok", r.content, None
        if r.status_code in (429, 503):
            self.stats["rate_limited"] += 1
            ra = r.headers.get("Retry-After")
            try:
                ra = float(ra) if ra is not None else None
            except ValueError:
                ra = None
            return "rate_limited", None, ra
        if 500 <= r.status_code < 600:
            self.stats["server_error"] += 1
            return "server_error", None, None
        self.stats[f"http_{r.status_code}"] += 1
        return "client_error", None, None

    def _wait(self, seconds: float, why: str) -> None:
        self.log(f"  waiting {seconds:.0f}s ({why})")
        end = self.clock() + seconds
        while True:
            left = end - self.clock()
            if left <= 0:
                return
            self.sleep(min(left, self.heartbeat_seconds)); self._heartbeat()

    # ---------------------------------------------------------------- one item
    def fetch_item(self, item: dict, validate: Callable[[bytes, Path], None]) -> str:
        """item: {id, urls: [..], dst: Path}. validate(bytes, tmp_path) must write the stored file or raise.
        Returns the new state."""
        iid, dst = item["id"], Path(item["dst"])
        rec = self.state.get(iid)
        if rec["status"] == "validated" and dst.exists() and (not rec.get("sha256") or rec["sha256"] == sha256_file(dst)):
            self.stats["skipped_done"] += 1; return "validated"
        if rec["status"] in ("permanent_failure", "quarantined"):
            self.stats["skipped_failed"] += 1; return rec["status"]
        if dst.exists() and rec["status"] != "validated":                 # legacy file from a run without state: check, adopt
            try:
                with open(dst, "rb") as fh:
                    validate(fh.read(), dst.with_suffix(".part"))
                dst.with_suffix(".part").unlink(missing_ok=True)
                self.state.set(iid, "validated", sha256=sha256_file(dst), adopted=True); self.stats["adopted"] += 1
                return "validated"
            except Exception as e:  # noqa: BLE001
                dst.with_suffix(".part").unlink(missing_ok=True)
                self.state.set(iid, "quarantined", error=f"existing file invalid: {e}"[:200]); self.stats["quarantined"] += 1
                return "quarantined"
        urls = [u for u in item.get("urls", []) if u]
        if not urls:
            self.state.set(iid, "permanent_failure", error="no_url"); self.stats["no_url"] += 1; return "permanent_failure"
        attempts = int(rec.get("attempts", 0))
        if attempts >= self.max_attempts:
            self.state.set(iid, "permanent_failure", error="attempts exhausted"); self.stats["exhausted"] += 1; return "permanent_failure"
        self.state.set(iid, "downloading", attempts=attempts)
        rl_waits, err = 0, "unknown"
        for ui, url in enumerate(urls):
            for attempt in range(attempts, self.max_attempts):
                self.sleep(self.spacing)
                kind, content, ra = self.fetch(url)
                if kind == "ok":
                    tmp = dst.with_suffix(".part")
                    try:
                        validate(content, tmp)
                        if not tmp.exists() or tmp.stat().st_size == 0:
                            raise ValueError("validator wrote nothing")
                        os.replace(tmp, dst)                                   # atomic: readers never see a partial file
                    except Exception as e:  # noqa: BLE001
                        tmp.unlink(missing_ok=True)
                        self.state.set(iid, "quarantined", error=str(e)[:200], url=url); self.stats["quarantined"] += 1
                        return "quarantined"
                    self.state.set(iid, "validated", sha256=sha256_file(dst), url=url, attempts=attempt + 1, bytes=dst.stat().st_size)
                    self.stats["validated"] += 1; self.last_success = self.clock()
                    return "validated"
                if kind == "rate_limited":
                    rl_waits += 1; self.rl_streak += 1
                    if rl_waits > 2 and ui + 1 < len(urls):
                        break                                                  # try the alternative rendition instead
                    if self.rl_streak > self.max_rl_waits:
                        self.state.set(iid, "retryable_failure", error="rate limited", attempts=attempt, url=url); self.stop_reason = "rate_limited"
                        return "retryable_failure"
                    self._wait(min(self.max_retry_after, ra if ra else 5 * 2 ** attempt), f"HTTP 429/503 on {url[-60:]}")
                    continue
                if kind == "client_error":
                    err = f"http error on {url[-60:]}"; break                   # next URL, if any
                if kind == "network_error":
                    err = "network error"
                    if self.last_response is not None and self.clock() - self.last_response > self.stall_seconds:
                        self.state.set(iid, "retryable_failure", error=err, attempts=attempt + 1); self.stop_reason = "stalled"
                        return "retryable_failure"
                    self._wait(min(self.max_retry_after, 2 * 2 ** attempt), err); continue
                if kind == "server_error":
                    err = "server error"; self._wait(min(self.max_retry_after, 5 * 2 ** attempt), err); continue
            else:
                continue
        attempts_now = min(self.max_attempts, attempts + 1)
        status = "permanent_failure" if (err.startswith("http error") or attempts_now >= self.max_attempts) else "retryable_failure"
        self.state.set(iid, status, error=err, attempts=attempts_now); self.stats[status] += 1
        return status

    # ---------------------------------------------------------------- loop
    def _heartbeat(self, force: bool = False) -> None:
        now = self.clock()
        if force or now - self.last_heartbeat >= self.heartbeat_seconds:
            self.last_heartbeat = now
            age = f"{now - self.last_success:.0f}s" if self.last_success else "never"
            self.log(f"heartbeat: {self.done}/{self.total} done this run, last success {age} ago, "
                     f"elapsed {now - self.t0:.0f}s, stats {dict(self.stats)}")
            self.write_status("running")

    def write_status(self, run_status: str) -> dict:
        st = {"run_status": run_status, "done_this_run": self.done, "total_items": self.total, "counts": self.state.counts(),
              "stats": dict(self.stats), "elapsed_seconds": round(self.clock() - self.t0, 1),
              "last_success": self.last_success, "last_response": self.last_response, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if self.log_path:
            (self.log_path.parent / "download_status.json").write_text(json.dumps(st, indent=1))
        return st

    def run(self, items: list[dict], validate: Callable[[bytes, Path], None]) -> dict:
        self.t0 = self.clock(); self.last_heartbeat = self.t0; self.last_success = None; self.last_response = self.t0
        self.done, self.total, self.rl_streak, self.stop_reason = 0, len(items), 0, None
        pending = [it for it in items if self.state.get(it["id"])["status"] not in ("permanent_failure", "quarantined")]
        self.log(f"download run: {len(items)} items, {len(items) - len(pending)} already failed permanently, "
                 f"state {self.state.counts()}")
        for it in pending:
            if self.max_seconds is not None and self.clock() - self.t0 > self.max_seconds:
                self.stop_reason = "paused"; break
            st = self.fetch_item(it, validate)
            if st == "validated":
                self.done += 1
                if self.stats["validated"] and self.stats["validated"] % self.checkpoint_every == 0 and self.on_checkpoint:
                    self.on_checkpoint()
                if self.stats["validated"]:
                    self.rl_streak = 0
            if self.stop_reason in ("stalled", "rate_limited"):
                break
            self._heartbeat()
        remaining = sum(1 for it in items if self.state.get(it["id"])["status"] in ("pending", "downloading", "retryable_failure"))
        run_status = self.stop_reason or ("complete" if remaining == 0 else "paused")
        self._heartbeat(force=True)
        st = self.write_status(run_status); st["remaining"] = remaining
        self.log(f"run finished: {run_status}, remaining {remaining}, counts {st['counts']}")
        if self.on_checkpoint:
            self.on_checkpoint()
        return st
