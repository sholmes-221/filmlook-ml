"""Data pipeline: resumable downloader (interruption/resume, duplicates, timeouts and retries, corrupt files, rate limits,
stall and runtime caps, atomic writes) and the dataset state machine."""
import json
import sys
from pathlib import Path

import pytest

from research import data_pipeline as DP
from research.downloader import Downloader, DownloadState, sha256_file

GOOD = b"IMG:" + b"x" * 100


class Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}


class FakeSession:
    """Scripted responses per URL: a list consumed in order (last one repeats). An Exception instance is raised."""

    def __init__(self, script):
        self.script, self.calls = {k: list(v) for k, v in script.items()}, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        seq = self.script.get(url, [Resp(404)])
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        return r


def validate(content, tmp):
    if not content.startswith(b"IMG:"):
        raise ValueError("not an image")
    Path(tmp).write_bytes(content)


def mk(tmp_path, session, clock, **kw):
    args = dict(session=session, spacing=0.0, heartbeat_seconds=30, sleep=clock.sleep, clock=clock.now, log_path=tmp_path / "dl.log")
    args.update(kw)
    return Downloader(tmp_path / "state.jsonl", **args)


def item(tmp_path, i, urls=None):
    return {"id": f"f{i}", "urls": urls or [f"http://x/{i}"], "dst": tmp_path / "img" / f"f{i}.jpg"}


def test_resume_skips_validated_and_retries_failed(tmp_path):
    clock = Clock()
    s = FakeSession({"http://x/1": [Resp(200, GOOD)], "http://x/2": [TimeoutError("read")]})
    (tmp_path / "img").mkdir()
    items = [item(tmp_path, 1), item(tmp_path, 2)]
    r = mk(tmp_path, s, clock, max_attempts=2, stall_seconds=10 ** 6).run(items, validate)
    assert r["counts"] == {"validated": 1, "permanent_failure": 1} or r["counts"].get("validated") == 1
    n1 = s.calls.count("http://x/1")
    # second run: network back for file 2; file 1 must not be fetched again (state + sha256 match)
    s2 = FakeSession({"http://x/1": [Resp(200, GOOD)], "http://x/2": [Resp(200, GOOD)]})
    st = DownloadState(tmp_path / "state.jsonl")
    if st.get("f2")["status"] == "permanent_failure":              # attempts exhausted in run 1: reset is explicit
        st.set("f2", "retryable_failure", attempts=0)
    r2 = mk(tmp_path, s2, clock, max_attempts=2, stall_seconds=10 ** 6).run(items, validate)
    assert r2["run_status"] == "complete" and "http://x/1" not in s2.calls and n1 == 1
    assert DownloadState(tmp_path / "state.jsonl").get("f2")["status"] == "validated"


def test_duplicate_ids_and_legacy_files_are_not_refetched(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    legacy = tmp_path / "img" / "f9.jpg"; legacy.write_bytes(GOOD)         # from a run without state: adopted after validation
    s = FakeSession({"http://x/1": [Resp(200, GOOD)]})
    r = mk(tmp_path, s, clock).run([item(tmp_path, 1), item(tmp_path, 1), item(tmp_path, 9)], validate)
    assert s.calls.count("http://x/1") == 1 and "http://x/9" not in s.calls
    st = DownloadState(tmp_path / "state.jsonl")
    assert st.get("f9")["status"] == "validated" and st.get("f9")["sha256"] == sha256_file(legacy) and st.get("f9").get("adopted")
    assert r["stats"]["skipped_done"] == 1


def test_timeouts_retry_with_backoff_then_succeed(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({"http://x/1": [TimeoutError(), ConnectionError(), Resp(200, GOOD)]})
    dl = mk(tmp_path, s, clock, max_attempts=4, stall_seconds=10 ** 6)
    r = dl.run([item(tmp_path, 1)], validate)
    assert r["run_status"] == "complete" and dl.state.get("f1")["attempts"] == 3 and dl.stats["network_error"] == 2
    assert clock.t - 1000 >= 2 + 4                                           # 2*2**0 + 2*2**1 backoff waited


def test_corrupt_download_is_quarantined_without_partial_file(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({"http://x/1": [Resp(200, b"<html>not an image")]})
    dl = mk(tmp_path, s, clock)
    dl.run([item(tmp_path, 1)], validate)
    assert dl.state.get("f1")["status"] == "quarantined"
    assert not (tmp_path / "img" / "f1.jpg").exists() and not list((tmp_path / "img").glob("*.part"))
    # a later run does not retry a quarantined item
    s2 = FakeSession({"http://x/1": [Resp(200, GOOD)]}); mk(tmp_path, s2, clock).run([item(tmp_path, 1)], validate)
    assert s2.calls == []


def test_rate_limit_honours_retry_after_bounded_and_falls_back_to_alternative_url(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({"http://x/a": [Resp(429, headers={"Retry-After": "600"}), Resp(429, headers={"Retry-After": "600"}), Resp(429, headers={"Retry-After": "600"})],
                     "http://x/b": [Resp(200, GOOD)]})
    dl = mk(tmp_path, s, clock, max_retry_after=20, max_attempts=6, stall_seconds=10 ** 6)
    t0 = clock.t; r = dl.run([item(tmp_path, 1, ["http://x/a", "http://x/b"])], validate)
    assert r["run_status"] == "complete" and dl.state.get("f1")["url"] == "http://x/b"
    assert 40 <= clock.t - t0 < 100                                          # two bounded waits of 20 s, never 600


def test_client_error_is_permanent_and_server_error_retryable(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({"http://x/1": [Resp(404)], "http://x/2": [Resp(500), Resp(500), Resp(500), Resp(500), Resp(500)]})
    dl = mk(tmp_path, s, clock, max_attempts=2, stall_seconds=10 ** 6)
    dl.run([item(tmp_path, 1), item(tmp_path, 2)], validate)
    assert dl.state.get("f1")["status"] == "permanent_failure" and dl.state.get("f1")["error"].startswith("http error")
    assert dl.state.get("f2")["status"] in ("retryable_failure", "permanent_failure") and dl.stats["server_error"] >= 2


def test_stall_detection_distinguishes_dead_network_from_slow_server(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    # dead network: every request raises; the clock advances past stall_seconds -> run exits "stalled", item retryable
    class Dead:
        calls = 0

        def get(self, url, headers=None, timeout=None):
            self.calls += 1; clock.t += 400; raise TimeoutError()
    dl = mk(tmp_path, Dead(), clock, max_attempts=10, stall_seconds=900, max_retry_after=1)
    r = dl.run([item(tmp_path, 1), item(tmp_path, 2)], validate)
    assert r["run_status"] == "stalled" and dl.state.get("f1")["status"] == "retryable_failure" and "http://x/2" not in []
    # slow but alive: 429s keep arriving; that is rate limiting, not a stall
    clock2 = Clock(); s = FakeSession({"http://x/1": [Resp(429, headers={"Retry-After": "5"})] * 3 + [Resp(200, GOOD)]})
    dl2 = mk(tmp_path / "b", s, clock2, max_attempts=6, stall_seconds=1, max_retry_after=5); (tmp_path / "b" / "img").mkdir(parents=True)
    r2 = dl2.run([{"id": "f1", "urls": ["http://x/1"], "dst": tmp_path / "b" / "img" / "f1.jpg"}], validate)
    assert r2["run_status"] == "complete"


def test_max_runtime_pauses_and_logs_heartbeat_without_completions(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({f"http://x/{i}": [Resp(200, GOOD)] for i in range(5)})
    dl = mk(tmp_path, s, clock, max_seconds=100, spacing=40.0, heartbeat_seconds=30)
    r = dl.run([item(tmp_path, i) for i in range(5)], validate)
    assert r["run_status"] == "paused" and 0 < r["done_this_run"] < 5 and r["remaining"] > 0
    log = (tmp_path / "dl.log").read_text(); assert "heartbeat" in log
    assert json.loads((tmp_path / "download_status.json").read_text())["run_status"] == "paused"
    # a slow run where nothing completes still writes heartbeats
    clock3 = Clock(); s3 = FakeSession({"http://x/1": [Resp(429, headers={"Retry-After": "50"})] * 8 + [Resp(200, GOOD)]})
    d3 = tmp_path / "c"; (d3 / "img").mkdir(parents=True)
    dl3 = mk(d3, s3, clock3, max_attempts=10, max_retry_after=50, heartbeat_seconds=30, stall_seconds=10 ** 6)
    dl3.run([{"id": "f1", "urls": ["http://x/1"], "dst": d3 / "img" / "f1.jpg"}], validate)
    assert (d3 / "dl.log").read_text().count("heartbeat") >= 3


def test_atomic_write_leaves_no_part_file_and_records_sha(tmp_path):
    clock = Clock(); (tmp_path / "img").mkdir()
    s = FakeSession({"http://x/1": [Resp(200, GOOD)]})
    dl = mk(tmp_path, s, clock); dl.run([item(tmp_path, 1)], validate)
    f = tmp_path / "img" / "f1.jpg"
    assert f.read_bytes() == GOOD and dl.state.get("f1")["sha256"] == sha256_file(f) and not list((tmp_path / "img").glob("*.part"))
    f.write_bytes(b"IMG:tampered")                                           # stored file changed: re-fetched on the next run
    s2 = FakeSession({"http://x/1": [Resp(200, GOOD)]}); mk(tmp_path, s2, clock).run([item(tmp_path, 1)], validate)
    assert s2.calls == ["http://x/1"] and f.read_bytes() == GOOD


# ------------------------------------------------------------------------------------------------ dataset state machine
@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(DP, "STATE_PATH", tmp_path / "dataset_state.json")
    man = tmp_path / "m.jsonl"; img = tmp_path / "a.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(img)
    man.write_text(json.dumps({"image_id": "a", "path": str(img), "sha256": sha256_file(img)}) + "\n")
    monkeypatch.setitem(DP.DATASETS, "ds_ok", {"manifest": man, "min_rows": 1, "sample": 5})
    monkeypatch.setitem(DP.DATASETS, "ds_missing", {"manifest": tmp_path / "none.jsonl", "min_rows": 1})
    monkeypatch.setitem(DP.DATASETS, "ds_dl", {"manifest": tmp_path / "dl.jsonl", "min_rows": 1,
                                                "acquire": [sys.executable, "-c", "import sys; sys.exit(3)"], "summary": tmp_path / "summary.json", "max_attempts": 2})
    return tmp_path


def test_dataset_validation_and_readiness(ds):
    r = DP.validate("ds_ok")
    assert r["state"] == "ready" and r["rows"] == 1 and r["manifest_sha256"]
    r = DP.validate("ds_missing")
    assert r["state"] == "retryable_failure" and "manifest" in r["last_error"]
    assert DP.is_ready("ds_ok") and not DP.is_ready("ds_missing")
    st = json.loads(DP.STATE_PATH.read_text()); assert st["ds_ok"]["history"][-1]["state"] == "ready"


def test_corrupt_manifest_file_quarantines_dataset(ds):
    man = DP.DATASETS["ds_ok"]["manifest"]; row = json.loads(man.read_text()); Path(row["path"]).write_bytes(b"junk");
    r = DP.validate("ds_ok")
    assert r["state"] == "quarantined" and "decode" in r["last_error"]


def test_acquire_exit_3_is_retryable_then_permanent_and_independent_of_other_datasets(ds):
    (ds / "summary.json").write_text(json.dumps({"complete": False}))
    r = DP.acquire("ds_dl")
    assert r["state"] == "retryable_failure" and r["attempts"] == 1
    r = DP.acquire("ds_dl")
    assert r["state"] == "permanent_failure" and r["attempts"] == 2
    assert DP.validate("ds_ok")["state"] == "ready"                       # the failed dataset does not block another
    assert DP.acquire("ds_dl")["state"] == "permanent_failure"            # not retried once permanent
    DP.reset("ds_dl"); assert DP.status()["ds_dl"]["state"] == "pending" and DP.status()["ds_dl"]["attempts"] == 0
