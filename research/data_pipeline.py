"""Data pipeline: dataset readiness as a persistent state machine, independent of training and evaluation.

  python -m research.data_pipeline status
  python -m research.data_pipeline acquire commons_film      # runs the registered (resumable) acquisition once
  python -m research.data_pipeline validate commons_film [--allow-partial]
  python -m research.data_pipeline reset commons_film

States: pending -> downloading -> validated -> ready; failures: retryable_failure, permanent_failure, quarantined.
A dataset is `ready` when its manifest validates (files present, a sample decodes, recorded sha256 matches) and,
for acquired datasets, the acquisition reports complete (or the owner validates with --allow-partial, which records
the manifest's sha256 so evaluations name the snapshot they used). One dataset's failure never changes another's state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from research import experiment_registry as R

ROOT = R.ROOT
STATE_PATH = ROOT / "data" / "manifests" / "dataset_state.json"
STATES = ("pending", "downloading", "validated", "ready", "retryable_failure", "permanent_failure", "quarantined")
PY = sys.executable

DATASETS: dict[str, dict] = {
    # evaluation-only real film (Wikimedia Commons, uploader-asserted stock labels); acquisition is resumable
    "commons_film": {"manifest": ROOT / "data" / "manifests" / "commons_film_manifest.jsonl", "min_rows": 1000, "sample": 40,
                     "acquire": [PY, "scripts/ingest_commons_film.py", "--download", "--max-minutes", "230", "--stall-minutes", "20", "--spacing", "2.0"],
                     "summary": ROOT / "data" / "raw" / "commons_film" / "summary.json", "max_attempts": 6, "training_allowed": False},
    # evaluation-only real film from CommonCatalog CC-BY film scans with uploader stock tags (class B); official Hub reads
    "commoncatalog_film": {"manifest": ROOT / "data" / "manifests" / "commoncatalog_film_manifest.jsonl", "min_rows": 300, "sample": 40,
                           "acquire": [PY, "scripts/ingest_commoncatalog_film.py", "--download", "--max-minutes", "110"],
                           "summary": ROOT / "data" / "raw" / "commoncatalog_film" / "summary.json", "max_attempts": 6, "training_allowed": False},
    # evaluation-only: ARTeFACT damaged/restored 35mm scans with documented emulsions (class A, CC BY 4.0, figshare)
    "artefact_film": {"manifest": ROOT / "data" / "manifests" / "artefact_film_manifest.jsonl", "min_rows": 40, "sample": 20,
                      "acquire": [PY, "scripts/ingest_artefact_film.py", "--download", "--max-minutes", "60"],
                      "summary": ROOT / "data" / "raw" / "artefact_film" / "summary.json", "max_attempts": 4, "training_allowed": False},
    # curated photography (museum, CC0): Art Institute of Chicago public-domain photographs with process records
    "aic_photographs": {"manifest": ROOT / "data" / "manifests" / "aic_photographs_manifest.jsonl", "min_rows": 200, "sample": 20,
                        "acquire": [PY, "scripts/ingest_aic_photographs.py", "--download", "--max-minutes", "90"],
                        "summary": ROOT / "data" / "raw" / "aic_photographs" / "summary.json", "max_attempts": 4, "training_allowed": True},
    # existing datasets: validated from their manifests (no acquisition command: already on disk)
    "commoncatalog_ccby": {"manifest": ROOT / "data" / "manifests" / "manifest.jsonl", "filter": {"dataset_id": "commoncatalog_ccby"}, "min_rows": 1000, "sample": 40},
    "pd12m_pilot": {"manifest": ROOT / "data" / "manifests" / "manifest.jsonl", "filter": {"dataset_id": "pd12m_pilot"}, "min_rows": 500, "sample": 20},
    "blueneg": {"manifest": ROOT / "data" / "manifests" / "blueneg_manifest.jsonl", "min_rows": 400, "sample": 40, "training_allowed": False},
    "film_synth_v2": {"manifest": ROOT / "data" / "synthetic" / "film_synth_v2" / "annotations.jsonl", "min_rows": 500, "sample": 40, "path_key": "path"},
    "sdxl_i2i_v1": {"manifest": ROOT / "data" / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl", "min_rows": 100, "sample": 20, "path_key": "path"},
    "zimage_i2i_v1": {"manifest": ROOT / "data" / "synthetic" / "zimage_i2i_v1" / "annotations.jsonl", "min_rows": 100, "sample": 20, "path_key": "path"},
    "bench_features_v1": {"manifest": ROOT / "data" / "embeddings" / "bench_features_v1.json", "kind": "feature_cache", "npz": ROOT / "data" / "embeddings" / "bench_features_v1.npz"},
}


# ------------------------------------------------------------------------------------------------ state file
def _load() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def _save(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.part"); tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, STATE_PATH)


def _set(ds: str, state: str, **fields) -> dict:
    assert state in STATES, state
    st = _load(); cur = st.get(ds, {"dataset": ds, "attempts": 0, "history": []})
    cur.update(fields); cur["state"] = state; cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cur["history"] = (cur.get("history") or [])[-19:] + [{"state": state, "ts": cur["updated"], "note": fields.get("last_error") or fields.get("note")}]
    st[ds] = cur; _save(st)
    return cur


def status(ds: str | None = None) -> dict:
    st = _load()
    out = {d: st.get(d, {"dataset": d, "state": "pending", "attempts": 0}) for d in DATASETS}
    return out[ds] if ds else out


def is_ready(ds: str) -> bool:
    return status(ds).get("state") == "ready"


def reset(ds: str) -> dict:
    return _set(ds, "pending", attempts=0, last_error=None, note="reset")


# ------------------------------------------------------------------------------------------------ validation
def _manifest_rows(spec: dict) -> list[dict]:
    p = Path(spec["manifest"])
    if spec.get("kind") == "feature_cache":
        meta = json.loads(p.read_text()); return meta.get("rows", [])
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for k, v in (spec.get("filter") or {}).items():
        rows = [r for r in rows if r.get(k) == v]
    return rows


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(ds: str, allow_partial: bool = False, sample: int | None = None) -> dict:
    """Manifest present and large enough, files exist, a deterministic sample decodes and matches its sha256."""
    spec = DATASETS[ds]; man = Path(spec["manifest"])
    if not man.exists():
        return _set(ds, "retryable_failure", last_error=f"manifest missing: {man}")
    try:
        rows = _manifest_rows(spec)
    except (ValueError, OSError) as e:
        return _set(ds, "quarantined", last_error=f"manifest unreadable: {e}")
    if spec.get("kind") == "feature_cache":
        npz = Path(spec["npz"])
        if not npz.exists():
            return _set(ds, "retryable_failure", last_error=f"npz missing: {npz}")
        return _set(ds, "ready", rows=len(rows), manifest_sha256=_sha(man), last_error=None)
    if len(rows) < spec.get("min_rows", 1):
        return _set(ds, "retryable_failure", last_error=f"{len(rows)} rows < min_rows {spec.get('min_rows', 1)}", rows=len(rows))
    key = spec.get("path_key", "path"); missing = [r for r in rows if not Path(r.get(key, "")).exists()]
    if missing:
        return _set(ds, "retryable_failure", last_error=f"{len(missing)} manifest files missing (e.g. {missing[0].get(key)})", rows=len(rows))
    n = sample or spec.get("sample", 20); step = max(1, len(rows) // n)
    from PIL import Image
    for r in rows[::step][:n]:
        p = Path(r[key])
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            return _set(ds, "quarantined", last_error=f"decode failed for {p.name}: {e}", rows=len(rows))
        if r.get("sha256") and r["sha256"] != _sha(p):
            return _set(ds, "quarantined", last_error=f"sha256 mismatch for {p.name}", rows=len(rows))
    complete = True
    if spec.get("summary"):
        sp = Path(spec["summary"])
        complete = bool(json.loads(sp.read_text()).get("complete")) if sp.exists() else False
    fields = dict(rows=len(rows), manifest_sha256=_sha(man), last_error=None, partial=not complete)
    if complete or allow_partial:
        return _set(ds, "ready", **fields)
    return _set(ds, "validated", note="acquisition incomplete; validate --allow-partial to evaluate a snapshot", **fields)


# ------------------------------------------------------------------------------------------------ acquisition
def acquire(ds: str, log_dir: Path | None = None) -> dict:
    """Run the dataset's acquisition command once (it is resumable itself). Exit 0 -> validate; exit 3 (paused/stalled/
    rate-limited) -> retryable_failure (counted); other -> retryable until max_attempts, then permanent_failure."""
    spec = DATASETS[ds]; cur = status(ds)
    if "acquire" not in spec:
        return validate(ds)
    if cur.get("state") in ("permanent_failure", "quarantined", "ready"):
        return cur
    attempts = int(cur.get("attempts", 0)) + 1
    _set(ds, "downloading", attempts=attempts, pid=os.getpid(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    log_dir = Path(log_dir or ROOT / "reports" / "data_pipeline"); log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{ds}.log", "a") as fh:
        fh.write(f"\n=== acquire attempt {attempts} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"); fh.flush()
        try:
            rc = subprocess.run(spec["acquire"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, timeout=spec.get("timeout", 5 * 3600)).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    if rc == 0:
        return validate(ds)
    err = {3: "acquisition paused/stalled/rate-limited (resumable)", 124: "acquisition timed out"}.get(rc, f"acquisition exit code {rc}")
    if attempts >= spec.get("max_attempts", 5):
        return _set(ds, "permanent_failure", last_error=f"{err}; attempts exhausted ({attempts})", pid=None)
    return _set(ds, "retryable_failure", last_error=err, pid=None)


def refresh() -> dict:
    """Re-check datasets whose state can change without us: downloading with a dead process -> retryable."""
    from research.resources import pid_alive
    st = _load()
    for ds, cur in st.items():
        if cur.get("state") == "downloading" and not pid_alive(cur.get("pid")):
            _set(ds, "retryable_failure", last_error="acquisition process died", pid=None)
    return status()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "acquire", "validate", "reset", "validate-all"])
    ap.add_argument("dataset", nargs="?"); ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        for d, s in refresh().items():
            print(f"{d:22s} {s.get('state', 'pending'):18s} rows={s.get('rows', '-')} attempts={s.get('attempts', 0)} {s.get('last_error') or ''}")
    elif a.cmd == "validate-all":
        for d in DATASETS:
            s = validate(d, a.allow_partial); print(f"{d:22s} {s['state']:18s} {s.get('last_error') or ''}")
    else:
        fn = {"acquire": acquire, "validate": lambda d: validate(d, a.allow_partial), "reset": reset}[a.cmd]
        r = fn(a.dataset); print(json.dumps(r, indent=1))
        # exit codes for the orchestrator: 0 ready/pending, 3 retryable (resumable), 2 permanent/quarantined
        return {"ready": 0, "pending": 0, "validated": 3, "retryable_failure": 3, "downloading": 3}.get(r["state"], 2)


if __name__ == "__main__":
    raise SystemExit(main() or 0)

