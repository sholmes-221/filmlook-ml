"""Evaluation pipeline: independent real-film validation of the eligible candidates when a dataset is ready.

  python -m research.eval_pipeline run --dataset commons_film --split analysis      # selection half (repeatable)
  python -m research.eval_pipeline run --dataset commons_film --split test --final  # held-out half: ONCE per dataset
  python -m research.eval_pipeline status

Steps: (1) the dataset must be `ready` in the data pipeline (snapshot = manifest sha256); (2) features are computed
once per snapshot (one GPU pass; cached and stamped with the snapshot, so a resumed download invalidates them);
(3) scripts/eval_commons_film.py runs on the requested half with every eligible student checkpoint added
(checkpoints/student_style_*/model.onnx, excluding checkpoints that already have a final-split record);
(4) each candidate is compared with the established baseline (hybrid_supcon_family) under the pre-registered rules
below; (5) the report and decisions are appended to reports/eval_pipeline/history.jsonl.

Pre-registered rules (docs/BENCHMARK.md, "Real-film evaluation"): primary metric same_stock precision@5 with the
query's uploader excluded, bootstrap CI over uploaders. A candidate is `improved` on real film when its delta over the
baseline is >= 0.02 and the CIs do not overlap; `regressed` symmetrically; otherwise `inconclusive`. Leakage guards:
uploader_share_top5 (minus its prior) may not exceed the baseline's by more than 0.02 and the scanner probe accuracy
may not exceed the baseline's by more than 0.05; a candidate failing a guard is `reject: leakage` whatever its primary
metric. The test half is evaluated once per dataset snapshot and never used for selection; a second final run needs
--force-final and is recorded as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research import data_pipeline as DP
from research import experiment_registry as R

ROOT = R.ROOT
OUT = ROOT / "reports" / "eval_pipeline"
HISTORY, FINAL = OUT / "history.jsonl", OUT / "final_runs.json"
BASELINE = "hybrid_supcon_family"
RULES = {"primary": "same_stock", "min_delta": 0.02, "uploader_share_margin": 0.02, "scanner_probe_margin": 0.05}
PY = sys.executable
DATASET_SCRIPTS = {"commons_film": {"script": "scripts/eval_commons_film.py", "features": ROOT / "data" / "embeddings" / "commons_film_features_v1",
                                    "report": lambda split: ROOT / "reports" / f"commons_film_{split}.json", "splits": ("analysis", "test")}}


def eligible_students(exclude_default: bool = True) -> list[str]:
    tags = []
    for d in sorted((ROOT / "checkpoints").glob("student_style_*")):
        if (d / "model.onnx").exists() and (d / "meta.json").exists() and "style_embedding" in json.loads((d / "meta.json").read_text()).get("outputs", []):
            tag = d.name[len("student_"):]
            if exclude_default and tag == "style_fs_v2":
                continue                                        # already the benchmark's student_fs_v2_64 candidate
            tags.append(tag)
    return tags


def _features_stamp(feat: Path) -> str | None:
    j = feat.with_suffix(".json")
    return json.loads(j.read_text()).get("manifest_sha256") if j.exists() else None


def ensure_features(dataset: str, snapshot: str, tags: list[str], log) -> dict:
    spec = DATASET_SCRIPTS[dataset]; feat = spec["features"]; j = feat.with_suffix(".json")
    have = json.loads(j.read_text()) if j.exists() else {}
    if have.get("manifest_sha256") == snapshot and set(tags) <= set(have.get("student_tags", [])):
        return {"computed": False, "seconds": 0}
    t0 = time.time()
    env = {**os.environ, "STUDENT_TAGS": ",".join(tags)}
    rc = subprocess.run([PY, spec["script"], "--features"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"feature pass failed (exit {rc}); see log")
    meta = json.loads(j.read_text()); meta["manifest_sha256"] = snapshot; meta["student_tags"] = tags; j.write_text(json.dumps(meta))
    return {"computed": True, "seconds": round(time.time() - t0, 1)}


def _ci(m):
    return m.get("ci95") if isinstance(m, dict) else None


def decide(cand: dict, base: dict) -> dict:
    """Apply RULES to one candidate's metrics vs the baseline's (both from scripts/eval_commons_film.py)."""
    p = RULES["primary"]; v, bv = cand[p].get("precision@5"), base[p].get("precision@5")
    if v is None or bv is None:
        return {"verdict": "inconclusive", "reason": "primary metric missing"}
    ci, bci = _ci(cand[p]), _ci(base[p]); d = v - bv
    overlap = bool(ci and bci and not (ci[0] > bci[1] or ci[1] < bci[0]))
    verdict = "improved" if d >= RULES["min_delta"] and not overlap else "regressed" if d <= -RULES["min_delta"] and not overlap else "inconclusive"
    us = cand["uploader_share_top5"]["share"] - cand["uploader_share_top5"]["prior"]; bus = base["uploader_share_top5"]["share"] - base["uploader_share_top5"]["prior"]
    guards = {"uploader_share_excess": round(us - bus, 4), "uploader_guard_ok": us - bus <= RULES["uploader_share_margin"]}
    sa, sb = (cand.get("scanner_probe") or {}).get("accuracy"), (base.get("scanner_probe") or {}).get("accuracy")
    guards["scanner_probe_delta"] = round(sa - sb, 4) if sa is not None and sb is not None else None
    guards["scanner_guard_ok"] = guards["scanner_probe_delta"] is None or guards["scanner_probe_delta"] <= RULES["scanner_probe_margin"]
    decision = ("reject: leakage" if not (guards["uploader_guard_ok"] and guards["scanner_guard_ok"]) else
                "reject: real regression" if verdict == "regressed" else "candidate: real improvement" if verdict == "improved" else "inconclusive")
    return {"verdict": verdict, "primary": p, "value": v, "ci95": ci, "baseline": bv, "baseline_ci95": bci, "delta": round(d, 4), **guards, "decision": decision}


def run(dataset: str, split: str = "analysis", final: bool = False, force_final: bool = False, tags: list[str] | None = None) -> dict:
    spec = DATASET_SCRIPTS[dataset]; OUT.mkdir(parents=True, exist_ok=True)
    st = DP.status(dataset)
    if st.get("state") != "ready":
        raise SystemExit(f"dataset {dataset} is {st.get('state')}, not ready: nothing evaluated")
    snapshot = st["manifest_sha256"]; short = snapshot[:8]
    if split == "test":
        done = json.loads(FINAL.read_text()) if FINAL.exists() else {}
        if not final:
            raise SystemExit("the test half needs --final (held-out: evaluated once, never for selection)")
        if dataset in done and not force_final:
            raise SystemExit(f"final run already recorded for {dataset} ({done[dataset]['ts']}, snapshot {done[dataset]['snapshot'][:8]}); use --force-final to record another")
    tags = eligible_students() if tags is None else tags
    log = open(OUT / f"{dataset}_{split}_{short}.log", "a"); log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} split={split} snapshot={short} tags={tags}\n"); log.flush()
    feats = ensure_features(dataset, snapshot, tags, log)
    env = {**os.environ, "STUDENT_TAGS": ",".join(tags)}
    rc = subprocess.run([PY, spec["script"], "--split", split], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"evaluation failed (exit {rc}); see {log.name}")
    rep = json.loads(spec["report"](split).read_text()); base = rep["candidates"][BASELINE]
    decisions = {name: decide(c, base) for name, c in rep["candidates"].items() if name != BASELINE}
    out = {"dataset": dataset, "split": split, "snapshot": snapshot, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "features": feats, "baseline": BASELINE,
           "rules": RULES, "n_film": rep["n_film"], "n_uploaders": rep["n_uploaders_film"], "student_tags": tags, "decisions": decisions, "final": split == "test"}
    dst = OUT / f"{dataset}_{split}_{short}.json"; shutil.copy(spec["report"](split), dst); out["report"] = str(dst)
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    if split == "test":
        done = json.loads(FINAL.read_text()) if FINAL.exists() else {}
        done[dataset] = {"ts": out["ts"], "snapshot": snapshot, "forced": dataset in done, "report": str(dst)}; FINAL.write_text(json.dumps(done, indent=1))
    print(json.dumps({k: out[k] for k in ("dataset", "split", "n_film", "n_uploaders", "student_tags")}))
    print(f"{'candidate':28s} {'p@5':>6s} {'base':>6s} {'delta':>7s} {'upl+':>7s} decision")
    for name, d in decisions.items():
        print(f"{name:28s} {d.get('value', 0) or 0:6.3f} {d.get('baseline', 0) or 0:6.3f} {d.get('delta', 0) or 0:7.3f} {d.get('uploader_share_excess', 0) or 0:7.3f} {d['decision']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status"]); ap.add_argument("--dataset", default="commons_film"); ap.add_argument("--split", default="analysis")
    ap.add_argument("--final", action="store_true"); ap.add_argument("--force-final", action="store_true"); ap.add_argument("--tags", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print("eligible students:", eligible_students())
        if HISTORY.exists():
            for l in HISTORY.read_text().splitlines()[-10:]:
                r = json.loads(l); print(r["ts"], r["dataset"], r["split"], r["snapshot"][:8], {k: v["decision"] for k, v in r["decisions"].items()})
        print("final runs:", json.loads(FINAL.read_text()) if FINAL.exists() else {})
    else:
        run(a.dataset, a.split, a.final, a.force_final, [t for t in a.tags.split(",") if t] if a.tags is not None else None)


if __name__ == "__main__":
    main()
