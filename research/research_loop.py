"""Research pipeline: one autonomous experiment cycle on the data that is already available.

  python -m research.research_loop --once [--dry-run] [--id E26_layer_statistic_ablation]

Cycle: (1) read previous results (registry); (2) pick the highest value-per-cost pending backlog entry that carries a
runnable spec (`command`, `result_json`, `metrics_of_interest`; `baseline_id` optional) and whose dataset dependencies
are ready; (3) estimate cost (gpu_minutes/gpu_mb from the spec) and check GPU free memory and the session budget;
(4) register and run it through research.experiment_runner (metered); (5) compare with the baseline under
research.evaluator's rules, apply the entry's pre-registered `falsify` rule when present; (6) write the decision
(supported / rejected / inconclusive) back to the backlog and the registry and append the result paragraph to
docs/EXPERIMENT_LOG.md; (7) record reports/research_loop/last.json so the orchestrator knows whether another cycle
is available. Entries without a runnable spec are reported, not run (they need a script first).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from research import data_pipeline as DP
from research import evaluator
from research import experiment_registry as R
from research import experiment_runner as RR
from research import planner
from research import result_analyzer as A
from research.resources import estimate, gpu_can_launch

OUT = R.ROOT / "reports" / "research_loop"
LOG_MD = R.ROOT / "docs" / "EXPERIMENT_LOG.md"
MARKER = "## Decisions taken from these runs"


def runnable_entries(path: Path = planner.BACKLOG) -> tuple[list[dict], list[dict]]:
    """Pending backlog entries with a runnable spec and ready datasets, and the ones that lack a spec."""
    ok, needs = [], []
    for e in planner.pending(path):
        if not (e.get("command") and e.get("result_json") and e.get("metrics_of_interest")):
            needs.append(e); continue
        deps = [d for d in e.get("datasets", []) if d in DP.DATASETS and DP.status(d).get("state") != "ready"]
        if deps:
            needs.append({**e, "waiting_on": deps}); continue
        ok.append(e)
    return ok, needs


def select(path: Path = planner.BACKLOG, exp_id: str | None = None) -> tuple[dict | None, list[dict]]:
    ok, needs = runnable_entries(path)
    if exp_id:
        return next((e for e in ok if e["id"] == exp_id), None), needs
    rem = RR.remaining()
    ok = [e for e in ok if e.get("gpu_minutes", 0) * 60 <= rem["gpu_seconds"] and e.get("disk_gb", 0) * 1024 ** 3 <= rem["disk_bytes"] and not e.get("blocked_by")]
    return (max(ok, key=lambda e: e.get("value", 1) / max(e.get("gpu_minutes", 0) + e.get("cpu_minutes", 1), 1)) if ok else None), needs


def apply_falsify(rule: dict | None, metrics: dict) -> tuple[bool | None, str]:
    """rule: {"metric": name, "below": x} or {"metric": name, "above": x}: falsified when the metric's value is below/above."""
    if not rule or rule.get("metric") not in metrics:
        return None, "no falsification rule or metric missing"
    v, _ = evaluator._val(metrics[rule["metric"]])
    if v is None:
        return None, "metric not scalar"
    if "below" in rule:
        return v < rule["below"], f"{rule['metric']}={v:.4f} {'<' if v < rule['below'] else '>='} {rule['below']}"
    if "above" in rule:
        return v > rule["above"], f"{rule['metric']}={v:.4f} {'>' if v > rule['above'] else '<='} {rule['above']}"
    return None, "rule has neither below nor above"


def set_backlog_status(exp_id: str, status: str, result: str, path: Path = planner.BACKLOG) -> None:
    d = yaml.safe_load(Path(path).read_text()) or {}
    for e in d.get("experiments", []):
        if e["id"] == exp_id:
            e["status"], e["result"] = status, result
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=120))


def append_log(paragraph: str) -> bool:
    if not LOG_MD.exists():
        return False
    s = LOG_MD.read_text(); i = s.find(MARKER)
    s = (s[:i] + paragraph + "\n\n" + s[i:]) if i >= 0 else (s.rstrip() + "\n\n" + paragraph + "\n")
    LOG_MD.write_text(s); return True


def cycle(exp_id: str | None = None, dry_run: bool = False, path: Path = planner.BACKLOG) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    e, needs = select(path, exp_id)
    rep = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "selected": e["id"] if e else None, "needs_spec_or_data": [{"id": n["id"], "waiting_on": n.get("waiting_on")} for n in needs],
           "budget_remaining": RR.remaining()}
    if e is None:
        rep["next_available"] = False; (OUT / "last.json").write_text(json.dumps(rep, indent=1)); return rep
    est = estimate(e); ok, why = gpu_can_launch(est["gpu_mb"]); rep.update(estimate=est, gpu_check=why)
    if not ok:
        rep["next_available"] = True; rep["deferred"] = why; (OUT / "last.json").write_text(json.dumps(rep, indent=1)); return rep
    if dry_run:
        rep["next_available"] = True; return rep
    if R.get(e["id"]) is None:
        R.register(e["id"], e["hypothesis"], e["command"], baseline_id=e.get("baseline_id"), metrics_of_interest=e["metrics_of_interest"],
                   labels=e.get("labels", ""), falsify=e.get("falsify"), falsified_if=e.get("falsified_if"))
    res = RR.run(e["id"], e["result_json"], uses_gpu=est["gpu_minutes"] > 0, expected_gpu_seconds=est["gpu_minutes"] * 60,
                 watched_paths=e.get("watched_paths", []), timeout=int(est["wall_minutes_max"] * 60))
    rep["run"] = {k: res.get(k) for k in ("status", "returncode", "wall_seconds", "peak_gpu_mb_total", "metrics_missing")}
    if res.get("status") != "done":
        set_backlog_status(e["id"], "failed" if res.get("status") == "failed" else res.get("status", "failed"), res.get("reason") or f"run {res.get('status')}", path)
        rep["decision"] = res.get("status"); rep["next_available"] = True; (OUT / "last.json").write_text(json.dumps(rep, indent=1, default=str)); return rep
    comp = evaluator.compare(e["id"]) if e.get("baseline_id") else {"decision": "no baseline registered", "per_metric": {}}
    fals, fals_why = apply_falsify(e.get("falsify"), res.get("metrics", {}))
    if fals is True:
        status, why = "rejected", f"falsified: {fals_why}"
    elif comp["decision"].startswith("reject"):
        status, why = "rejected", comp["decision"]
    elif comp["decision"].startswith("accept") or fals is False and e.get("falsify_sufficient"):
        status, why = "supported", comp["decision"] if comp["decision"].startswith("accept") else f"not falsified: {fals_why}"
    else:
        status, why = "inconclusive", f"{comp['decision']}; {fals_why}"
    R.update(e["id"], decision=status, decision_reason=why)
    set_backlog_status(e["id"], status, why, path)
    rep.update(decision=status, reason=why, comparison=comp.get("per_metric"), falsify=fals_why)
    learned = e.get("learned_template", "see metrics") if False else f"decision {status}: {why}"
    rep["log_appended"] = append_log(A.log_paragraph(e["id"], learned=learned, next_step=e.get("next_step", "re-plan from the backlog")))
    ok2, _ = select(path); rep["next_available"] = ok2 is not None
    (OUT / "last.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--id", default=None)
    a = ap.parse_args(argv)
    rep = cycle(a.id, a.dry_run)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep.get("decision") not in ("failed", "blocked") else 1       # a failed run is a failed task (retryable)


if __name__ == "__main__":
    raise SystemExit(main())
