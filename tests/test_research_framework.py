"""Research orchestration: registry round trip, runner ledger and budget, evaluator decision rules, planner choice."""
import json
import sys
from pathlib import Path

import pytest

from research import evaluator
from research import experiment_registry as R
from research import experiment_runner as RR
from research import planner


@pytest.fixture
def reg(tmp_path, monkeypatch):
    p = tmp_path / "registry.jsonl"
    monkeypatch.setattr(R, "REGISTRY", p)
    monkeypatch.setattr(RR, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(RR, "LOGS", tmp_path / "logs")
    return p


def test_registry_round_trip_and_updates(reg):
    R.register("e1", "h", ["true"], path=reg, metrics_of_interest={"m": {"path": "a.b", "kind": "real"}})
    with pytest.raises(ValueError):
        R.register("e1", "h", ["true"], path=reg)
    R.update("e1", path=reg, status="done", metrics={"m": 0.5})
    e = R.get("e1", path=reg)
    assert e["status"] == "done" and e["metrics"] == {"m": 0.5} and e["hypothesis"] == "h"
    assert R.dig({"a": {"b": [1, {"c": 2}]}}, "a.b.1.c") == 2


def test_runner_records_resources_and_extracts_metrics(reg, tmp_path):
    out = tmp_path / "res.json"
    cmd = [sys.executable, "-c", f"import json; json.dump({{'x': {{'auroc': 0.7, 'ci95': [0.6, 0.8]}}}}, open(r'{out}', 'w'))"]
    R.register("e2", "h", cmd, metrics_of_interest={"x": {"path": "x", "kind": "synthetic"}})
    r = RR.run("e2", str(out), uses_gpu=False, watched_paths=[str(tmp_path)])
    assert r["status"] == "done" and r["metrics"]["x"]["auroc"] == 0.7 and r["wall_seconds"] >= 0
    led = json.loads((tmp_path / "ledger.json").read_text())
    assert led["runs"][0]["id"] == "e2" and led["gpu_seconds"] == 0.0


def test_runner_blocks_when_budget_exhausted(reg, tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps({"gpu_seconds": RR.LIMITS["gpu_seconds"] - 10, "disk_bytes": 0, "runs": []}))
    R.register("e3", "h", ["true"])
    r = RR.run("e3", str(tmp_path / "none.json"), uses_gpu=True, expected_gpu_seconds=60)
    assert r["status"] == "blocked" and "GPU budget" in r["reason"]


def test_evaluator_rejects_synthetic_up_real_down(reg):
    moi = {"syn": {"path": "s", "kind": "synthetic"}, "real": {"path": "r", "kind": "real"}}
    R.register("base", "b", ["true"], metrics_of_interest=moi); R.update("base", status="done", metrics={"syn": [0.5, [0.45, 0.55]], "real": [0.8, [0.75, 0.85]]})
    R.register("new", "n", ["true"], baseline_id="base", metrics_of_interest=moi); R.update("new", status="done", metrics={"syn": [0.7, [0.65, 0.75]], "real": [0.65, [0.6, 0.7]]})
    c = evaluator.compare("new")
    assert c["per_metric"]["syn"]["verdict"] == "improved" and c["per_metric"]["real"]["verdict"] == "regressed"
    assert c["decision"].startswith("reject")
    R.update("new", metrics={"syn": [0.7, [0.65, 0.75]], "real": [0.81, [0.76, 0.86]]})
    assert evaluator.compare("new")["decision"].startswith("neutral")


def test_planner_prefers_value_per_cost_within_budget(reg, tmp_path):
    bl = tmp_path / "backlog.yaml"
    bl.write_text("experiments:\n  - {id: a, hypothesis: x, value: 5, gpu_minutes: 1000}\n  - {id: b, hypothesis: y, value: 2, gpu_minutes: 2}\n"
                  "  - {id: c, hypothesis: z, value: 4, gpu_minutes: 10, blocked_by: a}\n")
    (tmp_path / "ledger.json").write_text(json.dumps({"gpu_seconds": 0, "disk_bytes": 0, "runs": []}))
    assert planner.select_next(bl)["id"] == "b"                   # a exceeds the budget, c is blocked
    md = planner.render_markdown(bl)
    assert "| a |" in md and "pending" in md


def test_same_stock_metric_reports_roll_bootstrap_ci():
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import synth_experiment_v3 as V3
    rng = np.random.default_rng(0)
    stock = np.array(["a"] * 20 + ["b"] * 20); roll = np.array([f"r{i // 4}" for i in range(40)])
    F = np.vstack([rng.normal(size=(20, 8)) + 3, rng.normal(size=(20, 8)) - 3])      # perfectly separable stocks
    r = V3.same_stock(F, stock, roll, n_boot=50)
    assert r["precision@5"] == 1.0 and r["ci95"] == [1.0, 1.0] and r["n_rolls"] == 10 and 0.4 < r["chance"] < 0.6
    r2 = V3.same_stock(rng.normal(size=(40, 8)), stock, roll, n_boot=50)
    assert r2["ci95"][0] <= r2["precision@5"] <= r2["ci95"][1]
