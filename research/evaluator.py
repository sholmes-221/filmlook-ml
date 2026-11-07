"""Compare an experiment with its baseline and apply the decision rules (real-film regression => reject)."""
from __future__ import annotations

from research import experiment_registry as R


def _val(m):
    """Metric value; accepts a number or [value, [lo, hi]] / {'auroc':..,'ci95':[..]} shapes."""
    if isinstance(m, dict):
        for k in ("auroc", "value", "precision@5", "macro_f1", "accuracy"):
            if k in m:
                return float(m[k]), m.get("ci95")
    if isinstance(m, (list, tuple)) and m and isinstance(m[0], (int, float)):
        return float(m[0]), (m[1] if len(m) > 1 and isinstance(m[1], (list, tuple)) else None)
    if isinstance(m, (int, float)):
        return float(m), None
    return None, None                                   # nested record without a scalar: skipped by compare()


def compare(exp_id: str, baseline_id: str | None = None, min_delta: float = 0.02) -> dict:
    exp = R.get(exp_id)
    base = R.get(baseline_id or exp.get("baseline_id") or "") if (baseline_id or exp.get("baseline_id")) else None
    out = {"id": exp_id, "baseline": base["id"] if base else None, "per_metric": {}, "synthetic": [], "real": [], "proxy": []}
    for name, spec in exp.get("metrics_of_interest", {}).items():
        if name not in exp.get("metrics", {}):
            continue
        v, ci = _val(exp["metrics"][name])
        if v is None:
            continue
        entry = {"value": v, "ci95": ci, "kind": spec.get("kind", "proxy")}
        if base and name in base.get("metrics", {}):
            bv, bci = _val(base["metrics"][name])
            if bv is None:
                out["per_metric"][name] = entry; out[entry["kind"]].append((name, None)); continue
            d = (v - bv) * (1 if spec.get("higher_is_better", True) else -1)
            overlap = bool(ci and bci and not (ci[0] > bci[1] or ci[1] < bci[0]))
            entry.update({"baseline": bv, "delta": round(d, 4),
                          "verdict": "improved" if d >= min_delta and not overlap else "regressed" if d <= -min_delta and not overlap
                          else "neutral" if abs(d) < min_delta else ("improved?" if d > 0 else "regressed?")})
        out["per_metric"][name] = entry
        out[entry["kind"]].append((name, entry.get("verdict")))
    synth_up = any(v and v.startswith("improved") for _, v in out["synthetic"])
    real_down = any(v and v.startswith("regressed") for _, v in out["real"])
    real_up = any(v and v.startswith("improved") for _, v in out["real"])
    out["decision"] = ("reject: synthetic up, real down" if synth_up and real_down else
                       "reject: real regression" if real_down else
                       "accept: real improvement" if real_up else "neutral: no credible real-film change")
    return out
