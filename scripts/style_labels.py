#!/usr/bin/env python
"""Collect photographer-STYLE judgements from the Claude teacher (offline, gated, budget-capped).

  python scripts/style_labels.py --images 60                         # dry run: selection + cost estimate
  python scripts/style_labels.py --images 60 --no-dry-run --confirm-paid

Uses the SAME cumulative US$5 ledger as film labeling (data/labels/claude_usage.jsonl). Only originals from
registry-approved, consented datasets are uploaded. Output: data/labels/style_judgements.jsonl (source=claude).
Human judgements come from the review app (Style mode) into the same file with source=human.

COMPLIANCE (docs/AI_DATA_POLICY.md): requires --intended-use training|evaluation and passes the labeling
preflight first. Without a written training authorization, judgements are evaluation-only (provenance
training_permission=uncertain) and scripts/style_evaluate.py refuses to train learned methods on them.
"""
import argparse
import json
import random

from filmlook_ml import paths as P
from filmlook_ml.compliance.policy import LabelingLock
from filmlook_ml.compliance.preflight import run_labeling_preflight
from filmlook_ml.compliance.provenance import make_provenance
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.labeling.schema import StyleJudgement
from filmlook_ml.labeling.teachers import (BudgetExceeded, ClaudeTeacher, api_key_status, style_system_prompt,
                                           training_authorization)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=60)
    ap.add_argument("--no-dry-run", action="store_true")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intended-use", choices=["training", "evaluation"], default=None,
                    help="declared use of the judgements (required; checked by the compliance preflight)")
    a = ap.parse_args()
    lcfg = P.load_yaml("labeling.yaml")
    ccfg = lcfg["claude"]
    pre = run_labeling_preflight(root=P.ROOT, ccfg=ccfg, intended_use=a.intended_use or "undeclared",
                                 n_images=a.images, dry_run=not a.no_dry_run,
                                 prompts={"style_system_prompt": style_system_prompt()},
                                 report_path=P.REPORTS / "preflight_labeling.json")
    print(pre.summary())
    if not pre.ok:
        raise SystemExit("labeling preflight failed; Claude labeling must not proceed (docs/AI_DATA_POLICY.md)")
    consented = set(ccfg["consented_datasets"])
    man = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    pool = [r for r in man if r["edit_type"] == "original" and r["dataset_id"] in consented
            and r["dataset_id"] != "synthetic_fixture"]
    random.Random(a.seed).shuffle(pool)
    sel = pool[: a.images]
    t = ClaudeTeacher(ccfg, P.LABELS / "claude_cache.jsonl", lcfg["prompt_version"], dry_run=not a.no_dry_run,
                      confirm=a.confirm_paid, usage_log=P.LABELS / "claude_usage.jsonl")
    est = t.estimate(len(sel))
    print(json.dumps({"selected": len(sel), "estimate_usd": est["total_usd"], "api_key": api_key_status(P.ROOT),
                      "spent_so_far_usd": round(t.cost.spent_before, 4), "cap_usd": ccfg["max_budget_usd"],
                      "training_authorization": training_authorization(ccfg), "dry_run": t.dry_run}, indent=1))
    if not t.dry_run and api_key_status(P.ROOT) == "missing":
        raise SystemExit("ANTHROPIC_API_KEY not found in environment or .env")
    out = []
    lock = LabelingLock(P.LABELS / "claude_labeling.lock")
    lock.__enter__()
    try:
        for r in sel:
            res = t.style(P.ROOT / r["path"] if not r["path"].startswith("/") else r["path"], r["sha256"],
                          r["image_id"], consent_ok=True)
            if res is None:
                continue
            ids = [c.id for c in res.candidates if c.resemblance >= 0.5][:3]
            out.append(StyleJudgement(image_id=r["image_id"], sha256=r["sha256"], source="claude",
                                      reviewer=t.model_id, matches=ids, no_match=res.no_reliable_match or not ids,
                                      uncertainty=res.uncertainty,
                                      provenance=make_provenance("claude", r, model_id=t.model_id, claude_cfg=ccfg)))
    except BudgetExceeded as e:
        print("stopped:", e)
    finally:
        lock.__exit__(None, None, None)
    if out:
        with open(P.LABELS / "style_judgements.jsonl", "a") as fh:
            for j in out:
                fh.write(j.model_dump_json() + "\n")
    print(f"judgements written: {len(out)}; spent this run ${t.cost.spent_this_run:.4f}, total ${t.cost.spent:.4f}")


if __name__ == "__main__":
    main()
