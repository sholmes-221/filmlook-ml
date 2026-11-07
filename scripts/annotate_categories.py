#!/usr/bin/env python
"""Exploratory photography categorization with the Claude API (photographer style, film aesthetic, attributes).

  python scripts/annotate_categories.py --status                     # discovery, progress, distributions
  python scripts/annotate_categories.py --max-images 100             # dry run: preflight + cost estimate
  python scripts/annotate_categories.py --max-images 100 --no-dry-run --confirm-paid   # paid (see below)

A paid run needs ALL of: ANTHROPIC_API_KEY (environment or .env), the labeling preflight passing with
--intended-use exploratory (which requires claude.exploratory_use.approved / approved_by / recorded in
configs/labeling.yaml), the shared cumulative budget cap, and --no-dry-run --confirm-paid.

Unattended use: safe to re-run (cron, or a shell loop that YOU start). It resumes and skips annotated images,
writes atomically, never overwrites, honours rate limits, and exits non-zero on authentication, policy or
budget stops. Do not wrap it in an automatic restart loop: a stop needs a human decision.

Exit codes: 0 completed / per-run limit reached; 2 preflight or configuration blocked; 3 run stopped.
Output: data/annotations/claude_exploratory/ (git-ignored), schema categorization-2.0, annotation_interface
anthropic_api. Uploader tags and capture medium are stored as metadata and are never sent to the model.
"""
import argparse
import json
import sys

from filmlook_ml import paths as P
from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.qa import distributions
from filmlook_ml.annotation.workflow import (PROMPT_VERSION, AnnotationStore, Categorizer, discover,
                                             load_uploader_tags, run, system_prompt, validate_store)
from filmlook_ml.compliance.policy import ConcurrentLabelingJob, LabelingLock, claude_training_permission
from filmlook_ml.compliance.preflight import run_labeling_preflight
from filmlook_ml.datasets.preprocess import read_jsonl
from filmlook_ml.datasets.quarantine import load_quarantine
from filmlook_ml.datasets.registry import Registry
from filmlook_ml.labeling.teachers import ClaudeTeacher, api_key_status

ANN_ROOT = P.DATA / "annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=100, help="per-run bound (<= claude.max_images_per_run)")
    ap.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--no-dry-run", action="store_true")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--status", action="store_true", help="report only; no API calls")
    a = ap.parse_args()

    lcfg = P.load_yaml("labeling.yaml")
    ccfg = lcfg["claude"]
    registry = Registry.load()
    manifest = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    eligible, skipped = discover(manifest, registry, set(ccfg["consented_datasets"]),
                                  load_quarantine())
    store = AnnotationStore(ANN_ROOT)
    anns = store.load_all()
    status, basis = claude_training_permission(ccfg)
    report = {"images_in_manifest": len(manifest),
              "originals_discovered": sum(r.get("edit_type") == "original" for r in manifest),
              "eligible": len(eligible), "skipped_by_reason": dict(skipped),
              "already_categorized": len(anns), "validation": validate_store(store),
              "distributions": distributions(anns), "taxonomy_version": T.TAXONOMY_VERSION,
              "prompt_version": PROMPT_VERSION, "training_permission": {"status": status, "basis": basis},
              "api_key": api_key_status(P.ROOT)}
    if a.status:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        (P.REPORTS / "categorization_status.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        return 0

    dry = not a.no_dry_run
    pre = run_labeling_preflight(root=P.ROOT, ccfg=ccfg, intended_use="exploratory", n_images=a.max_images,
                                 dry_run=dry, prompts={"categorize_system_prompt": system_prompt()},
                                 registry=registry, report_path=P.REPORTS / "preflight_categorization.json")
    print(pre.summary())
    teacher = ClaudeTeacher(ccfg, P.LABELS / "claude_cache.jsonl", lcfg["prompt_version"], dry_run=dry,
                            confirm=a.confirm_paid, usage_log=P.LABELS / "claude_usage.jsonl")
    todo = sum(not store.has(r["image_id"]) for r in eligible)
    est = teacher.estimate(min(todo, a.max_images))
    print(json.dumps({"eligible": len(eligible), "todo": todo, "this_run": min(todo, a.max_images),
                      "estimate_usd": est["total_usd"], "per_image_usd": est["per_image_usd"],
                      "spent_so_far_usd": round(teacher.cost.spent_before, 4), "cap_usd": ccfg["max_budget_usd"],
                      "api_key": report["api_key"], "dry_run": dry}, indent=1))
    if not dry:
        if not pre.ok:
            print("BLOCKED: labeling preflight failed; no API calls made (docs/AI_DATA_POLICY.md)", file=sys.stderr)
            return 2
        if report["api_key"] == "missing":
            print("BLOCKED: ANTHROPIC_API_KEY not found in environment or .env", file=sys.stderr)
            return 2
    try:
        with LabelingLock(P.LABELS / "claude_labeling.lock"):
            res = run(eligible=eligible, store=store, categorizer=Categorizer(teacher), ccfg=ccfg,
                      max_images=a.max_images, min_interval_s=a.min_interval, root=P.ROOT,
                      features=json.loads((P.EMBEDDINGS / "features_ms1280_c512.json").read_text()),
                      usertags=load_uploader_tags(P.DATA / "candidates" / "commoncatalog_candidates.jsonl"))
    except ConcurrentLabelingJob as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 3 if res["stop_reason"].startswith(("stopped", "interrupted")) else 0


if __name__ == "__main__":
    sys.exit(main())
