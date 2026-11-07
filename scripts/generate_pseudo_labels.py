#!/usr/bin/env python
"""Generate labels.

  python scripts/generate_pseudo_labels.py --teacher heuristic            # free, local (default)
  python scripts/generate_pseudo_labels.py --teacher claude --images 40 --pairs 10   # DRY RUN: estimate only
  python scripts/generate_pseudo_labels.py --teacher claude --images 40 --pairs 10 --no-dry-run --confirm-paid

Claude: the API key is read from ANTHROPIC_API_KEY or the git-ignored .env (never printed). The budget cap in
configs/labeling.yaml is cumulative over all runs (data/labels/claude_usage.jsonl). Re-running resumes from
the cache without repeating paid calls.

COMPLIANCE (docs/AI_DATA_POLICY.md): the Claude teacher requires --intended-use training|evaluation and runs
the labeling preflight first. With no written training authorization, `training` is refused and `evaluation`
needs claude.evaluation_use.approved in configs/labeling.yaml; labels are then stored usable_for_training=false.
"""
import argparse
import json

from filmlook_ml import paths as P
from filmlook_ml.compliance.policy import LabelingLock
from filmlook_ml.compliance.preflight import run_labeling_preflight
from filmlook_ml.compliance.provenance import make_provenance
from filmlook_ml.datasets.preprocess import read_jsonl, write_jsonl
from filmlook_ml.labeling.pipeline import (
    claude_labels,
    claude_selection,
    extract_features,
    heuristic_labels,
    synthetic_labels,
)
from filmlook_ml.labeling.teachers import PAIR_V2, SYSTEM_V2, ClaudeTeacher, api_key_status, training_authorization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["heuristic", "claude", "mock"], default=None)
    ap.add_argument("--images", type=int, default=40, help="Claude: images to label (incl. ~25%% edited variants)")
    ap.add_argument("--pairs", type=int, default=10, help="Claude: original-vs-edit pair comparisons")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--no-dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intended-use", choices=["training", "evaluation"], default=None,
                    help="Claude: declared use of the outputs (required; checked by the compliance preflight)")
    a = ap.parse_args()
    lcfg, tcfg = P.load_yaml("labeling.yaml"), P.load_yaml("training.yaml")
    teacher = a.teacher or lcfg["teacher"]
    manifest = read_jsonl(P.MANIFESTS / "manifest.jsonl")
    from filmlook_ml.training.data import feature_cache
    feats = extract_features(manifest, feature_cache(tcfg["max_side"], tcfg["texture_crop"]), tcfg["texture_crop"],
                             a.workers, tcfg["max_side"])

    if teacher in ("heuristic", "mock"):
        syn, pairs = synthetic_labels(manifest)
        write_jsonl(P.LABELS / "synthetic_labels.jsonl", [x.model_dump() for x in syn])
        write_jsonl(P.LABELS / "synthetic_pairs.jsonl", [p.model_dump() for p in pairs])
        print(f"synthetic-ordering labels: {len(syn)} images, {len(pairs)} pairs")
    if teacher == "heuristic":
        labs = heuristic_labels(manifest, feats, lcfg["prompt_version"])
        write_jsonl(P.LABELS / "pseudo_heuristic.jsonl", [x.model_dump() for x in labs])
        print(f"heuristic pseudo-labels: {len(labs)}")
    elif teacher == "mock":
        from pathlib import Path

        from filmlook_ml.labeling.schema import ImageLabel
        from filmlook_ml.labeling.teachers import MockTeacher
        t = MockTeacher()
        labs = [ImageLabel(image_id=r["image_id"], sha256=r["sha256"], source="mock", teacher_model=t.model_id,
                           prompt_version="mock", label=t.label(Path(r["path"])), usable_for_training=False,
                           provenance=make_provenance("mock", r, model_id=t.model_id))
                for r in manifest[: a.images]]
        write_jsonl(P.LABELS / "pseudo_mock.jsonl", [x.model_dump() for x in labs])
        print(f"mock labels: {len(labs)} (never used for training)")
    else:
        ccfg = lcfg["claude"]
        dry = not a.no_dry_run
        pre = run_labeling_preflight(root=P.ROOT, ccfg=ccfg, intended_use=a.intended_use or "undeclared",
                                     n_images=a.images, n_pairs=a.pairs, dry_run=dry,
                                     prompts={"SYSTEM_V2": SYSTEM_V2, "PAIR_V2": PAIR_V2},
                                     report_path=P.REPORTS / "preflight_labeling.json")
        print(pre.summary())
        if not pre.ok:
            raise SystemExit("labeling preflight failed; Claude labeling must not proceed (docs/AI_DATA_POLICY.md)")
        t = ClaudeTeacher(ccfg, P.LABELS / "claude_cache.jsonl", lcfg["prompt_version"], dry_run=dry,
                          confirm=a.confirm_paid, usage_log=P.LABELS / "claude_usage.jsonl")
        consented = set(ccfg["consented_datasets"])
        sel, pairs = claude_selection(manifest, consented, a.images, a.pairs, a.seed)
        est = t.estimate(len(sel), len(pairs))
        print(json.dumps({"model": ccfg["model"], "api_key": api_key_status(P.ROOT), "selected_images": len(sel),
                          "selected_pairs": len(pairs), "estimate": est,
                          "spent_so_far_usd": round(t.cost.spent_before, 4), "cap_usd": ccfg["max_budget_usd"],
                          "training_authorization": training_authorization(ccfg), "dry_run": dry}, indent=1))
        if not dry and api_key_status(P.ROOT) == "missing":
            raise SystemExit("ANTHROPIC_API_KEY not found in environment or .env")
        with LabelingLock(P.LABELS / "claude_labeling.lock"):
            labs, plabs, rep = claude_labels(sel, pairs, t, consented, feats)
        if labs:
            with open(P.LABELS / "pseudo_claude.jsonl", "a") as fh:
                for x in labs:
                    fh.write(x.model_dump_json() + "\n")
        if plabs:
            with open(P.LABELS / "claude_pairs.jsonl", "a") as fh:
                for x in plabs:
                    fh.write(x.model_dump_json() + "\n")
        flagged = [{"image_id": x.image_id, "flags": x.flags} for x in labs if x.flags]
        write_jsonl(P.LABELS / "claude_review_queue.jsonl", flagged)
        P.REPORTS.mkdir(parents=True, exist_ok=True)
        (P.REPORTS / "claude_labeling_cost.json").write_text(json.dumps({**rep, "estimate": est}, indent=1))
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
