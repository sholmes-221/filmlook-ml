#!/usr/bin/env python
"""Compliance preflight (docs/AI_DATA_POLICY.md). Run before any training, evaluation or AI-assisted labeling.

  python scripts/preflight_check.py                              # training/evaluation preflight (default)
  python scripts/preflight_check.py --mode labeling --intended-use evaluation --images 40 --pairs 10

Exit status 0 = PASS, 1 = FAIL. The same checks run automatically inside training.data.load_train_data()
(train / compare / evaluate / ablate) and inside the Claude labeling scripts; this CLI is the stand-alone
version for CI and for the pre-training checklist. Writes reports/preflight_<mode>.json.
"""
import argparse
import json

from filmlook_ml import paths as P
from filmlook_ml.compliance.preflight import run_labeling_preflight, run_training_preflight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["training", "labeling"], default="training")
    ap.add_argument("--intended-use", choices=["training", "evaluation", "exploratory"], default=None, help="labeling mode")
    ap.add_argument("--images", type=int, default=40, help="labeling mode: images planned for this run")
    ap.add_argument("--pairs", type=int, default=10, help="labeling mode: pairs planned for this run")
    ap.add_argument("--json", action="store_true", help="print the JSON report instead of the summary")
    a = ap.parse_args()
    lcfg = P.load_yaml("labeling.yaml")
    if a.mode == "training":
        rep = run_training_preflight(root=P.ROOT, label_dir=P.LABELS, manifest_path=P.MANIFESTS / "manifest.jsonl",
                                     labeling_cfg=lcfg, report_path=P.REPORTS / "preflight_training.json")
    else:
        from filmlook_ml.labeling.teachers import PAIR_V2, SYSTEM_V2, style_system_prompt
        rep = run_labeling_preflight(root=P.ROOT, ccfg=lcfg["claude"], intended_use=a.intended_use or "undeclared",
                                     n_images=a.images, n_pairs=a.pairs, dry_run=True,
                                     prompts={"SYSTEM_V2": SYSTEM_V2, "PAIR_V2": PAIR_V2,
                                              "style_system_prompt": style_system_prompt()},
                                     report_path=P.REPORTS / "preflight_labeling.json")
    print(json.dumps(rep.to_dict(), indent=1) if a.json else rep.summary())
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
