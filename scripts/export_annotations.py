#!/usr/bin/env python
"""Export exploratory categorization annotations to data/annotations/training_approved/.

Refused unless training on Claude outputs is `permitted`, which requires a written authorization reference
(docs/AI_DATA_POLICY.md sections 2 and 9). Annotations stamped `uncertain` are never exported.

  python scripts/export_annotations.py
"""
import json
import sys

from filmlook_ml import paths as P
from filmlook_ml.annotation.export import ExportBlocked, export_training
from filmlook_ml.annotation.workflow import AnnotationStore


def main():
    ccfg = P.load_yaml("labeling.yaml")["claude"]
    try:
        res = export_training(AnnotationStore(P.DATA / "annotations"), ccfg)
    except ExportBlocked as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
