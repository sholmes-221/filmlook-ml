"""Export gate: exploratory annotations -> data/annotations/training_approved/.

Refuses unless BOTH hold:
1. the CURRENT configuration classifies training on Claude outputs as ``permitted``, which requires a written
   authorization reference (compliance.policy.claude_training_permission);
2. the annotation's own provenance was ``permitted`` when it was made. Annotations stamped ``uncertain`` are
   never revived; re-annotate after the authorization is recorded (docs/AI_DATA_POLICY.md section 5).
"""
from __future__ import annotations

from pathlib import Path

from filmlook_ml.annotation.schema import TrainingApprovedAnnotation
from filmlook_ml.annotation.workflow import AnnotationStore, atomic_write_text
from filmlook_ml.compliance.policy import claude_training_permission


class ExportBlocked(RuntimeError):
    pass


def export_training(store: AnnotationStore, ccfg: dict, out_dir: Path | None = None) -> dict:
    status, basis = claude_training_permission(ccfg)
    if status != "permitted":
        raise ExportBlocked(f"training permission is {status}: {basis}. Exploratory annotations stay exploratory.")
    out_dir = Path(out_dir or store.root / "training_approved")
    exported, refused, existing = 0, 0, 0
    for a in store.load_all():
        if a.provenance.training_permission != "permitted":
            refused += 1
            continue
        p = out_dir / f"{a.image_id}.json"
        if p.exists():
            existing += 1
            continue
        atomic_write_text(p, TrainingApprovedAnnotation(annotation=a, approved_basis=basis).model_dump_json(indent=1))
        exported += 1
    return {"exported": exported, "refused_uncertain_provenance": refused, "already_exported": existing,
            "basis": basis}
