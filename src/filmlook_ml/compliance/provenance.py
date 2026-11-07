"""Build and interpret label provenance (docs/AI_DATA_POLICY.md section 4)."""
from __future__ import annotations

from filmlook_ml.compliance.policy import LOCAL_SOURCES, THIRD_PARTY_SOURCES, claude_training_permission, worst
from filmlook_ml.labeling.schema import Provenance


def make_provenance(source: str, image: dict, *, model_id: str | None = None, reviewer: str | None = None,
                    claude_cfg: dict | None = None, annotated_at: str | None = None) -> Provenance:
    """Provenance for a NEW label on manifest record ``image`` (needs image_id, dataset_id, license)."""
    base = dict(image_id=image["image_id"], image_dataset_id=image.get("dataset_id"), image_license=image.get("license"),
                label_source=source)
    if annotated_at:
        base["annotated_at"] = annotated_at
    if source in LOCAL_SOURCES:
        basis = {"human": f"human annotation by {reviewer or 'unknown reviewer'}",
                 "heuristic": "deterministic formulas over measured statistics (local, no AI provider)",
                 "synthetic_ordering": "known edit applied in-repo (local, no AI provider)"}[source]
        return Provenance(**base, labeling_method=LOCAL_SOURCES[source], model_id=model_id,
                          training_permission="permitted", permission_basis=basis)
    if source in THIRD_PARTY_SOURCES:
        status, basis = claude_training_permission(claude_cfg or {})
        return Provenance(**base, labeling_method="third_party_api", provider="anthropic", model_id=model_id,
                          training_permission=status, permission_basis=basis)
    if source == "mock":
        return Provenance(**base, labeling_method="mock", model_id=model_id, training_permission="not_permitted",
                          permission_basis="mock output; never a training target")
    raise ValueError(f"unknown label source {source!r}")


def effective_provenance(record, image: dict | None, claude_cfg: dict | None) -> Provenance:
    """Provenance to use at preflight time for a stored label/pair/judgement (``record`` is a pydantic model
    with ``source`` and optional ``provenance``). Legacy records without the field are derived from their
    source; third-party-API labels are re-evaluated against the CURRENT config and the stricter status wins."""
    source = record.source
    img = image or {"image_id": getattr(record, "image_id", getattr(record, "image_a", "?")),
                    "dataset_id": None, "license": None}
    stored = getattr(record, "provenance", None)
    if stored is None:
        stored = make_provenance(source, img, model_id=getattr(record, "teacher_model", None),
                                 reviewer=getattr(record, "reviewer", None), claude_cfg=claude_cfg,
                                 annotated_at=getattr(record, "created_at", None))
    if source in THIRD_PARTY_SOURCES:
        now, basis = claude_training_permission(claude_cfg or {})
        status = worst(stored.training_permission, now)
        if status != stored.training_permission:
            stored = stored.model_copy(update={"training_permission": status, "permission_basis": basis})
    if image is not None and (stored.image_license is None or stored.image_dataset_id is None):
        stored = stored.model_copy(update={"image_license": image.get("license"), "image_dataset_id": image.get("dataset_id")})
    return stored
