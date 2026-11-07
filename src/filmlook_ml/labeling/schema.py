"""Structured label schema (v2) shared by teachers, human review, and training.

Two forms:
* ``TeacherOutput``    - internal, dict-based; what is stored and trained on.
* ``ClaudeAssessment`` - explicit-field model sent to Claude as a strict structured-output schema
                         (no free-form dicts), converted to TeacherOutput by ``to_teacher_output``.
Attribute direction: higher = more film-like / natural (see filmlook_infer.vocab.RUBRIC).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from filmlook_ml.constants import ATTRIBUTES, COLOR_ATTRIBUTES, MOOD_TAGS

LabelSource = Literal["human", "synthetic_ordering", "claude", "heuristic", "mock"]
LabelingMethod = Literal["human_review", "heuristic_formula", "synthetic_edit_ordering", "third_party_api",
                         "claude_code_interactive", "local_model", "mock"]
TrainingPermission = Literal["permitted", "uncertain", "not_permitted"]
ReviewStatus = Literal["unreviewed", "accepted", "corrected", "rejected", "needs_review"]
Temperature = Literal["warm", "cool", "neutral", "not_applicable"]
Saturation = Literal["muted", "natural", "vivid", "not_applicable"]
Sharpness = Literal["soft", "natural", "crisp"]
Key = Literal["high_key", "balanced", "low_key"]
Grain = Literal["no_visible_grain", "fine_grain", "coarse_grain"]
Mood = Literal["cinematic", "documentary", "editorial", "nostalgic"]
assert list(Mood.__args__) == MOOD_TAGS


class StyleVector(BaseModel):
    temperature: Temperature
    saturation: Saturation
    sharpness: Sharpness
    key: Key
    grain: Grain
    mood: list[Mood] = Field(default_factory=list, max_length=2,
                             description="0-2 tags, only when visually justified; empty is fine")


class TeacherOutput(BaseModel):
    film_score: float = Field(ge=0, le=100)
    attributes: dict[str, float | None]
    style: StyleVector | None = None
    monochrome: bool | None = None
    uncertainty: Literal["low", "medium", "high"] = "medium"
    label_confidence: float = Field(ge=0, le=1, description="self-reported; NOT a calibrated probability")
    strengths: list[str] = Field(default_factory=list, max_length=3)
    weaknesses: list[str] = Field(default_factory=list, max_length=3)
    visual_evidence: str = Field(default="", max_length=2000)

    @field_validator("attributes")
    @classmethod
    def _check_attrs(cls, v):
        if set(v) != set(ATTRIBUTES):
            raise ValueError(f"attributes must be exactly {ATTRIBUTES}, got {sorted(v)}")
        for k, x in v.items():
            if x is None:
                if k not in COLOR_ATTRIBUTES:
                    raise ValueError(f"{k} may not be null")
            elif not 0.0 <= x <= 1.0:
                raise ValueError(f"attribute {k}={x} outside [0,1]")
        return v


# --------------------------------------------------------------------------- Claude API schema
class ClaudeAttributes(BaseModel):
    highlight_rolloff: float = Field(description="0-1")
    color_response: float | None = Field(description="0-1, or null if the image is monochrome")
    microcontrast: float = Field(description="0-1")
    grain_naturalness: float = Field(description="0-1")
    tonal_depth: float = Field(description="0-1")
    film_coherence: float = Field(description="0-1")


class ClaudeAssessment(BaseModel):
    monochrome: bool
    film_score: float = Field(description="0-100")
    attributes: ClaudeAttributes
    style: StyleVector
    uncertainty: Literal["low", "medium", "high"]
    ambiguity_reasons: list[str] = Field(description="why the image is hard to judge; empty if not")
    strengths: list[str] = Field(description="1-3 short phrases, strongest positive rendering traits")
    weaknesses: list[str] = Field(description="0-3 short phrases, strongest negative rendering traits")
    label_confidence: float = Field(description="0-1")


def to_teacher_output(a: ClaudeAssessment) -> TeacherOutput:
    """Validate ranges client-side (the API schema cannot express them) and convert."""
    attrs = a.attributes.model_dump()
    if a.monochrome:
        attrs["color_response"] = None
    notes = "; ".join(a.ambiguity_reasons)[:2000]
    return TeacherOutput(film_score=a.film_score, attributes=attrs, style=a.style, monochrome=a.monochrome,
                         uncertainty=a.uncertainty, label_confidence=a.label_confidence,
                         strengths=a.strengths[:3], weaknesses=a.weaknesses[:3], visual_evidence=notes)


PairChoice = Literal["A", "B", "tie"]


class ClaudePairAssessment(BaseModel):
    """Two versions/photos compared; 'more film-like' per criterion. Ties are allowed."""
    film_score: PairChoice
    highlight_rolloff: PairChoice
    color_response: PairChoice
    microcontrast: PairChoice
    grain_naturalness: PairChoice
    tonal_depth: PairChoice
    film_coherence: PairChoice
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(description="one sentence")


# --------------------------------------------------------------------------- provenance
class Provenance(BaseModel):
    """Who/what produced a label, on which image, and whether training on it is permitted.

    Required by docs/AI_DATA_POLICY.md section 4 for every training label. ``training_permission`` is the
    status at annotation time; the training preflight re-evaluates third-party-API labels against the
    CURRENT configuration and uses the stricter of the two. ``permission_basis`` is a human-readable record
    (a license id, a reviewer name, or an authorization reference); it is never invented.
    """
    image_id: str
    image_dataset_id: str | None = None
    image_license: str | None = None                 # per-record license id (see configs/datasets.yaml)
    label_source: LabelSource
    labeling_method: LabelingMethod
    annotated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider: str | None = None                      # e.g. "anthropic"; None for human / local methods
    model_id: str | None = None
    training_permission: TrainingPermission
    permission_basis: str


# --------------------------------------------------------------------------- stored records
class ImageLabel(BaseModel):
    image_id: str
    sha256: str
    source: LabelSource
    teacher_model: str
    prompt_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    review_status: ReviewStatus = "unreviewed"
    reviewer: str | None = None
    label: TeacherOutput
    supervised_attributes: list[str] | None = None   # partial labels (synthetic orderings)
    usable_for_training: bool = True
    flags: list[str] = Field(default_factory=list)   # automatic consistency checks
    authorization: str | None = None                 # legacy free-text provenance (kept for old records)
    provenance: Provenance | None = None             # structured provenance (policy section 4)
    notes: str = ""


class PairLabel(BaseModel):
    """Relative judgement: ``winner`` looks more authentically film-like on ``attribute``."""
    image_a: str
    image_b: str
    attribute: str  # one of ATTRIBUTES or "film_score"
    winner: Literal["a", "b", "tie"]
    source: LabelSource
    reviewer: str | None = None
    confidence: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: Provenance | None = None


# --------------------------------------------------------------------------- photographer-style judgements
class StyleCandidate(BaseModel):
    id: str = Field(description="one of the listed style profile ids")
    resemblance: float = Field(description="0-1 visual resemblance of the PHOTO'S RENDERING to the described style")
    evidence: str = Field(description="one short phrase naming the visible shared characteristics")


class ClaudeStyleAssessment(BaseModel):
    no_reliable_match: bool = Field(description="true if no listed style is a meaningful match")
    candidates: list[StyleCandidate] = Field(description="0-3 best-matching styles, best first")
    uncertainty: Literal["low", "medium", "high"]


class StyleJudgement(BaseModel):
    """One independent judgement of which styles a photo resembles (human or authorized teacher)."""
    image_id: str
    sha256: str
    source: Literal["human", "claude"]
    reviewer: str | None = None
    matches: list[str] = Field(default_factory=list, max_length=3)   # ordered, best first
    no_match: bool = False
    uncertainty: Literal["low", "medium", "high"] = "medium"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: Provenance | None = None
