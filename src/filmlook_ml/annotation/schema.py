"""Strict schemas for categorization annotations (schema categorization-2.0).

``CategorizationResult`` is what the annotator returns (API structured output, or an interactive pilot
annotation): categorical fields, model-reported confidences and a few short evidence phrases. It contains no
explanations or reasoning text. ``Annotation`` is the stored record: the result plus image metadata (kept apart
from inferred labels), provenance, QA flags and request metadata.

Confidence values are the annotator's self-reports. They are not calibrated probabilities.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.labeling.schema import Provenance

ANNOTATION_SCHEMA_VERSION = "categorization-2.0"


def _lit(values):
    return Literal[tuple(values)]  # type: ignore[valid-type]


PhotographerLabel = _lit(T.PHOTOGRAPHER_LABELS)
PhotographerName = _lit(list(T.PHOTOGRAPHERS))
FilmLabel = _lit(T.FILM_LABELS)
FilmName = _lit(list(T.FILM_STOCKS))
Evidence = list[str]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_call(m):
    if len(m.alternatives) > T.MAX_ALTERNATIVES or len(set(m.alternatives)) != len(m.alternatives):
        raise ValueError(f"at most {T.MAX_ALTERNATIVES} distinct alternatives")
    if m.primary in m.alternatives:
        raise ValueError("primary must not be repeated in alternatives")
    if not 1 <= len(m.evidence) <= T.MAX_EVIDENCE:
        raise ValueError(f"1-{T.MAX_EVIDENCE} evidence phrases")
    if any(not e.strip() or len(e) > T.EVIDENCE_MAX_CHARS for e in m.evidence):
        raise ValueError(f"evidence phrases must be non-empty and <= {T.EVIDENCE_MAX_CHARS} chars")
    return m


class PhotographerCall(_Strict):
    primary: PhotographerLabel
    alternatives: list[PhotographerName] = Field(default_factory=list, description="0-2 secondary resemblances")
    confidence: float = Field(ge=0, le=1, description="self-reported resemblance strength of the primary label")
    evidence: Evidence = Field(description="1-3 short visual observations")
    _v = model_validator(mode="after")(_check_call)


class FilmCall(_Strict):
    primary: FilmLabel
    alternatives: list[FilmName] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evidence: Evidence
    _v = model_validator(mode="after")(_check_call)


class Color(_Strict):
    temperature: _lit(T.COLOR["temperature"])
    saturation: _lit(T.COLOR["saturation"])
    dominant_colors: list[_lit(T.DOMINANT_COLORS)]
    color_cast: _lit(T.COLOR["color_cast"])

    @field_validator("dominant_colors")
    @classmethod
    def _dc(cls, v):
        if not 1 <= len(v) <= 3 or len(set(v)) != len(v):
            raise ValueError("1-3 distinct dominant colours")
        return v


class Lighting(_Strict):
    source: _lit(T.LIGHTING["source"])
    quality: _lit(T.LIGHTING["quality"])
    key: _lit(T.LIGHTING["key"])
    direction: _lit(T.LIGHTING["direction"])


class Tone(_Strict):
    contrast: _lit(T.TONE["contrast"])
    highlight_rolloff: _lit(T.TONE["highlight_rolloff"])
    shadow_depth: _lit(T.TONE["shadow_depth"])


class Texture(_Strict):
    visible_grain: _lit(T.TEXTURE["visible_grain"])
    grain_appearance: _lit(T.TEXTURE["grain_appearance"])
    sharpness: _lit(T.TEXTURE["sharpness"])


class Attributes(_Strict):
    color: Color
    lighting: Lighting
    tone: Tone
    texture: Texture
    genre: list[_lit(T.GENRES)]
    overall_aesthetic: list[_lit(T.AESTHETICS)]
    image_flags: list[_lit(T.IMAGE_FLAGS)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _lists(self):
        for name, v, lo, hi in (("genre", self.genre, 1, 3), ("overall_aesthetic", self.overall_aesthetic, 1, 3),
                                ("image_flags", self.image_flags, 0, len(T.IMAGE_FLAGS))):
            if not lo <= len(v) <= hi or len(set(v)) != len(v):
                raise ValueError(f"{name}: {lo}-{hi} distinct values")
        mono = self.color.saturation == "monochrome"
        if mono and (self.color.temperature != "not_applicable" or self.color.color_cast != "not_applicable"):
            raise ValueError("monochrome images use temperature/color_cast = not_applicable")
        if not mono and "not_applicable" in (self.color.temperature, self.color.color_cast):
            raise ValueError("not_applicable is reserved for monochrome images")
        if self.texture.visible_grain == "none" and self.texture.grain_appearance != "indeterminate":
            raise ValueError("grain_appearance must be indeterminate when visible_grain is none")
        return self


class CategorizationResult(_Strict):
    photographer_style: PhotographerCall
    film_aesthetic: FilmCall
    attributes: Attributes


class FilmStockMetadata(_Strict):
    """Film stock named in METADATA (never inferred from pixels). Uploader tags are asserted, not verified."""
    value: str
    source: Literal["uploader_tag", "exif"]
    verified: bool = False


class ImageMetadata(_Strict):
    capture_medium: str | None = None      # from EXIF device (film scanner / camera / phone), see DATASET_SOURCES.md
    camera_or_scanner: str | None = None
    film_stock_metadata: list[FilmStockMetadata] = Field(default_factory=list)


class QA(_Strict):
    flags: list[str] = Field(default_factory=list)
    needs_review: bool = False
    measured: dict = Field(default_factory=dict)      # deterministic statistics used by the consistency checks


class Annotation(_Strict):
    schema_version: Literal["categorization-2.0"] = ANNOTATION_SCHEMA_VERSION
    taxonomy_version: str = T.TAXONOMY_VERSION
    image_id: str
    image_path: str                                    # relative to the repository root
    sha256: str
    photographer_style: PhotographerCall
    film_aesthetic: FilmCall
    attributes: Attributes
    metadata: ImageMetadata                            # verified / asserted metadata, separate from inferred labels
    annotation_source: Literal["claude_vision"] = "claude_vision"
    annotation_type: Literal["inferred_aesthetic_labels"] = "inferred_aesthetic_labels"
    annotation_interface: Literal["claude_code_interactive", "anthropic_api"]
    provenance: Provenance
    image_source: dict                                 # dataset, license, license_url, source_url, attribution, split
    qa: QA
    review_status: Literal["unreviewed", "accepted", "corrected", "rejected"] = "unreviewed"
    training_eligible: bool                            # permission permitted AND license ok AND not needing review
    prompt_version: str
    request: dict = Field(default_factory=dict)        # model, elapsed_s, tokens/cost when available
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def _eligibility(self):
        if self.training_eligible and (self.provenance.training_permission != "permitted" or self.qa.needs_review):
            raise ValueError("training_eligible requires permitted provenance and no pending review")
        return self


class TrainingApprovedAnnotation(_Strict):
    """Record in data/annotations/training_approved/. Created only by annotation.export."""
    annotation: Annotation
    usable_for_training: Literal[True] = True
    approved_basis: str
    approved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
