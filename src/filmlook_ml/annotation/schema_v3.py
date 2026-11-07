"""Schema categorization-3.0: relative rankings kept apart from reliable class assignment.

The annotator returns:
- a RANKING of photographers and film stocks (top 3 with uncalibrated similarity estimates) whenever the image
  can be analysed, plus explicit flags saying whether the evidence SUPPORTS a meaningful resemblance;
- a broad film-aesthetic category and an apparent-capture impression;
- the learnable visual attributes (the initial supervised targets).

Training labels are derived later by rule (``labels_v3``). The top-ranked name never becomes a positive class
by itself. Metadata (uploader tags, EXIF capture medium) is stored apart from inferred labels and is never
shown to the annotator.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.schema import Color, FilmStockMetadata, Lighting, Texture, Tone
from filmlook_ml.labeling.schema import Provenance

SCHEMA_V3 = "categorization-3.0"


def _lit(values):
    return Literal[tuple(values)]  # type: ignore[valid-type]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _evidence_ok(ev: list[str], lo: int = 1) -> None:
    if not lo <= len(ev) <= T.MAX_EVIDENCE or any(not e.strip() or len(e) > T.EVIDENCE_MAX_CHARS for e in ev):
        raise ValueError(f"{lo}-{T.MAX_EVIDENCE} evidence phrases of <= {T.EVIDENCE_MAX_CHARS} chars")


class Ranked(_Strict):
    name: str
    similarity: float = Field(ge=0, le=1, description="model-generated, uncalibrated")


def _ranking_ok(top: list[Ranked], allowed: set[str]) -> None:
    names = [r.name for r in top]
    if len(top) != T.TOP_K or len(set(names)) != T.TOP_K:
        raise ValueError(f"exactly {T.TOP_K} distinct ranked candidates")
    if any(n not in allowed for n in names):
        raise ValueError(f"ranked names must come from the taxonomy: {sorted(set(names) - allowed)}")
    sims = [r.similarity for r in top]
    if sims != sorted(sims, reverse=True):
        raise ValueError("ranking must be ordered by descending similarity")


def _allowed(info: ValidationInfo | None, key: str, default) -> set[str]:
    """Allowed names/ids: the database taxonomy passed as validation context, else taxonomy-v3 defaults."""
    ctx = (info.context or {}) if info is not None else {}
    return set(ctx.get(key) or default)


class PhotographerRanking(_Strict):
    analyzable: bool
    closest_photographer: str | None = None
    top_3_photographers: list[Ranked] = Field(default_factory=list)
    style_match_supported: bool = False
    none_of_the_above: bool = False
    style_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v(self, info: ValidationInfo):
        if not self.analyzable:
            if self.closest_photographer or self.top_3_photographers or self.style_match_supported:
                raise ValueError("a non-analyzable image has no ranking")
            _evidence_ok(self.style_evidence)
            return self
        _ranking_ok(self.top_3_photographers, _allowed(info, "photographers", T.PHOTOGRAPHERS))
        if self.closest_photographer != self.top_3_photographers[0].name:
            raise ValueError("closest_photographer must equal the first ranked candidate")
        if self.none_of_the_above and self.style_match_supported:
            raise ValueError("none_of_the_above excludes style_match_supported")
        if not self.none_of_the_above and not self.style_match_supported:
            raise ValueError("choose: style_match_supported, or none_of_the_above")
        _evidence_ok(self.style_evidence)
        return self


class FilmRanking(_Strict):
    analyzable: bool
    top_3_film_stocks: list[Ranked] = Field(default_factory=list)
    stock_match_supported: bool = False
    aesthetic_category: _lit(list(T.FILM_CATEGORIES))
    apparent_capture: _lit(T.APPARENT_CAPTURE)
    film_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v(self, info: ValidationInfo):
        if self.analyzable:
            _ranking_ok(self.top_3_film_stocks, _allowed(info, "film", T.FILM_STOCKS))
        elif self.top_3_film_stocks or self.stock_match_supported:
            raise ValueError("a non-analyzable image has no film ranking")
        if self.stock_match_supported and self.aesthetic_category in (
                "clean_digital", "unknown_indeterminate", "digital_film_filter_emulation"):
            raise ValueError(f"a specific stock match is unsupported for category {self.aesthetic_category}")
        _evidence_ok(self.film_evidence)
        return self


class AttributesV3(_Strict):
    color: Color
    lighting: Lighting
    tone: Tone
    texture: Texture
    genre: list[_lit(T.GENRES)]
    image_flags: list[_lit(T.IMAGE_FLAGS_V3)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v(self):
        if not 1 <= len(self.genre) <= 3 or len(set(self.genre)) != len(self.genre):
            raise ValueError("genre: 1-3 distinct values")
        if len(set(self.image_flags)) != len(self.image_flags):
            raise ValueError("duplicate image_flags")
        mono = self.color.saturation == "monochrome"
        if mono != (self.color.temperature == "not_applicable") or mono != (self.color.color_cast == "not_applicable"):
            raise ValueError("not_applicable temperature/color_cast <=> monochrome")
        if self.texture.visible_grain == "none" and self.texture.grain_appearance != "indeterminate":
            raise ValueError("grain_appearance must be indeterminate when visible_grain is none")
        return self


class CategorizationV3(_Strict):
    photographer: PhotographerRanking
    film: FilmRanking
    attributes: AttributesV3

    @model_validator(mode="after")
    def _v(self, info: ValidationInfo):
        mono = self.attributes.color.saturation == "monochrome"
        bw = {"bw_negative", "high_grain_monochrome"}
        if self.film.aesthetic_category in bw and not mono:
            raise ValueError("black-and-white film categories require a monochrome image")
        if self.film.stock_match_supported and self.film.top_3_film_stocks:
            top = self.film.top_3_film_stocks[0].name
            bw_names = _allowed(info, "film_monochrome", T.BW_STOCKS)
            if (top in bw_names) != mono:
                raise ValueError("a supported top stock must match the image's monochrome status")
        return self


class MetadataV3(_Strict):
    """Facts from metadata only. Never shown to the annotator; compared with its impressions afterwards."""
    capture_type_exif: str | None = None                  # film_scan / digital_camera / smartphone / ... / None
    camera_or_scanner: str | None = None
    verified_film_stock: str | None = None                # only if independently verified; none are in v1 data
    uploader_asserted_film_stock: list[FilmStockMetadata] = Field(default_factory=list)


class QAV3(_Strict):
    flags: list[str] = Field(default_factory=list)
    needs_review: bool = False
    measured: dict = Field(default_factory=dict)


class AnnotationV3(_Strict):
    schema_version: Literal["categorization-3.0"] = SCHEMA_V3
    taxonomy_version: str = T.TAXONOMY_V3_VERSION
    image_id: str
    image_path: str
    sha256: str
    result: CategorizationV3
    metadata: MetadataV3
    annotation_source: Literal["claude_vision"] = "claude_vision"
    annotation_type: Literal["inferred_aesthetic_labels"] = "inferred_aesthetic_labels"
    annotation_interface: Literal["claude_code_interactive", "anthropic_api"]
    blind_to_metadata: bool                               # annotator saw only pixels (no tags / capture strata)
    provenance: Provenance
    image_source: dict
    qa: QAV3
    review_status: Literal["unreviewed", "accepted", "corrected", "rejected"] = "unreviewed"
    attributes_training_eligible: bool                    # Stage A targets
    prompt_version: str
    pilot: str | None = None
    request: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def _v(self):
        if self.attributes_training_eligible and (self.provenance.training_permission != "permitted"
                                                  or self.qa.needs_review):
            raise ValueError("attributes_training_eligible requires permitted provenance and no pending review")
        return self
