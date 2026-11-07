"""Categorization v3: prompt, quarantine, QA, record construction and the label rules for training.

Label rules (``labels_v3``). Rankings are kept for retrieval and evaluation but never become class labels on
their own:
- Stage A attribute targets: every attribute of an annotation that is ``attributes_training_eligible``.
- Stage B/C photographer positive: analyzable, ``style_match_supported``, top similarity >=
  STYLE_POSITIVE_MIN, and ``review_status`` accepted or corrected (human-confirmed). Otherwise there is NO
  photographer class label. none_of_the_above becomes a negative only when the annotator said so explicitly.
- Film stock positive: the same rule with ``stock_match_supported``. The broad film category is a Stage A
  attribute-like target when the annotation is eligible.
"""
from __future__ import annotations

from pathlib import Path

from filmlook_ml.annotation import qa
from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.schema_v3 import AnnotationV3, CategorizationV3, MetadataV3, QAV3
from filmlook_ml.annotation.workflow import film_stock_tags
from filmlook_ml.compliance.provenance import make_provenance

PROMPT_VERSION_V3 = "categorize-v3"
STYLE_POSITIVE_MIN = 0.6
HUMAN_CONFIRMED = {"accepted", "corrected"}
# QA flags that only inform. Every other flag sends the image to manual review.
INFO_FLAGS_V3 = {"style_none_of_the_above", "film_category_unknown"}


def system_prompt_v3() -> str:
    ph = "\n".join(f"  - {k}: {v}" for k, v in T.PHOTOGRAPHERS.items())
    fs = "\n".join(f"  - {k}: {v}" for k, v in T.FILM_STOCKS.items())
    fc = "\n".join(f"  - {k}: {v}" for k, v in T.FILM_CATEGORIES.items())
    return f"""You categorize photographs for FilmLook-ML, a specialized photography-aesthetics classification
project. Your structured labels are used to develop and evaluate a small DINOv2-small classifier. You see only
the pixels: a full view, and a native-resolution centre crop for grain and sharpness.

photographer: if the image can be analysed, ALWAYS rank the 3 closest styles below, with model-generated
similarity estimates in 0-1 (uncalibrated; they need not be high), in descending order; closest_photographer is
the first. Then decide separately whether the visual evidence SUPPORTS a meaningful resemblance
(style_match_supported) or whether all candidates are poor matches (none_of_the_above). Exactly one of the two
is true. A ranking alone is not a match. Resemblance only, never authorship; ignore watermarks and signatures.
{ph}
film: if the image can be analysed, ALWAYS rank the 3 closest stock LOOKS below; stock_match_supported is true
only when the rendering clearly resembles the top stock. Also pick one broad aesthetic_category and an
apparent_capture impression. These are visual impressions, not claims about the actual process or stock.
{fs}
  categories:
{fc}
Black-and-white stocks and categories apply only to monochrome images.
evidence: 1-3 short visual observations (at most 80 characters each) for photographer and for film.
attributes: only what is visible, using the allowed values; prefer "indeterminate" or "unknown" over guessing.
Use temperature and color_cast "not_applicable" exactly when the image is monochrome. Add the image_flag
"misoriented" when the frame is clearly rotated or upside down.
Set analyzable false (with no rankings) only when the image cannot be judged.
Return only the structured result."""


# ----------------------------------------------------------------------------- QA
def check_v3(res: CategorizationV3, measured: dict) -> list[str]:
    flags: list[str] = []
    ph, fi, at = res.photographer, res.film, res.attributes
    if not ph.analyzable or not fi.analyzable:
        flags.append("not_analyzable")
    if ph.analyzable and ph.none_of_the_above:
        flags.append("style_none_of_the_above")
    if ph.style_match_supported and ph.top_3_photographers[0].similarity < STYLE_POSITIVE_MIN:
        flags.append("style_supported_but_low_similarity")
    if fi.aesthetic_category == "unknown_indeterminate":
        flags.append("film_category_unknown")
    if T.ORIENTATION_FLAG in at.image_flags:
        flags.append("misoriented")
    if {"not_a_photograph", "artwork_reproduction"} & set(at.image_flags):
        flags.append("not_a_photograph")
    if measured:
        tags, mono_claim = set(measured["tags"]), at.color.saturation == "monochrome"
        # toned archival prints measure as colour: only flag the opposite direction, or colour claims on grey images
        if mono_claim != measured["is_mono"] and not (mono_claim and fi.aesthetic_category == "archival_early_process"):
            flags.append("monochrome_disagrees_with_measurement")
        if not mono_claim and not measured["is_mono"]:
            t = at.color.temperature
            if (t == "warm" and "cool" in tags) or (t == "cool" and "warm" in tags):
                flags.append("temperature_disagrees_with_measurement")
            if (at.color.saturation == "high" and "muted" in tags) or (at.color.saturation == "low" and "vivid" in tags):
                flags.append("saturation_disagrees_with_measurement")
        g = at.texture.visible_grain
        if g in ("moderate", "strong") and measured["noise_std"] < qa.THRESHOLDS["grain_none"]:
            flags.append("grain_claim_unsupported")
        if g == "none" and measured["noise_std"] > qa.THRESHOLDS["grain_coarse"]:
            flags.append("grain_missed")
        c, k = at.tone.contrast, at.lighting.key
        dark = "low_key" in tags                       # y_std is low on mostly dark frames: skip that comparison
        if (c == "high" and "low_contrast" in tags and not dark) or (c == "low" and "high_contrast" in tags):
            flags.append("contrast_disagrees_with_measurement")
        if (k == "high_key" and "low_key" in tags) or (k == "low_key" and "high_key" in tags):
            flags.append("key_disagrees_with_measurement")
    return flags


def needs_review_v3(flags: list[str]) -> bool:
    return any(f not in INFO_FLAGS_V3 for f in flags)


# ----------------------------------------------------------------------------- record construction
def metadata_v3(r: dict, usertags: dict[str, str]) -> MetadataV3:
    tags = usertags.get(str(r.get("source_id")), "") if r["dataset_id"] == "commoncatalog_ccby" else ""
    return MetadataV3(capture_type_exif=r.get("medium"), camera_or_scanner=r.get("camera") or None,
                      verified_film_stock=None, uploader_asserted_film_stock=film_stock_tags(tags))


def build_v3(res: CategorizationV3, r: dict, *, interface: str, ccfg: dict, model_id: str, features: dict | None,
             metadata: MetadataV3, request: dict, root: Path, blind: bool, pilot: str | None = None) -> AnnotationV3:
    measured = qa.measured_summary(features)
    flags = check_v3(res, measured)
    prov = make_provenance("claude", r, model_id=model_id, claude_cfg=ccfg)
    if interface == "claude_code_interactive":
        prov = prov.model_copy(update={"labeling_method": "claude_code_interactive"})
    review = needs_review_v3(flags)
    path = Path(r["path"])
    rel = str(path.relative_to(root)) if path.is_absolute() and path.is_relative_to(root) else str(path)
    return AnnotationV3(
        image_id=r["image_id"], image_path=rel, sha256=r["sha256"], result=res, metadata=metadata,
        annotation_interface=interface, blind_to_metadata=blind, provenance=prov,
        image_source={k: r.get(k) for k in ("dataset_id", "license", "license_url", "source_url", "attribution",
                                            "author", "group_id", "family_id", "split")},
        qa=QAV3(flags=flags, needs_review=review, measured=measured),
        attributes_training_eligible=prov.training_permission == "permitted" and not review,
        prompt_version=PROMPT_VERSION_V3, pilot=pilot, request=request)


# ----------------------------------------------------------------------------- label rules
def attribute_targets(a: AnnotationV3) -> dict[str, str] | None:
    """Stage A targets (single-label attributes), or None when the annotation is not eligible."""
    if not a.attributes_training_eligible:
        return None
    at = a.result.attributes
    return {"temperature": at.color.temperature, "saturation": at.color.saturation, "contrast": at.tone.contrast,
            "highlight_rolloff": at.tone.highlight_rolloff, "shadow_depth": at.tone.shadow_depth,
            "visible_grain": at.texture.visible_grain, "lighting_quality": at.lighting.quality,
            "film_category": a.result.film.aesthetic_category}


def multilabel_targets(a: AnnotationV3) -> dict[str, list[str]] | None:
    if not a.attributes_training_eligible:
        return None
    return {"genre": list(a.result.attributes.genre), "palette": list(a.result.attributes.color.dominant_colors)}


def style_label(a: AnnotationV3) -> str | None:
    """Reliable photographer label for Stage B references / Stage C heads, or None (no label)."""
    ph = a.result.photographer
    if a.provenance.training_permission != "permitted" or a.review_status not in HUMAN_CONFIRMED:
        return None
    if ph.analyzable and ph.none_of_the_above:
        return T.NONE
    if ph.style_match_supported and ph.top_3_photographers[0].similarity >= STYLE_POSITIVE_MIN:
        return ph.closest_photographer
    return None


def stock_label(a: AnnotationV3) -> str | None:
    fi = a.result.film
    if a.provenance.training_permission != "permitted" or a.review_status not in HUMAN_CONFIRMED:
        return None
    if fi.stock_match_supported and fi.top_3_film_stocks[0].similarity >= STYLE_POSITIVE_MIN:
        return fi.top_3_film_stocks[0].name
    return None


# ----------------------------------------------------------------------------- storage (v3 lives beside v2)
class StoreV3:
    """data/annotations/claude_exploratory/annotations_v3/<image_id>.json: atomic, never overwritten."""

    def __init__(self, root: Path):
        from filmlook_ml.annotation.workflow import atomic_write_text
        self._write = atomic_write_text
        self.dir = Path(root) / "claude_exploratory"
        self.ann_dir = self.dir / "annotations_v3"
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        tax = self.dir / f"taxonomy_{T.TAXONOMY_V3_VERSION}.json"
        if not tax.exists():
            import json
            self._write(tax, json.dumps(T.describe_v3(), indent=1))

    def path(self, image_id: str) -> Path:
        return self.ann_dir / f"{image_id}.json"

    def has(self, image_id: str) -> bool:
        return self.path(image_id).exists()

    def write(self, a: AnnotationV3) -> Path:
        p = self.path(a.image_id)
        self._write(p, a.model_dump_json(indent=1))
        return p

    def load_all(self) -> list[AnnotationV3]:
        return [AnnotationV3.model_validate_json(p.read_text()) for p in sorted(self.ann_dir.glob("*.json"))]
