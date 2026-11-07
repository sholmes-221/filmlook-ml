"""Resumable categorization workflow (Claude vision annotations, schema categorization-2.0).

Two annotation interfaces write the same validated record:
- ``claude_code_interactive``: a bounded pilot annotated inside an interactive Claude Code session at the owner's
  explicit request (scripts/categorize_pilot.py prepares views, then records each result);
- ``anthropic_api``: the API-key path for larger runs (scripts/annotate_categories.py), budget-capped.

Layout (git-ignored under data/annotations/):
  claude_exploratory/
    annotations/<image_id>.json   one validated Annotation per image; written atomically, never overwritten
    progress.jsonl, errors.jsonl  processing and error logs (no API response bodies)
    taxonomy.json                 the taxonomy the annotations were made under
  training_approved/              written only by annotation.export

API runs: sequential, a single job lock, a minimum request interval, and rate limits honoured. Authentication,
permission, bad-request, not-found, refusal, non-compliant-request and budget errors STOP the run; three
consecutive failures stop it too. Nothing is restarted automatically.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filmlook_ml.annotation import qa
from filmlook_ml.annotation import taxonomy as T
from filmlook_ml.annotation.schema import Annotation, CategorizationResult, FilmStockMetadata, ImageMetadata, QA
from filmlook_ml.compliance.policy import NonCompliantRequest, PolicyStop
from filmlook_ml.compliance.provenance import make_provenance
from filmlook_ml.labeling.teachers import BudgetExceeded, ClaudeTeacher, LabelCache, LabelingNotAuthorized, _img

PROMPT_VERSION = "categorize-v2"
NON_PHOTO_DATASETS = {"synthetic_fixture"}    # generated test scenes, not photographs
MAX_CONSECUTIVE_FAILURES = 3


def system_prompt() -> str:
    ph = "\n".join(f"  - {k}: {v}" for k, v in T.PHOTOGRAPHERS.items())
    fs = "\n".join(f"  - {k}: {v}" for k, v in T.FILM_STOCKS.items())
    return f"""You categorize photographs for FilmLook-ML, a specialized photography-aesthetics classification
project. Your structured labels are used to develop and evaluate a small DINOv2-small classifier that predicts
photographer-inspired styles and film-stock-inspired aesthetics.

Two views of one photo are given: a full view, and a native-resolution centre crop for grain and sharpness.

photographer_style: the closest VISUAL resemblance among these descriptions. It never means authorship. Ignore
watermarks, signatures and captions.
{ph}
  Use "none_of_the_above" for a judgeable photo that resembles none of them, and "unknown" when the image gives
  too little evidence. Do not force a match. Give up to 2 alternatives and a 0-1 confidence for the primary label.
film_aesthetic: the closest film-stock LOOK, never a claim about the medium or stock actually used.
{fs}
  Use "none_of_the_above" for a clean digital look or one that matches no stock, and "unknown" when it cannot
  be judged. Black-and-white stocks are only for monochrome images.
evidence: 1-3 short visual observations (at most 80 characters each) for each of the two labels.
attributes: choose only from the allowed values, and only what is visible. Use "indeterminate" or "unknown"
rather than guessing. For monochrome images, temperature and color_cast are "not_applicable". genre and
overall_aesthetic take 1-3 values. image_flags lists only issues that are present.
Return only the structured result."""


# ----------------------------------------------------------------------------- discovery + metadata
def discover(manifest: list[dict], registry, consented: set[str],
             quarantine: dict | None = None) -> tuple[list[dict], Counter]:
    """Eligible ORIGINAL photographs (registry-approved per record, consented for upload, not quarantined)."""
    from filmlook_ml.datasets.quarantine import is_quarantined
    eligible, skipped = [], Counter()
    for r in manifest:
        if quarantine and is_quarantined(r, quarantine):
            skipped["quarantined"] += 1
            continue
        if r.get("edit_type") != "original":
            skipped["derived_edit"] += 1
            continue
        if r["dataset_id"] in NON_PHOTO_DATASETS:
            skipped["not_a_photograph"] += 1
            continue
        ok, _ = registry.check_record(r["dataset_id"], r.get("license"))
        if not ok:
            skipped["license_not_approved"] += 1
            continue
        if r["dataset_id"] not in consented:
            skipped["no_upload_consent"] += 1
            continue
        if not Path(r["path"]).exists():
            skipped["file_missing"] += 1
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["image_id"])
    return eligible, skipped


STOCK_TAG_PATTERNS = {
    "Kodak Portra 160": r"portra ?160", "Kodak Portra 400": r"portra ?400", "Kodak Portra 800": r"portra ?800",
    "Kodak Gold 200": r"(kodak ?)?gold ?200", "Kodak Ektar 100": r"ektar", "Kodak ColorPlus 200": r"colou?rplus",
    "Kodak Ultramax 400": r"ultra ?max", "Fujifilm Pro 400H": r"pro ?400 ?h", "Fujifilm Superia X-TRA 400": r"superia",
    "Fujifilm C200": r"\bc ?200\b", "Fujifilm Velvia 50": r"velvia", "Fujifilm Provia 100F": r"provia",
    "Kodak Tri-X 400": r"tri ?-?x", "Ilford HP5 Plus 400": r"hp ?5", "Ilford FP4 Plus 125": r"fp ?4",
    "Ilford Delta 3200": r"delta ?3200", "Cinestill 800T": r"cinestill",
}


def film_stock_tags(usertags: str | None) -> list[FilmStockMetadata]:
    """Film stocks named in the uploader's Flickr tags: asserted by the uploader, not verified."""
    if not usertags:
        return []
    tags = [t.replace("+", " ").lower() for t in usertags.split(",")]
    hits = [k for k, pat in STOCK_TAG_PATTERNS.items() if any(re.search(pat, t) for t in tags)]
    return [FilmStockMetadata(value=k, source="uploader_tag", verified=False) for k in hits]


def load_uploader_tags(candidates_path: Path) -> dict[str, str]:
    """photoid -> usertags from CommonCatalog discovery metadata (CC-BY rows only)."""
    out = {}
    if candidates_path.exists():
        with open(candidates_path) as fh:
            for line in fh:
                r = json.loads(line)
                out[str(r["photoid"])] = r.get("usertags") or ""
    return out


def image_metadata(r: dict, usertags: dict[str, str]) -> ImageMetadata:
    tags = usertags.get(str(r.get("source_id")), "") if r["dataset_id"] == "commoncatalog_ccby" else ""
    return ImageMetadata(capture_medium=r.get("medium"), camera_or_scanner=(r.get("camera") or None),
                         film_stock_metadata=film_stock_tags(tags))


# ----------------------------------------------------------------------------- storage
def atomic_write_text(path: Path, text: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; annotations are never overwritten")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)          # fails if the target appeared meanwhile: no silent overwrite
            os.unlink(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "claude_exploratory"
        self.ann_dir = self.dir / "annotations"
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        tax = self.dir / f"taxonomy_{T.TAXONOMY_VERSION}.json"
        if not tax.exists():
            atomic_write_text(tax, json.dumps(T.describe(), indent=1))

    def path(self, image_id: str) -> Path:
        return self.ann_dir / f"{image_id}.json"

    def has(self, image_id: str) -> bool:
        return self.path(image_id).exists()

    def write(self, ann: Annotation) -> Path:
        p = self.path(ann.image_id)
        atomic_write_text(p, ann.model_dump_json(indent=1))
        return p

    def load_all(self) -> list[Annotation]:
        return [Annotation.model_validate_json(p.read_text()) for p in sorted(self.ann_dir.glob("*.json"))]

    def _append(self, name: str, row: dict) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
        with open(self.dir / name, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def progress(self, **row) -> None:
        self._append("progress.jsonl", row)

    def error(self, **row) -> None:
        self._append("errors.jsonl", row)


def validate_store(store: AnnotationStore) -> dict:
    ok, bad = 0, []
    for p in sorted(store.ann_dir.glob("*.json")):
        try:
            a = Annotation.model_validate_json(p.read_text())
            assert a.image_id == p.stem, "file name != image_id"
            ok += 1
        except Exception as e:  # noqa: BLE001 - report every problem
            bad.append({"file": p.name, "error": f"{type(e).__name__}: {e}"[:300]})
    return {"valid": ok, "invalid": bad}


# ----------------------------------------------------------------------------- record construction
def build_annotation(result: CategorizationResult, r: dict, *, interface: str, ccfg: dict, model_id: str,
                     features: dict | None, metadata: ImageMetadata, request: dict, root: Path) -> Annotation:
    """Validated Annotation for manifest record ``r`` with QA flags, provenance and eligibility."""
    measured = qa.measured_summary(features)
    flags = qa.check(result, measured, metadata)
    prov = make_provenance("claude", r, model_id=model_id, claude_cfg=ccfg)
    if interface == "claude_code_interactive":
        prov = prov.model_copy(update={"labeling_method": "claude_code_interactive"})
    review = qa.needs_review(flags)
    path = Path(r["path"])
    rel = str(path.relative_to(root)) if path.is_absolute() and path.is_relative_to(root) else str(path)
    return Annotation(
        image_id=r["image_id"], image_path=rel, sha256=r["sha256"],
        photographer_style=result.photographer_style, film_aesthetic=result.film_aesthetic,
        attributes=result.attributes, metadata=metadata, annotation_interface=interface, provenance=prov,
        image_source={k: r.get(k) for k in ("dataset_id", "license", "license_url", "source_url", "attribution",
                                            "author", "group_id", "family_id", "split")},
        qa=QA(flags=flags, needs_review=review, measured=measured),
        training_eligible=prov.training_permission == "permitted" and not review,
        prompt_version=PROMPT_VERSION, request=request)


# ----------------------------------------------------------------------------- API categorizer
class Categorizer:
    """API path: same client, image views, retry policy, compliance audit and budget as ClaudeTeacher."""

    def __init__(self, teacher: ClaudeTeacher):
        self.t = teacher
        self.key_suffix = f"-{PROMPT_VERSION}-{T.TAXONOMY_VERSION}"

    def categorize(self, path: Path, sha256: str, image_id: str) -> tuple[CategorizationResult | None, dict]:
        t = self.t
        key = LabelCache.key(sha256, t.prompt_version + self.key_suffix, t.model_id)
        if (hit := t.cache.get(key)) is not None:      # paid earlier but not yet stored: no second call
            return CategorizationResult.model_validate(hit), {"model": t.model_id, "cached": True}
        views, img_tokens = t._views(path)
        if not t._gate(img_tokens + t.prompt_tokens + 900, 600, consent_ok=True):
            return None, {"dry_run": True}
        before = (t.cost.calls, t.cost.input_tokens, t.cost.output_tokens, t.cost.spent)
        t0 = time.monotonic()
        content = [{"type": "text", "text": "Full view:"}, _img(views[0]),
                   {"type": "text", "text": "Native-resolution centre crop:"}, _img(views[1]),
                   {"type": "text", "text": "Categorize this photograph."}]
        res = t._call_with_retry(content, CategorizationResult, {"kind": "categorize", "image_id": image_id},
                                 system=system_prompt())
        t.cache.put(key, res.model_dump())
        return res, {"model": t.model_id, "calls": t.cost.calls - before[0],
                     "input_tokens": t.cost.input_tokens - before[1], "output_tokens": t.cost.output_tokens - before[2],
                     "cost_usd": round(t.cost.spent - before[3], 6), "elapsed_s": round(time.monotonic() - t0, 2),
                     "cached": False}


STOP_ERRORS = (PolicyStop, BudgetExceeded, NonCompliantRequest, LabelingNotAuthorized)


def _is_stop_error(e: BaseException) -> bool:
    if isinstance(e, STOP_ERRORS):
        return True
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False
    return isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                          anthropic.BadRequestError, anthropic.NotFoundError))


def run(*, eligible: list[dict], store: AnnotationStore, categorizer: Categorizer, ccfg: dict, max_images: int,
        features: dict | None = None, usertags: dict | None = None, root: Path = Path("."),
        min_interval_s: float = 1.0, sleep=time.sleep) -> dict:
    """API run: annotate up to ``max_images`` not-yet-annotated images. Returns counts and the stop reason."""
    features, usertags = features or {}, usertags or {}
    todo = [r for r in eligible if not store.has(r["image_id"])]
    counts = Counter(already_done=len(eligible) - len(todo))
    stop_reason, consecutive, last = "completed", 0, 0.0
    store.progress(event="run_start", interface="anthropic_api", eligible=len(eligible), todo=len(todo),
                   max_images=max_images, dry_run=categorizer.t.dry_run, prompt_version=PROMPT_VERSION,
                   taxonomy=T.TAXONOMY_VERSION)
    for r in todo[:max_images]:
        iid = r["image_id"]
        wait = min_interval_s - (time.monotonic() - last)
        if wait > 0 and not categorizer.t.dry_run:
            sleep(wait)
        last = time.monotonic()
        try:
            res, meta = categorizer.categorize(Path(r["path"]), r["sha256"], iid)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                stop_reason = "interrupted"
                break
            msg = f"{type(e).__name__}: {e}"[:500]
            store.error(image_id=iid, error=msg, fatal=_is_stop_error(e))
            if _is_stop_error(e):
                stop_reason = f"stopped: {msg}"
                break
            counts["failed"] += 1
            consecutive += 1
            store.progress(image_id=iid, status="failed")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                stop_reason = f"stopped after {consecutive} consecutive failures"
                break
            continue
        consecutive = 0
        if res is None:
            counts["dry_run_not_sent"] += 1
            continue
        ann = build_annotation(res, r, interface="anthropic_api", ccfg=ccfg, model_id=categorizer.t.model_id,
                               features=features.get(r["sha256"]), metadata=image_metadata(r, usertags),
                               request=meta, root=root)
        store.write(ann)
        counts["categorized"] += 1
        store.progress(image_id=iid, status="categorized", flags=ann.qa.flags)
    else:
        if len(todo) > max_images:
            stop_reason = f"per-run limit {max_images} reached; {len(todo) - max_images} images remain"
    out = {k: counts[k] for k in ("already_done", "categorized", "failed", "dry_run_not_sent")}
    store.progress(event="run_stop", reason=stop_reason, **out)
    return {"stop_reason": stop_reason, **out, "remaining": len(todo) - out["categorized"]}
