"""Pseudo-label teachers.

* HeuristicTeacher - transparent formulas over measured statistics. Free, local, license-clean.
                     Its thresholds are hand-set, NOT validated against human judgement.
* MockTeacher      - deterministic fake output for tests / dry runs. Never trained on.
* ClaudeTeacher    - Anthropic API, OFFLINE labeling only (never used at inference).

Training authorization: governed by docs/AI_DATA_POLICY.md. Only a WRITTEN authorization reference
(FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF or training_authorization.reference) makes Claude outputs
*permitted* training targets. The project owner's 2026-11-06 statement (no reference) is recorded in
configs/labeling.yaml and classified *uncertain*: labels are stored with usable_for_training=false and the
training preflight refuses to use them. Permission to train on Claude outputs does NOT grant rights to the
source photographs: only images from registry-approved datasets are ever uploaded. Requests never ask for
reasoning: structured output, low effort, bounded max_tokens (compliance.policy.assert_request_compliant).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from filmlook_ml.compliance.policy import (  # noqa: F401  (AUTH_ENV re-exported for callers)
    AUTH_ENV,
    PolicyStop,
    assert_request_compliant,
    claude_training_permission,
)
from filmlook_ml.constants import ATTRIBUTES, COLOR_ATTRIBUTES, RUBRIC
from filmlook_ml.labeling.features import center_crop_native, compute_features
from filmlook_ml.labeling.schema import (
    ClaudeAssessment,
    ClaudePairAssessment,
    StyleVector,
    TeacherOutput,
    to_teacher_output,
)

SCORE_WEIGHTS = {"highlight_rolloff": 0.2, "color_response": 0.15, "microcontrast": 0.15,
                 "grain_naturalness": 0.15, "tonal_depth": 0.15, "film_coherence": 0.2}


def _sig(x: float, c: float, s: float) -> float:
    return float(1 / (1 + np.exp(-(x - c) / s)))


def heuristic_attributes(f: dict[str, float]) -> dict[str, float | None]:
    """Hand-set mapping from measurements to the v1 rubric (higher = more film-like/natural).

    Does NOT reward warmth, faded blacks or grain *amount*: grain is judged on structure (spatial
    correlation, luminance dependence, chroma), colour on clipping/over-saturation, tone on clipping,
    combing, HDR local contrast, crushed or milky blacks.
    """
    hr = 1 - 0.3 * _sig(f["clip_high"], 0.012, 0.005) - 0.35 * _sig(f["highlight_shoulder"], 0.16, 0.04) \
        - 0.2 * _sig(f["hist_gap_ratio"], 0.25, 0.06) - 0.15 * _sig(f["local_contrast_ratio"], 0.18, 0.015)
    if f["is_mono"]:
        cr = None
    else:
        cr = 1 - 0.45 * _sig(f["sat_clip"], 0.03, 0.012) - 0.35 * _sig(f["sat_p95"], 0.88, 0.04) \
            - 0.2 * _sig(f["colorfulness"], 0.65, 0.08)
    mc = 1 - 0.7 * _sig(f["overshoot"], 4.0, 1.5) - 0.3 * _sig(f["hf_ratio"], 0.65, 0.06)
    if f["noise_std"] < 0.005:
        gn = 0.5  # essentially grainless: neither natural grain nor fake grain
    else:
        clump = _sig(f["noise_autocorr1"], 0.15, 0.08)          # film grain clumps
        chroma_pen = _sig(f["chroma_noise_ratio"], 0.9, 0.2)    # strong colour speckle = digital
        shadow_pen = _sig(f["noise_shadow_ratio"], 1.5, 0.15)   # sensor noise lives in shadows
        gn = 0.25 + 0.55 * clump - 0.2 * chroma_pen - 0.15 * shadow_pen + 0.15 * _sig(f["noise_mid_ratio"], 1.0, 0.15)
    td = 1 - 0.35 * _sig(f["clip_low"], 0.02, 0.008) - 0.25 * _sig(f["p01"], 0.14, 0.03) \
        - 0.2 * _sig(f["local_contrast_ratio"], 0.18, 0.015) - 0.2 * _sig(f["hist_gap_ratio"], 0.25, 0.06)
    vals = {"highlight_rolloff": hr, "color_response": cr, "microcontrast": mc, "grain_naturalness": gn,
            "tonal_depth": td}
    vals = {k: (None if v is None else float(np.clip(v, 0, 1))) for k, v in vals.items()}
    vals["film_coherence"] = _weighted({k: v for k, v in vals.items()},
                                       {"highlight_rolloff": 0.25, "color_response": 0.15, "microcontrast": 0.2,
                                        "grain_naturalness": 0.2, "tonal_depth": 0.2})
    return vals


def _weighted(vals: dict, w: dict) -> float:
    num = sum(w[k] * v for k, v in vals.items() if v is not None and k in w)
    den = sum(w[k] for k, v in vals.items() if v is not None and k in w)
    return float(np.clip(num / max(den, 1e-6), 0, 1))


def film_score_from_attributes(a: dict[str, float | None]) -> float:
    """Weighted mean of available attributes (colour excluded for monochrome), 0-100."""
    return 100 * _weighted(a, SCORE_WEIGHTS)


def measured_style(f: dict) -> StyleVector:
    from filmlook_infer.style import style_tags
    t = set(style_tags(f))
    mono = "monochrome" in t
    return StyleVector(
        temperature="not_applicable" if mono else next(x for x in ("warm", "cool", "neutral") if x in t),
        saturation="not_applicable" if mono else ("muted" if "muted" in t else "vivid" if "vivid" in t else "natural"),
        sharpness="soft" if "soft" in t else "crisp" if "crisp" in t else "natural",
        key="high_key" if "high_key" in t else "low_key" if "low_key" in t else "balanced",
        grain=next(x for x in ("no_visible_grain", "fine_grain", "coarse_grain") if x in t))


class Teacher:
    name = "base"
    model_id = "none"

    def label(self, image_path: Path, features: dict | None = None) -> TeacherOutput:  # pragma: no cover
        raise NotImplementedError


class HeuristicTeacher(Teacher):
    name = "heuristic"
    model_id = "heuristic-v2"

    def label(self, image_path: Path, features: dict | None = None) -> TeacherOutput:
        if features is None:
            with Image.open(image_path) as im:
                features = compute_features(im)
        attrs = heuristic_attributes(features)
        return TeacherOutput(
            film_score=round(film_score_from_attributes(attrs), 2),
            attributes={k: (None if v is None else round(v, 4)) for k, v in attrs.items()},
            style=measured_style(features), monochrome=bool(features["is_mono"]),
            uncertainty="high", label_confidence=0.3,
            visual_evidence="Computed from measured clipping, histogram combing, local-contrast ratio, "
                            "sharpening overshoot, black level and grain-residual statistics.")


class MockTeacher(Teacher):
    """Deterministic, hash-seeded output. Never used as a real training target."""
    name = "mock"
    model_id = "mock-v2"

    def label(self, image_path: Path, features: dict | None = None) -> TeacherOutput:
        seed = int(hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        attrs = {k: round(rng.random(), 3) for k in ATTRIBUTES}
        return TeacherOutput(film_score=round(film_score_from_attributes(attrs), 2), attributes=attrs,
                             visual_evidence="mock", label_confidence=0.0)


# ----------------------------------------------------------------------------- Claude
class BudgetExceeded(RuntimeError):
    pass


class LabelingNotAuthorized(RuntimeError):
    pass


@dataclass
class CostTracker:
    """Cumulative across runs: initialised from the usage log, so the cap is a project-level cap."""
    max_budget_usd: float
    price_in: float
    price_out: float
    usage_log: Path | None = None
    spent_before: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __post_init__(self):
        if self.usage_log and Path(self.usage_log).exists():
            for line in Path(self.usage_log).read_text().splitlines():
                if line.strip():
                    self.spent_before += json.loads(line).get("usd", 0.0)

    @property
    def spent_this_run(self) -> float:
        return self.input_tokens / 1e6 * self.price_in + self.output_tokens / 1e6 * self.price_out

    @property
    def spent(self) -> float:
        return self.spent_before + self.spent_this_run

    def estimate_call(self, est_in: int, est_out: int) -> float:
        return est_in / 1e6 * self.price_in + est_out / 1e6 * self.price_out

    def check(self, est_in: int, est_out: int) -> None:
        if self.spent + self.estimate_call(est_in, est_out) > self.max_budget_usd:
            raise BudgetExceeded(f"budget ${self.max_budget_usd:.2f} would be exceeded "
                                 f"(spent ${self.spent:.4f} incl. previous runs)")

    def record(self, tin: int, tout: int, meta: dict) -> dict:
        self.input_tokens += tin
        self.output_tokens += tout
        self.calls += 1
        usd = self.estimate_call(tin, tout)
        row = {"ts": datetime.now(timezone.utc).isoformat(), "input_tokens": tin, "output_tokens": tout,
               "usd": round(usd, 6), **meta}
        if self.usage_log:
            Path(self.usage_log).parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        return row


class LabelCache:
    """JSONL cache keyed by (content hash(es), prompt_version, teacher_model). Resume = cache hits."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._d: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._d[r["key"]] = r["output"]

    @staticmethod
    def key(sha: str, prompt_version: str, model: str) -> str:
        return f"{sha}|{prompt_version}|{model}"

    def get(self, k: str) -> dict | None:
        return self._d.get(k)

    def put(self, k: str, output: dict) -> None:
        self._d[k] = output
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"key": k, "output": output}) + "\n")

    def __len__(self) -> int:
        return len(self._d)


def _rubric_text() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())


SYSTEM_V2 = f"""You label photographs for training a small film-aesthetics model. You judge the VISIBLE
RENDERING of the image (tone curve, colour, acutance, texture), not its subject, era, or artistic merit.

You receive the full image (downscaled) and a native-resolution centre crop for judging grain/texture.

Scoring rubric (for every attribute, higher = more convincingly film-like / natural):
{_rubric_text()}

Rules:
- Do NOT reward by themselves: heavy grain, warm casts, faded/lifted blacks, artificial halation, low
  contrast, blur, or vintage-looking subjects. These are trivially added by filters.
- A clean, subtle film look can deserve a higher score than an exaggerated film filter.
- A sharp, clean digital photo is not "bad"; it is simply less film-like. Score what you see.
- Do not guess the camera, film stock, or photographer.
- If the image is monochrome: monochrome=true, attributes.color_response=null,
  style.temperature and style.saturation = "not_applicable" (even for toned prints).
- style.mood: 0-2 tags only when clearly justified; an empty list is normal.
- uncertainty: "high" when the image is small, heavily compressed, a screenshot/scan of a print, or the
  cues conflict; list reasons in ambiguity_reasons.
- strengths / weaknesses: short concrete phrases about rendering (e.g. "clipped sky highlights",
  "fine clumped grain in midtones"). No subject descriptions.
- label_confidence: your own 0-1 confidence. Be honest; it is not treated as a probability."""

PAIR_V2 = """Image A and image B are shown (each as full view + native-resolution crop). For each criterion,
say which looks MORE convincingly film-like / natural under the rubric, or "tie" if the difference is
not clearly visible. Do not assume an edited-looking version is better or worse; judge what you see."""


def training_authorization(cfg: dict) -> str | None:
    """Legacy free-text provenance of the recorded authorization (or None if none is recorded).

    NOT a permission gate: an owner statement without a reference yields a string here but is classified
    *uncertain* by ``claude_training_permission`` and never enters training. Kept for the ``authorization``
    field of stored labels."""
    if not cfg.get("allow_training_use"):
        return None
    ref = os.environ.get(AUTH_ENV, "").strip()
    auth = cfg.get("training_authorization") or {}
    if ref:
        return f"written reference: {ref}"
    if auth.get("status") in ("confirmed_by_project_owner", "written_reference"):
        return (f"{auth['status']} (recorded {auth.get('recorded', '?')}; "
                f"reference: {auth.get('reference') or 'none supplied'})")
    return None


def api_key_status(root: Path | None = None) -> str:
    """'env' | '.env' | 'missing'. Never returns or logs the key itself."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "env"
    p = Path(root or ".") / ".env"
    if p.exists() and any(line.split("=", 1)[0].strip() == "ANTHROPIC_API_KEY" and line.split("=", 1)[1].strip()
                          for line in p.read_text().splitlines() if "=" in line and not line.lstrip().startswith("#")):
        return ".env"
    return "missing"


def _key_from_dotenv(root: Path | None = None) -> str | None:
    p = Path(root or ".") / ".env"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                return v.strip().strip('"').strip("'")
    return None


class ClaudeTeacher(Teacher):
    name = "claude"

    def __init__(self, cfg: dict, cache_path: Path, prompt_version: str = "v2",
                 dry_run: bool = True, confirm: bool = False, client=None, usage_log: Path | None = None):
        self.cfg = cfg
        self.model_id = cfg.get("model", "claude-opus-5-5")
        self.prompt_version = prompt_version
        self.dry_run = dry_run
        self.confirm = confirm
        self.cache = LabelCache(cache_path)
        self.cost = CostTracker(cfg.get("max_budget_usd", 5.0), cfg.get("price_input_per_mtok", 4.0),
                                cfg.get("price_output_per_mtok", 20.0), usage_log)
        self._client = client
        self.max_side = cfg.get("max_image_side", 1024)
        self.crop = cfg.get("texture_crop", 384)
        self.est_out = cfg.get("est_output_tokens", 700)
        self.prompt_tokens = cfg.get("est_prompt_tokens", 1300)

    @staticmethod
    def training_use_allowed(cfg: dict) -> bool:
        """True only with a written authorization reference (docs/AI_DATA_POLICY.md section 2)."""
        return claude_training_permission(cfg)[0] == "permitted"

    def _client_or_create(self):
        if self._client is None:
            import anthropic  # local import: not needed in deployment
            key = os.environ.get("ANTHROPIC_API_KEY") or _key_from_dotenv()
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        return self._client

    def _views(self, image_path: Path) -> tuple[list[str], int]:
        """Full view (<= max_side) + native-resolution centre crop. Returns base64 JPEGs, est tokens."""
        out, tokens = [], 0
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            crop = center_crop_native(im, self.crop)
            full = im.copy()
            full.thumbnail((self.max_side, self.max_side))
            for v in (full, crop):
                buf = io.BytesIO()
                v.save(buf, "JPEG", quality=92, subsampling=0)
                out.append(base64.standard_b64encode(buf.getvalue()).decode())
                tokens += int(v.width * v.height / 750) + 1
        return out, tokens

    def estimate(self, n_images: int, n_pairs: int = 0) -> dict:
        w, h = self.max_side, int(self.max_side * 2 / 3)
        img = int(w * h / 750) + int(self.crop * self.crop / 750)
        per_img = self.cost.estimate_call(img + self.prompt_tokens, self.est_out)
        per_pair = self.cost.estimate_call(2 * img + self.prompt_tokens + 100, 350)
        return {"per_image_usd": round(per_img, 4), "per_pair_usd": round(per_pair, 4),
                "total_usd": round(per_img * n_images + per_pair * n_pairs, 2),
                "assumptions": {"image_tokens": img, "prompt_tokens": self.prompt_tokens,
                                "output_tokens_image": self.est_out, "output_tokens_pair": 350,
                                "prices_per_mtok": [self.cost.price_in, self.cost.price_out]}}

    def _gate(self, est_in: int, est_out: int, consent_ok: bool) -> bool:
        """Return True if a paid call may proceed; False for dry-run. Raises if not allowed."""
        if not consent_ok:
            raise LabelingNotAuthorized("image lacks rights/consent for upload to a third-party API")
        self.cost.check(est_in, est_out)
        if self.dry_run:
            return False
        if not self.confirm:
            raise LabelingNotAuthorized("paid labeling requires explicit --confirm-paid")
        return True

    def label(self, image_path: Path, features: dict | None = None, sha256: str | None = None,
              image_id: str = "", consent_ok: bool = False) -> TeacherOutput | None:
        sha = sha256 or hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        key = LabelCache.key(sha, self.prompt_version, self.model_id)
        if not consent_ok:
            raise LabelingNotAuthorized("image lacks rights/consent for upload to a third-party API")
        if (hit := self.cache.get(key)) is not None:
            return TeacherOutput.model_validate(hit)
        views, img_tokens = self._views(image_path)
        if not self._gate(img_tokens + self.prompt_tokens, self.est_out, consent_ok):
            return None
        content = [{"type": "text", "text": "Full view:"}, _img(views[0]),
                   {"type": "text", "text": "Native-resolution centre crop:"}, _img(views[1]),
                   {"type": "text", "text": "Assess this image."}]
        parsed = self._call_with_retry(content, ClaudeAssessment, {"kind": "image", "image_id": image_id})
        out = to_teacher_output(parsed)
        self.cache.put(key, out.model_dump())
        return out

    def compare(self, path_a: Path, path_b: Path, sha_a: str, sha_b: str, ids: tuple[str, str] = ("", ""),
                consent_ok: bool = False) -> ClaudePairAssessment | None:
        key = LabelCache.key(f"{sha_a}+{sha_b}", self.prompt_version + "-pair", self.model_id)
        if not consent_ok:
            raise LabelingNotAuthorized("image lacks rights/consent for upload to a third-party API")
        if (hit := self.cache.get(key)) is not None:
            return ClaudePairAssessment.model_validate(hit)
        va, ta = self._views(path_a)
        vb, tb = self._views(path_b)
        if not self._gate(ta + tb + self.prompt_tokens + 100, 350, consent_ok):
            return None
        content = [{"type": "text", "text": "Image A (full view, then crop):"}, _img(va[0]), _img(va[1]),
                   {"type": "text", "text": "Image B (full view, then crop):"}, _img(vb[0]), _img(vb[1]),
                   {"type": "text", "text": PAIR_V2}]
        out = self._call_with_retry(content, ClaudePairAssessment, {"kind": "pair", "image_id": f"{ids[0]}|{ids[1]}"})
        self.cache.put(key, out.model_dump())
        return out

    def style(self, image_path: Path, sha256: str, image_id: str = "", consent_ok: bool = False):
        """Offline photographer-STYLE resemblance judgement (never authorship). Cached + budgeted like label()."""
        from filmlook_ml.labeling.schema import ClaudeStyleAssessment
        key = LabelCache.key(sha256, self.prompt_version + "-style", self.model_id)
        if not consent_ok:
            raise LabelingNotAuthorized("image lacks rights/consent for upload to a third-party API")
        if (hit := self.cache.get(key)) is not None:
            return ClaudeStyleAssessment.model_validate(hit)
        views, img_tokens = self._views(image_path)
        if not self._gate(img_tokens + self.prompt_tokens + 400, 300, consent_ok):
            return None
        content = [{"type": "text", "text": "Full view:"}, _img(views[0]),
                   {"type": "text", "text": "Native-resolution centre crop:"}, _img(views[1]),
                   {"type": "text", "text": "Which listed styles does this photo's rendering resemble?"}]
        out = self._call_with_retry(content, ClaudeStyleAssessment, {"kind": "style", "image_id": image_id},
                                    system=style_system_prompt())
        self.cache.put(key, out.model_dump())
        return out

    def _call_with_retry(self, content: list, schema, meta: dict, max_attempts: int = 5, system: str | None = None):
        import anthropic

        client = self._client_or_create()
        delay = 2.0
        kw = dict(
            model=self.model_id,
            max_tokens=self.cfg.get("max_tokens", 4000),
            system=[{"type": "text", "text": system or SYSTEM_V2, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.cfg.get("effort", "low")},   # no `thinking`: never requested
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
        assert_request_compliant(kw)   # section 3: structured, low effort, bounded, no reasoning elicitation
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.messages.parse(**kw)
                u = resp.usage
                tin = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
                    + (getattr(u, "cache_read_input_tokens", 0) or 0)
                self.cost.record(tin, u.output_tokens, {**meta, "model": self.model_id,
                                                        "prompt_version": self.prompt_version,
                                                        "stop_reason": resp.stop_reason, "attempt": attempt})
                if resp.stop_reason == "refusal":   # policy-related: stop immediately, never retry (section 6)
                    raise PolicyStop(f"API refused the request (stop_reason=refusal, {meta}); labeling stopped")
                if resp.parsed_output is None:
                    raise ValueError(f"no parsed output (stop_reason={resp.stop_reason})")
                return resp.parsed_output
            except (anthropic.BadRequestError, anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError, anthropic.NotFoundError):
                raise  # not retryable
            except anthropic.RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", delay))
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError):
                wait = delay
            if attempt == max_attempts:
                raise RuntimeError(f"Claude labeling failed after {max_attempts} attempts")
            self.cost.check(0, 0)
            time.sleep(wait + random.random())
            delay = min(delay * 2, 60)
        raise AssertionError("unreachable")


def style_system_prompt() -> str:
    from filmlook_ml.style.profiles import PROFILES
    lines = "\n".join(f"- {pid}: {p['basis']['en']}" for pid, p in PROFILES.items())
    return f"""You judge which photographic STYLES a photo's visible rendering resembles, for training a small
style-similarity model. Each style is defined only by these measurable characteristics:
{lines}
Rules:
- Judge colour, tone, contrast, texture, light and composition that you can SEE. Ignore subject matter, era,
  location and any text or watermark.
- This is resemblance, never authorship: do not guess who took the photo.
- Most ordinary photos resemble none of these styles: set no_reliable_match=true and return an empty or weak
  candidate list in that case. Do not force a match.
- Monochrome photos cannot resemble colour-defined styles, and vice versa.
- At most 3 candidates, best first, resemblance in [0,1], each with one short evidence phrase."""


def _img(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


# ----------------------------------------------------------------------------- quality checks
def check_label(out: TeacherOutput, f: dict | None) -> list[str]:
    """Automatic consistency checks. Flags route labels to human review; some exclude from training."""
    flags = []
    vals = [v for v in out.attributes.values() if v is not None]
    if vals and abs(out.film_score / 100 - film_score_from_attributes(out.attributes) / 100) > 0.25:
        flags.append("score_attr_mismatch")
    if out.label_confidence < 0.5 or out.uncertainty == "high":
        flags.append("low_confidence")
    if f is not None:
        if out.monochrome is not None and ((f["sat_p95"] < 0.05 and not out.monochrome)
                                           or (f["sat_p95"] > 0.2 and out.monochrome)):
            flags.append("mono_mismatch")
        if out.style is not None:
            if out.style.grain == "coarse_grain" and f["noise_std"] < 0.004:
                flags.append("grain_mismatch")
            if out.style.grain == "no_visible_grain" and f["noise_std"] > 0.035:
                flags.append("grain_mismatch")
            if (out.style.temperature == "warm" and f["warmth"] < -0.04) or \
               (out.style.temperature == "cool" and f["warmth"] > 0.1):
                flags.append("temperature_mismatch")
    return flags


EXCLUDING_FLAGS = {"score_attr_mismatch", "mono_mismatch"}


def make_teacher(name: str, labeling_cfg: dict, cache_dir: Path, **kw) -> Teacher:
    if name == "heuristic":
        return HeuristicTeacher()
    if name == "mock":
        return MockTeacher()
    if name == "claude":
        return ClaudeTeacher(labeling_cfg["claude"], cache_dir / "claude_cache.jsonl",
                             labeling_cfg.get("prompt_version", "v2"),
                             usage_log=cache_dir / "claude_usage.jsonl", **kw)
    raise ValueError(name)
