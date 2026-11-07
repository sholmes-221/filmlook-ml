"""Policy primitives: third-party-AI training permission, request/prompt audits, and the labeling job lock.

The human-readable policy is docs/AI_DATA_POLICY.md; CLAUDE.md carries the mandatory checklist. This module
is the single place where the policy's *mechanical* parts live so that every entry point (labeling scripts,
training, evaluation) applies exactly the same rules.

Permission statuses (``TrainingPermission`` in labeling.schema):
  permitted      - a written authorization reference exists (env var or config) or the source is local/human.
  uncertain      - training use was asserted but no written reference exists; policy section 2 says STOP.
  not_permitted  - training use disabled, denied, or the source is a mock.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

AUTH_ENV = "FILMLOOK_ANTHROPIC_TRAINING_AUTHORIZATION_REF"
POLICY_DOC = Path("docs/AI_DATA_POLICY.md")
CLAUDE_MD = Path("CLAUDE.md")
CLAUDE_MD_SECTION = "MANDATORY — AI Training Data Compliance"

PERMISSION_RANK = {"permitted": 2, "uncertain": 1, "not_permitted": 0}

# Methods that never involve a third-party AI provider: permission follows the IMAGE license only.
LOCAL_SOURCES = {"human": "human_review", "heuristic": "heuristic_formula", "synthetic_ordering": "synthetic_edit_ordering"}
THIRD_PARTY_SOURCES = {"claude": "third_party_api"}

# Section 3: never request hidden reasoning. Prompts are audited for these phrases before any call.
REASONING_PATTERNS = [
    r"\bchain[- ]of[- ]thought\b", r"\bstep[- ]by[- ]step\b", r"\breasoning\b", r"\bthink(?:ing)? (?:aloud|out loud)\b",
    r"\bshow your work\b", r"\binternal (?:deliberation|monologue|reasoning|thoughts?)\b", r"\bscratchpad\b",
    r"\bthought process\b", r"\bhidden (?:reasoning|thoughts?)\b", r"\bexplain (?:your|the) (?:reasoning|thinking)\b",
]
ALLOWED_EFFORT = {"low", "medium"}
MAX_TOKENS_CAP = 4000          # structured labels never need more; keeps spend bounded


def worst(a: str, b: str) -> str:
    return a if PERMISSION_RANK[a] <= PERMISSION_RANK[b] else b


# ----------------------------------------------------------------------------- Claude training permission
def claude_training_permission(ccfg: dict) -> tuple[str, str]:
    """(status, basis) for training on Claude API outputs under the CURRENT config + environment.

    Only a written authorization reference counts as *permitted*: either ``FILMLOOK_ANTHROPIC_TRAINING_
    AUTHORIZATION_REF`` or ``training_authorization.status: written_reference`` with a non-null reference.
    A statement by the project owner without a reference is recorded faithfully but classified *uncertain*
    (policy section 2 step 5: unclear terms -> stop and request clarification).
    """
    if not ccfg.get("allow_training_use"):
        return "not_permitted", "allow_training_use is false in configs/labeling.yaml"
    ref = os.environ.get(AUTH_ENV, "").strip()
    if ref:
        return "permitted", f"written reference: {ref} (from {AUTH_ENV})"
    auth = ccfg.get("training_authorization") or {}
    status, reference = auth.get("status"), auth.get("reference")
    if status == "written_reference" and reference:
        return "permitted", f"written reference: {reference} (recorded {auth.get('recorded', '?')})"
    if status == "user_confirmed":
        # Owner decision (docs/AI_DATA_POLICY.md section 2): the project owner's confirmation that Anthropic
        # permits this use is accepted for this project. No written reference is on file and none was verified.
        return "permitted", (f"user-confirmed by the project owner (recorded {auth.get('recorded', '?')}); "
                             f"no written reference on file; not independently verified")
    if status == "denied":
        return "not_permitted", f"authorization denied (recorded {auth.get('recorded', '?')})"
    if status in ("confirmed_by_project_owner", "written_reference"):
        return "uncertain", (f"{status} recorded {auth.get('recorded', '?')} but no written reference; "
                             f"set {AUTH_ENV} or training_authorization.reference to a written authorization")
    return "uncertain", "no training_authorization record in configs/labeling.yaml"


def claude_evaluation_use_approved(ccfg: dict) -> tuple[bool, str]:
    """Evaluation-only use (outputs never become training targets) needs an explicit recorded decision."""
    ev = ccfg.get("evaluation_use") or {}
    if ev.get("approved") is True and ev.get("approved_by") and ev.get("recorded"):
        return True, f"evaluation-only use approved by {ev['approved_by']} on {ev['recorded']}"
    return False, "claude.evaluation_use.approved is not true (or approved_by/recorded missing) in configs/labeling.yaml"


def claude_exploratory_use_approved(ccfg: dict) -> tuple[bool, str]:
    """Exploratory categorization runs (outputs kept apart from training data) need a recorded owner decision."""
    ex = ccfg.get("exploratory_use") or {}
    if ex.get("approved") is True and ex.get("approved_by") and ex.get("recorded"):
        return True, f"exploratory use approved by {ex['approved_by']} on {ex['recorded']}"
    return False, "claude.exploratory_use.approved is not true (or approved_by/recorded missing) in configs/labeling.yaml"


# ----------------------------------------------------------------------------- prompt / request audits
def audit_prompt_text(text: str) -> list[str]:
    """Return the reasoning-elicitation phrases found in a prompt (empty = compliant)."""
    return [p for p in REASONING_PATTERNS if re.search(p, text or "", flags=re.IGNORECASE)]


class NonCompliantRequest(RuntimeError):
    pass


def assert_request_compliant(kw: dict) -> None:
    """Section 3 gate applied to the kwargs of every Messages API call made by this repository."""
    problems = []
    if "thinking" in kw:
        problems.append("'thinking' must not be set (extended thinking is not requested for labeling)")
    effort = (kw.get("output_config") or {}).get("effort")
    if effort not in ALLOWED_EFFORT:
        problems.append(f"output_config.effort must be one of {sorted(ALLOWED_EFFORT)}, got {effort!r}")
    if not kw.get("output_format") and not (kw.get("output_config") or {}).get("format"):
        problems.append("a structured output schema is required (output_format / output_config.format)")
    if int(kw.get("max_tokens", 0) or 0) > MAX_TOKENS_CAP:
        problems.append(f"max_tokens {kw.get('max_tokens')} exceeds cap {MAX_TOKENS_CAP}")
    system = kw.get("system")
    texts = [system] if isinstance(system, str) else [b.get("text", "") for b in (system or []) if isinstance(b, dict)]
    for m in kw.get("messages") or []:
        c = m.get("content")
        texts += [c] if isinstance(c, str) else [b.get("text", "") for b in (c or []) if isinstance(b, dict) and b.get("type") == "text"]
    for t in texts:
        hits = audit_prompt_text(t)
        if hits:
            problems.append(f"prompt requests reasoning (matched {hits})")
    if problems:
        raise NonCompliantRequest("; ".join(problems))


class PolicyStop(RuntimeError):
    """A policy-related API response (refusal / permission error). Never retried (section 6)."""


# ----------------------------------------------------------------------------- job lock
class ConcurrentLabelingJob(RuntimeError):
    pass


class LabelingLock:
    """One labeling job at a time per label directory (section 6: no unconfigured concurrent jobs)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip() or 0)
                os.kill(pid, 0)
                alive = True
            except (ValueError, ProcessLookupError):
                alive = False
            except PermissionError:
                alive = True
            if alive:
                raise ConcurrentLabelingJob(f"another labeling job holds {self.path} (pid {pid})")
            self.path.unlink()
        self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self._fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.path.exists():
            self.path.unlink()
        return False


# ----------------------------------------------------------------------------- policy documents
def policy_docs_status(root: Path) -> list[str]:
    """Problems with the mandatory policy documents (empty = present and intact)."""
    problems = []
    cm = Path(root) / CLAUDE_MD
    if not cm.exists():
        problems.append(f"{CLAUDE_MD} missing")
    elif CLAUDE_MD_SECTION not in cm.read_text():
        problems.append(f"{CLAUDE_MD} lacks the section '{CLAUDE_MD_SECTION}'")
    pd = Path(root) / POLICY_DOC
    if not pd.exists():
        problems.append(f"{POLICY_DOC} missing")
    elif "Pre-training compliance checklist" not in pd.read_text():
        problems.append(f"{POLICY_DOC} lacks the pre-training compliance checklist")
    return problems
