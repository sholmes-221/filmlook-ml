"""AI training-data compliance: permission classification, provenance, preflight checks.

See docs/AI_DATA_POLICY.md (policy) and CLAUDE.md (mandatory checklist).
Import submodules explicitly (``filmlook_ml.compliance.preflight`` / ``.provenance``); this package init
stays import-light because ``labeling.teachers`` depends on ``compliance.policy``.
"""
from filmlook_ml.compliance.policy import (  # noqa: F401
    AUTH_ENV,
    ConcurrentLabelingJob,
    LabelingLock,
    NonCompliantRequest,
    PolicyStop,
    assert_request_compliant,
    audit_prompt_text,
    claude_evaluation_use_approved,
    claude_exploratory_use_approved,
    claude_training_permission,
    policy_docs_status,
)
