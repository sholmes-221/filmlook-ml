"""Domain vocabulary. Single source of truth lives in the runtime package (filmlook_infer.vocab).

v1 removed photographer-name matching and film-stock similarity: no licensed reference data or
reliable evaluation exists for them (see REPORT.md).
"""
from __future__ import annotations

from filmlook_infer.vocab import (  # noqa: F401
    ATTRIBUTE_NAMES,
    ATTRIBUTES,
    COLOR_ATTRIBUTES,
    MODEL_VERSION,
    MOOD_TAGS,
    RUBRIC,
    SCHEMA_VERSION,
    STYLE_GROUPS,
    STYLE_TAGS,
)
