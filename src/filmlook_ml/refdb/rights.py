"""Rights classes for the photographer reference database and the rule that decides what may be processed.

Classes (the brief's five): ``public_domain``, ``cc0``, ``licensed`` (an open licence with conditions, e.g. CC BY or
CC BY-SA), ``unverified`` (no machine-readable statement, or one this module does not recognise) and ``restricted``
(copyrighted without an open licence, non-free, or excluded by project policy such as Magnum agency images).

Only ``public_domain`` and ``cc0`` are eligible for the image-processing pipeline in the pilot. ``licensed`` works are
kept as metadata only until the owner decides that derived embeddings under attribution/share-alike terms are
acceptable; storing an embedding does not by itself settle that question.

What "verified" means here, stated plainly: the rights status is the source's own machine-readable statement (a
Wikimedia Commons licence template reviewed by that community, or a museum's ``is_public_domain`` / Open Access flag),
recorded with the evidence URL and retrieval time. It is not an independent legal opinion.
"""
from __future__ import annotations

import re

CLASSES = ("public_domain", "cc0", "licensed", "unverified", "restricted")
ELIGIBLE = frozenset({"public_domain", "cc0"})

_PD = re.compile(r"^(pd|public domain)\b|^pd[-_ ]|\bpublic[ _-]domain\b", re.I)
_CC0 = re.compile(r"^(cc0|cc-zero|cc zero)\b|creativecommons\.org/publicdomain/zero", re.I)
_OPEN = re.compile(r"^cc[- ]by(?:[- ]sa)?[- ]?\d|creativecommons\.org/licenses/by(?:-sa)?/", re.I)
_RESTRICTED = re.compile(r"\b(non[- ]?free|fair use|all rights reserved|nc|nd)\b", re.I)


def classify_commons(license_code: str, short_name: str, copyrighted: str, usage_terms: str = "") -> str:
    """Commons ``extmetadata`` fields License, LicenseShortName, Copyrighted, UsageTerms -> rights class."""
    code, short = (license_code or "").strip(), (short_name or "").strip()
    text = " ".join((code, short, usage_terms or ""))
    if _CC0.search(code) or _CC0.search(short):
        return "cc0"
    if (_PD.search(code) or _PD.search(short)) and (copyrighted or "").strip().lower() in ("false", ""):
        return "public_domain"
    if _OPEN.search(code) or _OPEN.search(short):
        return "restricted" if re.search(r"\b(nc|nd)\b", code, re.I) else "licensed"
    if _RESTRICTED.search(text):
        return "restricted"
    return "unverified"


def classify_museum(is_public_domain: bool | None) -> str:
    """AIC ``is_public_domain`` / Met ``isPublicDomain``: True means the museum releases the image under CC0."""
    if is_public_domain is True:
        return "cc0"
    if is_public_domain is False:
        return "restricted"
    return "unverified"


def eligible(rights_class: str) -> bool:
    return rights_class in ELIGIBLE
