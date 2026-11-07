"""Database-driven taxonomy for the main DINOv2-small model (taxonomy/*.v1.json).

The files are extracted read-only from the FilmLook evaluation app by scripts/extract_taxonomy.py. Categories
are data, not code. A new photographer or stock means re-extracting with a new version; nothing here changes.

- Stable ids come from the source (photographer ``id``; stock ids derived from the canonical name).
- Names: en and zh for photographers. The source has no Japanese names or aliases, so none are invented.
- Actual stocks (``film_stocks``) are separate from aesthetic directions (``film_aesthetics``, the app's
  "X-like" looks), which point at a reference stock.
- ``shortlist_*`` is the first, cheap stage of hierarchical retrieval: a hard gate on monochrome/colour
  compatibility using fields the source provides. It does not rank by resemblance; later stages do.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from filmlook_ml.paths import ROOT

TAXONOMY_DIR = ROOT / "taxonomy"


_BRANDS = {"fuji": "fujifilm", "agfaphoto": "agfa", "harman": "ilford"}
_DROP = {"plus", "film", "professional"}


def _norm(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", s).casefold() if ch.isalnum())


def _stock_key(s: str) -> str:
    """Same rule as scripts/extract_taxonomy.py norm_key: brand spelling, 'plus'/'film', '-like' ignored."""
    import re
    s = re.sub(r"[-\s]like$", "", unicodedata.normalize("NFKC", s).casefold().strip())
    toks = re.findall(r"[a-z0-9]+|[^\x00-\x7f]+", s)
    if toks and toks[0] in _BRANDS:
        toks[0] = _BRANDS[toks[0]]
    return "".join(t for t in toks if t not in _DROP)


@dataclass(frozen=True)
class Taxonomy:
    version: str
    photographers: dict[str, dict]
    stocks: dict[str, dict]
    aesthetics: dict[str, dict]
    axes: dict
    source: dict

    # ---------------------------------------------------------------- lookup
    def photographer_id(self, name: str) -> str | None:
        """Resolve an id, English or Chinese name (case/punctuation-insensitive) to a stable id."""
        if name in self.photographers:
            return name
        n = _norm(name)
        for pid, p in self.photographers.items():
            if n in {_norm(p["names"]["en"]), _norm(p["names"]["zh"]), *map(_norm, p.get("aliases", []))}:
                return pid
        return None

    def stock_id(self, name: str) -> str | None:
        """Resolve a stock id, any recorded name variant, or an aesthetic label ("Portra-like") to a stock id."""
        if name in self.stocks:
            return name
        for a in self.aesthetics.values():
            if name in (a["id"], a["label"]):
                return a["reference_stock_id"]
        k = _stock_key(name)
        for sid, s in self.stocks.items():
            if k in {_stock_key(x) for x in s["names"]}:
                return sid
        return None

    def display(self, kind: str, cid: str, lang: str = "en") -> str:
        if kind == "photographer":
            names = self.photographers[cid]["names"]
            return names.get(lang) or names["en"]
        if kind == "aesthetic":
            return self.aesthetics[cid]["display_name"]
        return self.stocks[cid]["canonical_name"]

    # ---------------------------------------------------------------- stage-1 shortlist (hard gates only)
    def shortlist_photographers(self, monochrome: bool) -> list[str]:
        """Photographers whose recorded palette is compatible with the photo's palette."""
        bad = "colour" if monochrome else "monochrome"
        return sorted(pid for pid, p in self.photographers.items() if p.get("palette") != bad)

    def shortlist_aesthetics(self, monochrome: bool) -> list[str]:
        return sorted(aid for aid, a in self.aesthetics.items() if bool(a.get("monochrome")) == monochrome)

    def summary(self) -> dict:
        types: dict[str, int] = {}
        for s in self.stocks.values():
            types[s["entity_type"]] = types.get(s["entity_type"], 0) + 1
        return {"version": self.version, "photographers": len(self.photographers), "stock_entities": len(self.stocks),
                "stock_entity_types": types, "aesthetic_directions": len(self.aesthetics),
                "source_commit": self.source.get("source_git_commit")}


@lru_cache(maxsize=4)
def load_taxonomy(directory: Path | str = TAXONOMY_DIR, version: str = "v1") -> Taxonomy:
    d = Path(directory)
    ph = json.loads((d / f"photographers.{version}.json").read_text())
    st = json.loads((d / f"film_stocks.{version}.json").read_text())
    ae = json.loads((d / f"film_aesthetics.{version}.json").read_text())
    src = json.loads((d / "SOURCE.json").read_text()) if (d / "SOURCE.json").exists() else {}
    versions = {ph["taxonomy_version"], st["taxonomy_version"], ae["taxonomy_version"]}
    if len(versions) != 1:
        raise ValueError(f"taxonomy files disagree on version: {versions}")
    photographers = {p["id"]: p for p in ph["photographers"]}
    stocks = {s["id"]: s for s in st["stocks"]}
    aesthetics = {a["id"]: a for a in ae["aesthetics"]}
    if len(photographers) != len(ph["photographers"]) or len(stocks) != len(st["stocks"]):
        raise ValueError("duplicate ids in taxonomy files")
    return Taxonomy(versions.pop(), photographers, stocks, aesthetics, ph.get("axes", {}), src)


def validation_context(t: Taxonomy, photographer_ids: list[str] | None = None,
                       aesthetic_ids: list[str] | None = None) -> dict:
    """Context for CategorizationV3.model_validate(..., context=...): rankings must use these database ids
    (optionally restricted to a shortlist that was shown to the annotator)."""
    ph = photographer_ids or list(t.photographers)
    ae = aesthetic_ids or list(t.aesthetics)
    return {"photographers": ph, "film": ae, "film_monochrome": [a for a in ae if t.aesthetics[a].get("monochrome")]}


def candidates_block(t: Taxonomy, photographer_ids: list[str], aesthetic_ids: list[str], lang: str = "en") -> str:
    """Prompt text for a SHORTLIST only (stage 2 of hierarchical retrieval), using the database's own traits.
    Never the whole database: the caller shortlists first."""
    lines = ["Photographer candidates (id: visual traits from the FilmLook database):"]
    for pid in photographer_ids:
        p = t.photographers[pid]
        tr = p.get(f"traits_{lang}") or p.get("traits_en") or {}
        lines.append(f"- {pid}: " + "; ".join(f"{k}: {v}" for k, v in tr.items() if k in ("color", "composition",
                                                                                           "light", "tone", "texture")))
    lines.append("Film aesthetic candidates (id: character):")
    for aid in aesthetic_ids:
        a = t.aesthetics[aid]
        desc = a.get("character") or (f"subjective 1-5 ratings {a['ratings_1_to_5']}" if a.get("ratings_1_to_5") else "")
        lines.append(f"- {aid}: {desc}")
    return "\n".join(lines)
