"""Build the Photographer Visual DNA knowledge graph (SQLite) from the reviewed JSON files in knowledge/visual_dna/.

    python -m filmlook_ml.visualdna.build            # validate, write data/visual_dna/visual_dna.sqlite and the reports

The JSON files are the source of truth (reviewed, committed, one per photographer). The database is derived and can
be rebuilt at any time. Every row that states a fact carries its evidence: ``evidence(entity, entity_id, source_id)``
links it to a source with URL, kind and retrieval date. The schema is additive: new kinds of entity get new tables,
and ``meta.schema_version`` records the layout.

Validation is strict and fails the build: unknown source ids, tags outside the vocabulary, unknown motif levels, a
claim without a source, a ``signature`` motif with fewer than two distinct sources. The completeness report is
separate: a profile that misses a required area is built but marked ``incomplete`` and must not drive a
high-confidence recommendation.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from filmlook_ml import paths as P

KNOWLEDGE = P.ROOT / "knowledge" / "visual_dna"
DB = P.DATA / "visual_dna" / "visual_dna.sqlite"
REPORT = P.REPORTS / "visual_dna"
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE photographers (id TEXT PRIMARY KEY, website_id TEXT, name_en TEXT, name_zh TEXT, name_ja TEXT, born TEXT, died TEXT,
    image_rights TEXT, image_rights_note TEXT, profile_status TEXT, completeness_json TEXT, uncertain_json TEXT);
CREATE TABLE sources (photographer_id TEXT, id TEXT, url TEXT, kind TEXT, retrieved TEXT, note TEXT, PRIMARY KEY (photographer_id, id));
CREATE TABLE photobooks (photographer_id TEXT, id TEXT, title_original TEXT, title_romaji TEXT, title_en TEXT, title_zh TEXT, title_ja TEXT,
    year INTEGER, publisher TEXT, isbn TEXT, isbn_src TEXT, editions_json TEXT, themes_json TEXT, subjects_json TEXT, visual_json TEXT, career TEXT,
    PRIMARY KEY (photographer_id, id));
CREATE TABLE series (photographer_id TEXT, id TEXT, title TEXT, title_given INTEGER, start_year INTEGER, end_year INTEGER, period_note TEXT,
    subjects_json TEXT, visual_json TEXT, context TEXT, refdb TEXT, PRIMARY KEY (photographer_id, id));
CREATE TABLE series_photobooks (photographer_id TEXT, series_id TEXT, photobook_id TEXT);
CREATE TABLE photographs (photographer_id TEXT, id TEXT, title TEXT, title_given INTEGER, year TEXT, series_id TEXT, photobook_id TEXT,
    location TEXT, subjects_json TEXT, composition TEXT, why TEXT, rights TEXT, refdb_work_id TEXT, PRIMARY KEY (photographer_id, id));
CREATE TABLE motifs (photographer_id TEXT, idx INTEGER, element TEXT, level TEXT, context TEXT, frequency_evidence TEXT, distinctiveness TEXT,
    confidence TEXT, series_json TEXT, representative_json TEXT, PRIMARY KEY (photographer_id, idx));
CREATE TABLE motif_tags (photographer_id TEXT, motif_idx INTEGER, tag TEXT);
CREATE TABLE style_traits (photographer_id TEXT, idx INTEGER, dimension TEXT, value TEXT, kind TEXT, scope TEXT, tags_json TEXT,
    PRIMARY KEY (photographer_id, idx));
CREATE TABLE style_tags (photographer_id TEXT, trait_idx INTEGER, tag TEXT, scope TEXT);
CREATE TABLE periods (photographer_id TEXT, idx INTEGER, label TEXT, start_year INTEGER, end_year INTEGER);
CREATE TABLE evidence (photographer_id TEXT, entity TEXT, entity_id TEXT, source_id TEXT);
CREATE TABLE vocabulary (tag TEXT PRIMARY KEY, kind TEXT);
CREATE INDEX idx_motif_tags ON motif_tags (tag);
CREATE INDEX idx_evidence ON evidence (photographer_id, entity, entity_id);
"""

LEVELS = ("signature", "repeated", "series_specific", "tendency", "weak")
STYLE_KINDS = ("observable", "interpretive")


class InvalidProfile(ValueError):
    pass


def load_vocabulary() -> dict:
    return json.loads((KNOWLEDGE / "_vocabulary.json").read_text(encoding="utf-8"))


def load_profiles() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(KNOWLEDGE.glob("*.json")) if not p.name.startswith("_")]


def validate(profile: dict, vocab: dict) -> list[str]:
    """Hard errors (the build fails on any)."""
    errors, pid = [], profile.get("id", "?")
    sources = set(profile.get("sources", {}))
    tags = set(vocab["tags"])

    def need_src(where: str, item: dict) -> None:
        src = item.get("src") or []
        if not src:
            errors.append(f"{pid}: {where} has no source")
        for s in src:
            if s not in sources:
                errors.append(f"{pid}: {where} cites unknown source {s}")

    for s in profile.get("life_src", []):
        if s not in sources:
            errors.append(f"{pid}: life cites unknown source {s}")
    for key in ("photobooks", "series", "photographs", "style", "periods"):
        for i, item in enumerate(profile.get(key, [])):
            need_src(f"{key}[{i}]", item)
            for t in item.get("tags", []):
                if t not in tags:
                    errors.append(f"{pid}: {key}[{i}] tag {t!r} not in vocabulary")
            if key == "style" and item.get("kind") not in STYLE_KINDS:
                errors.append(f"{pid}: style[{i}] kind must be observable or interpretive")
    series_ids = {s["id"] for s in profile.get("series", [])}
    photo_ids = {p["id"] for p in profile.get("photographs", [])}
    for i, m in enumerate(profile.get("motifs", [])):
        need_src(f"motifs[{i}]", m)
        if m.get("level") not in LEVELS:
            errors.append(f"{pid}: motifs[{i}] level {m.get('level')!r}")
        if m.get("level") == "signature" and len(set(m.get("src", []))) < 2:
            errors.append(f"{pid}: motifs[{i}] '{m['element']}' is signature with fewer than two sources")
        for t in m.get("tags", []):
            if t not in tags:
                errors.append(f"{pid}: motifs[{i}] tag {t!r} not in vocabulary")
        for s in m.get("series", []):
            if s not in series_ids:
                errors.append(f"{pid}: motifs[{i}] unknown series {s}")
        for r in m.get("representative", []):
            if r not in photo_ids:
                errors.append(f"{pid}: motifs[{i}] unknown photograph {r}")
    for p in profile.get("photographs", []):
        if p.get("series") and p["series"] not in series_ids:
            errors.append(f"{pid}: photograph {p['id']} unknown series {p['series']}")
    if not profile.get("image_rights", {}).get("status"):
        errors.append(f"{pid}: image rights status missing")
    return errors


def completeness(profile: dict) -> dict:
    """The research standard's checklist. Missing areas are reported, never filled in."""
    style = profile.get("style", [])
    motifs = profile.get("motifs", [])
    checks = {
        "visual_style_documented": len({s["dimension"] for s in style}) >= 3 and any(s["kind"] == "observable" for s in style),
        "major_photobooks_researched": bool(profile.get("photobooks")),
        "representative_works_identified": any(p.get("title_given") for p in profile.get("photographs", [])),
        "recurring_motifs_investigated": any(m["level"] != "weak" for m in motifs),
        "major_series_documented": bool(profile.get("series")),
        "style_evolution_considered": len(profile.get("periods", [])) >= 2,
        "claims_supported_by_sources": True,  # validate() fails the build otherwise
        "copyright_status_recorded": bool(profile.get("image_rights", {}).get("status")),
        "uncertain_claims_flagged": "uncertain" in profile,
        "primary_source_consulted": any(s.get("kind") in ("foundation", "museum", "archive", "publisher", "institution") for s in profile.get("sources", {}).values()),
    }
    status = "complete" if all(checks.values()) else "incomplete"
    return {"status": status, "checks": checks, "missing": [k for k, v in checks.items() if not v]}


def build(db: Path = DB) -> dict:
    vocab = load_vocabulary()
    profiles = load_profiles()
    errors = [e for p in profiles for e in validate(p, vocab)]
    if errors:
        raise InvalidProfile("\n".join(errors))
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    c = sqlite3.connect(tmp)
    c.executescript(SCHEMA)
    j = lambda v: json.dumps(v or [], ensure_ascii=False)  # noqa: E731
    c.executemany("INSERT INTO vocabulary VALUES (?, ?)", list(vocab["tags"].items()))
    c.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    report = {}
    for p in profiles:
        pid = p["id"]; comp = completeness(p); report[pid] = comp

        def ev(entity: str, eid: str, item: dict) -> None:
            c.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?)", [(pid, entity, eid, s) for s in item.get("src", [])])

        c.execute("INSERT INTO photographers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            pid, p.get("website_id"), p["name"]["en"], p["name"].get("zh"), p["name"].get("ja"), p.get("born"), p.get("died"),
            p["image_rights"]["status"], p["image_rights"].get("note"), comp["status"], json.dumps(comp), j(p.get("uncertain"))))
        ev("life", pid, {"src": p.get("life_src", [])})
        for sid, s in p["sources"].items():
            c.execute("INSERT INTO sources VALUES (?,?,?,?,?,?)", (pid, sid, s["url"], s["kind"], s["retrieved"], s.get("note")))
        for b in p.get("photobooks", []):
            c.execute("INSERT INTO photobooks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                pid, b["id"], b.get("title_original"), b.get("title_romaji"), b.get("title_en"), b.get("title_zh"), b.get("title_ja"),
                b.get("year"), b.get("publisher"), b.get("isbn"), b.get("isbn_src"), j(b.get("editions")), j(b.get("themes")),
                j(b.get("subjects")), j(b.get("visual")), b.get("career")))
            ev("photobook", b["id"], b)
        for s in p.get("series", []):
            start, end = (s.get("period") or [None, None])[:2]
            c.execute("INSERT INTO series VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                pid, s["id"], s["title"], int(s.get("title_given", True)), start, end, s.get("period_note"), j(s.get("subjects")),
                j(s.get("visual")), s.get("context"), s.get("refdb")))
            c.executemany("INSERT INTO series_photobooks VALUES (?,?,?)", [(pid, s["id"], b) for b in s.get("photobooks", [])])
            ev("series", s["id"], s)
        for ph in p.get("photographs", []):
            c.execute("INSERT INTO photographs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                pid, ph["id"], ph["title"], int(ph.get("title_given", False)), str(ph.get("year") or ""), ph.get("series"), ph.get("photobook"),
                ph.get("location"), j(ph.get("subjects")), ph.get("composition"), ph.get("why"), ph.get("rights"), ph.get("refdb_work_id")))
            ev("photograph", ph["id"], ph)
        for i, m in enumerate(p.get("motifs", [])):
            c.execute("INSERT INTO motifs VALUES (?,?,?,?,?,?,?,?,?,?)", (
                pid, i, m["element"], m["level"], m.get("context"), m.get("frequency_evidence"), m.get("distinctiveness"), m.get("confidence"),
                j(m.get("series")), j(m.get("representative"))))
            c.executemany("INSERT INTO motif_tags VALUES (?,?,?)", [(pid, i, t) for t in m.get("tags", [])])
            ev("motif", str(i), m)
        for i, s in enumerate(p.get("style", [])):
            c.execute("INSERT INTO style_traits VALUES (?,?,?,?,?,?,?)", (pid, i, s["dimension"], s["value"], s["kind"], s.get("scope"), j(s.get("tags"))))
            c.executemany("INSERT INTO style_tags VALUES (?,?,?,?)", [(pid, i, t, s.get("scope")) for t in s.get("tags", [])])
            ev("style", str(i), s)
        for i, pr in enumerate(p.get("periods", [])):
            a, b = (pr.get("range") or [None, None])[:2]
            c.execute("INSERT INTO periods VALUES (?,?,?,?,?)", (pid, i, pr["label"], a, b))
            ev("period", str(i), pr)
    c.commit(); c.close()
    tmp.replace(db)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "completeness.json").write_text(json.dumps(report, indent=1))
    lines = ["# Visual DNA completeness", "", "| photographer | status | missing |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {', '.join(v['missing']) or '-'} |" for k, v in report.items()]
    (REPORT / "completeness.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    for pid, r in build().items():
        print(f"{pid:24s} {r['status']:10s} {', '.join(r['missing'])}")
