"""C08b: build the film datasheet knowledge base from the Claude extractions, verifying every quoted entry against the
PDF's own text. Entries whose quote cannot be found are kept but marked `quote_verified: false` and are not used.

    python -m research.c08.datasheet_kb    # -> data/c08/film_datasheets.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SHEETS, CLAUDE, OUT = Path("data/c08/datasheets"), Path("data/c08/claude/datasheet"), Path("data/c08/film_datasheets.json")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def found(quote: str, text: str) -> bool:
    q = norm(quote)
    if not q:
        return False
    if q in text:
        return True
    words = q.split()
    if len(words) >= 4:  # tolerate line breaks, hyphenation and table layout: most 4-word windows must appear
        hits = sum(" ".join(words[i:i + 4]) in text for i in range(len(words) - 3))
        return hits / (len(words) - 3) >= 0.6
    return SequenceMatcher(None, q, text).find_longest_match(0, len(q), 0, len(text)).size >= 0.9 * len(q)


def words_found(quote: str, text: str) -> bool:
    """Second level for multi-column layouts, where pdftotext interleaves columns: 80 % of the quote's words of 4+
    characters occur in the document."""
    words = [w for w in norm(quote).split() if len(w) >= 4]
    return bool(words) and sum(f" {w} " in f" {text} " for w in words) / len(words) >= 0.8


def main() -> None:
    films, stats = [], Counter()
    for f in sorted(CLAUDE.glob("*.json")):
        row = json.loads(f.read_text())
        r = row["result"]
        text = norm((SHEETS / f"{f.stem}.txt").read_text(errors="replace"))
        for e in r.get("documented", []):
            ok = found(e.get("quote", "")or "", text)
            e["quote_verified"] = "verbatim" if ok else ("words" if words_found(e.get("quote") or "", text) else False)
            ok = e["quote_verified"]
            stats[(e["basis"], ok)] += 1
        for d in r.get("manufacturer_descriptions", []):
            d["quote_verified"] = found(d["phrase"], text)
            d["kind"] = "subjective manufacturer description"
            stats[("description", d["quote_verified"])] += 1
        films.append({"source_pdf": f"{f.stem}.pdf", "source_archive": "125px.com technical document archive / manufacturer or distributor copy",
                      "extraction": {"model": row["model"], "created": row["created"], "cost_usd": row["cost_usd"]}, **r})
    OUT.write_text(json.dumps({"generated": "2026-11-11", "note": ("Values with basis read_from_graph are approximate read-offs of printed curves. "
                                                                    "A datasheet describes the film under its reference process and printing/scanning conditions; it does not "
                                                                    "determine the RGB appearance of a digitised photograph."),
                               "films": films}, indent=1, ensure_ascii=False))
    print("films", len(films), "| types", Counter(f["film_type"] for f in films))
    for k, v in sorted(stats.items(), key=str):
        print(" ", k, v)


if __name__ == "__main__":
    main()
