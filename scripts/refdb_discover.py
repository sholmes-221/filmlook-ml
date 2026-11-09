#!/usr/bin/env python
"""Discover candidate works for the photographer reference database (metadata only) and write the licence manifest.

  python scripts/refdb_discover.py [--only dorothea_lange,walker_evans] [--depth 3]

Writes data/refdb/manifest.sqlite (works table) and reports/refdb/discovery.json (counts per photographer, rights
class and status, categories visited). Downloads nothing. Re-running refreshes metadata; removed works stay removed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.refdb import manifest, roster, sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    out = P.REPORTS / "refdb" / "discovery.json"; out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out.read_text()) if out.exists() else {}
    client = sources.Commons()
    for e in roster.collectable():
        if (only and e.id not in only) or (not only and not a.refresh and e.id in report):
            continue
        rows, meta = sources.commons_rows(e, client, a.depth) if e.commons else ([], {})
        rows += sources.aic_rows(e) if e.aic else []
        with manifest.connect() as c:
            for r in rows:
                manifest.upsert(c, dict(r))
        counts = Counter((r["rights_class"], r["rights_status"]) for r in rows)
        reasons = Counter(r["reject_reason"] for r in rows if r["reject_reason"])
        report[e.id] = {"candidates": len(rows), "by_class_status": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
                        "reject_reasons": dict(reasons), "categories_visited": meta.get("categories", [])[:80]}
        print(e.id, len(rows), dict(counts), flush=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
