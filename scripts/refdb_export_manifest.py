#!/usr/bin/env python
"""Export the auditable licence manifest and per-photographer counts.

  python scripts/refdb_export_manifest.py
  python scripts/refdb_export_manifest.py --remove commons:12345 --reason "rights holder request"

Writes reports/refdb/license_manifest.csv (every candidate, all five rights classes, status, reason, evidence URL,
checksums) and reports/refdb/photographer_counts.json. --remove takes a work out of the pipeline: its image file and
every derived embedding are deleted and the action is logged in the manifest's removals table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402
from filmlook_ml.refdb import manifest, roster  # noqa: E402

COLS = ("work_id", "photographer_id", "rights_class", "rights_status", "reject_reason", "license", "license_evidence_url", "source_url",
        "title", "year", "artist_text", "credit", "retrieved_at", "sha256", "phash", "width", "height", "is_monochrome", "duplicate_of")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--remove"); ap.add_argument("--reason")
    a = ap.parse_args()
    with manifest.connect() as c:
        if a.remove:
            if not a.reason:
                sys.exit("--reason is required")
            manifest.remove(c, a.remove, a.reason); print("removed", a.remove)
        rows = [dict(r) for r in c.execute(f"SELECT {', '.join(COLS)} FROM works ORDER BY photographer_id, rights_class, work_id")]
        emb = dict(c.execute("SELECT model, COUNT(*) FROM embeddings GROUP BY model").fetchall())
        removals = c.execute("SELECT COUNT(*) FROM removals").fetchone()[0]
    out = P.REPORTS / "refdb"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "license_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["photographer_id"]][f"{r['rights_class']}/{r['rights_status']}"] += 1
        if r["sha256"] and not r["duplicate_of"] and r["rights_status"] == "eligible":
            counts[r["photographer_id"]]["usable_downloaded"] += 1
        if r["duplicate_of"]:
            counts[r["photographer_id"]]["near_duplicates"] += 1
    summary = {"photographers": {k: dict(v) for k, v in counts.items()},
               "metadata_only_photographers": {e.id: e.note for e in roster.ROSTER if e.status == "metadata_only"},
               "embeddings": emb, "removals": removals}
    (out / "photographer_counts.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v.get("usable_downloaded", 0) for k, v in counts.items()}))


if __name__ == "__main__":
    main()
