"""The auditable licence manifest: one SQLite file with every candidate work, accepted or not, and a removal log.

Rows are never deleted silently. ``remove(work_id, reason)`` marks a work as removed, deletes its image file and every
derived embedding row, and appends to ``removals`` so the action can be audited and the work stays excluded on the
next discovery run.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filmlook_ml import paths as P

DB = P.DATA / "refdb" / "manifest.sqlite"
IMAGES = P.DATA / "raw" / "refdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS works (
    work_id TEXT PRIMARY KEY,           -- "<source>:<source id>"
    source TEXT NOT NULL,               -- commons | aic | met
    photographer_id TEXT NOT NULL,      -- roster id (website id where it exists)
    artist_text TEXT,                   -- the source's own attribution text
    title TEXT,
    year TEXT,
    source_url TEXT NOT NULL,           -- the record page
    image_url TEXT,                     -- the rendition fetched
    license TEXT,                       -- the source's licence code / short name
    license_evidence_url TEXT,
    rights_class TEXT NOT NULL,         -- rights.CLASSES
    rights_status TEXT NOT NULL,        -- eligible | metadata_only | rejected | removed
    reject_reason TEXT,
    credit TEXT,
    retrieved_at TEXT NOT NULL,
    sha256 TEXT,
    phash TEXT,
    width INTEGER,
    height INTEGER,
    is_monochrome INTEGER,
    duplicate_of TEXT,
    split TEXT,                         -- set by the evaluation split builder
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_works_photographer ON works (photographer_id);
CREATE TABLE IF NOT EXISTS removals (
    work_id TEXT NOT NULL, removed_at TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    work_id TEXT NOT NULL, model TEXT NOT NULL, version TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
    created_at TEXT NOT NULL, PRIMARY KEY (work_id, model, version)
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextmanager
def connect(db: Path = DB) -> Iterator[sqlite3.Connection]:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db, timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    try:
        with c:
            yield c
    finally:
        c.close()


def upsert(c: sqlite3.Connection, row: dict) -> None:
    """Insert or refresh a discovered work. A removed work keeps its removed status."""
    old = c.execute("SELECT rights_status FROM works WHERE work_id = ?", (row["work_id"],)).fetchone()
    if old and old["rights_status"] == "removed":
        return
    extra = row.pop("extra", None)
    row = {**row, "extra_json": json.dumps(extra, ensure_ascii=False) if extra is not None else None}
    cols = ",".join(row); marks = ",".join("?" * len(row))
    update = ",".join(f"{k}=excluded.{k}" for k in row if k not in ("work_id", "sha256", "phash", "width", "height", "is_monochrome", "duplicate_of", "split"))
    c.execute(f"INSERT INTO works ({cols}) VALUES ({marks}) ON CONFLICT(work_id) DO UPDATE SET {update}", tuple(row.values()))


def image_path(work_id: str) -> Path:
    source, sid = work_id.split(":", 1)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sid)[:150]
    return IMAGES / source / f"{safe}.jpg"


def remove(c: sqlite3.Connection, work_id: str, reason: str) -> None:
    """Take a work out: status removed, image file deleted, derived embeddings deleted, audit row written."""
    c.execute("UPDATE works SET rights_status = 'removed', reject_reason = ? WHERE work_id = ?", (reason, work_id))
    c.execute("DELETE FROM embeddings WHERE work_id = ?", (work_id,))
    c.execute("INSERT INTO removals (work_id, removed_at, reason) VALUES (?, ?, ?)", (work_id, now(), reason))
    image_path(work_id).unlink(missing_ok=True)
