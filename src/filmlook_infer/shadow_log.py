"""Local shadow-mode logging for research runs. Not wired into any application.

Writes one JSON line per analysed photo with the shadow record and a content hash, never the pixels, the file
name or any EXIF field. Intended for offline comparison of model outputs on photos the researcher already holds
the rights to. Collecting user photographs or hooking this into the production app needs a separate privacy and
retention review first; this module does neither.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from filmlook_infer.shadow import ShadowAnalyzer

LOG_SCHEMA_VERSION = "shadow-log-1.0"
DROP_KEYS = {"image_path", "path", "filename"}


def content_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShadowLogger:
    def __init__(self, analyzer: ShadowAnalyzer, log_path: str | Path, run_id: str):
        self.an, self.path, self.run_id = analyzer, Path(log_path), run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, image_path: str | Path, language: str = "en", context: dict | None = None) -> dict:
        record = self.an.analyze(str(image_path), language=language)
        row = {"log_schema": LOG_SCHEMA_VERSION, "run_id": self.run_id, "ts": datetime.now(timezone.utc).isoformat(),
               "content_sha256": content_hash(image_path), "record": {k: v for k, v in record.items() if k not in DROP_KEYS},
               "context": {k: v for k, v in (context or {}).items() if k not in DROP_KEYS}}
        line = json.dumps(row, ensure_ascii=False)
        assert str(image_path) not in line, "a path leaked into the shadow log"
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row
