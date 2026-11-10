"""Shared paths, HTTP etiquette and species-list loading for the species pipeline."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "species"
IMAGES = DATA / "images"
DIST = ROOT / "dist" / "species"
OUT = ROOT / "research" / "species"
WEB = Path("/home/shanmo/misc/film-evaluation/knowledge/nature")          # read-only website repo

# Generic project string, no personal contact data (the brief's wording).
USER_AGENT = "FilmAnalysisResearch/1.0 (species reference set; one request per second)"
GBIF = "https://api.gbif.org/v1"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"                      # iNaturalist research-grade observations

# Licences accepted for images (per media record). Everything else, including NC/SA/ND and empty, is rejected.
LICENCE_MAP = {
    "publicdomain/zero/1.0": "CC0-1.0",
    "licenses/by/4.0": "CC-BY-4.0",
    "licenses/by/3.0": "CC-BY-3.0",
    "licenses/by/2.5": "CC-BY-2.5",
    "licenses/by/2.0": "CC-BY-2.0",
    "licenses/by/1.0": "CC-BY-1.0",
    "publicdomain/mark/1.0": "PDM-1.0",
}


def licence_id(url: str | None) -> str | None:
    """Normalise a Creative Commons licence URL to a short id, or None when the licence is not accepted."""
    if not url:
        return None
    u = url.lower()
    if "-nc" in u or "-sa" in u or "-nd" in u or "/by-" in u:
        return None
    for frag, lid in LICENCE_MAP.items():
        if frag in u:
            return lid
    return None


class RateLimiter:
    """At most one request per `spacing` seconds per host; thread-safe."""

    def __init__(self, spacing: float = 1.0):
        self.spacing, self.next_ok, self.lock = spacing, {}, threading.Lock()

    def wait(self, host: str) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                t = self.next_ok.get(host, 0.0)
                if now >= t:
                    self.next_ok[host] = now + self.spacing
                    return
                delay = t - now
            time.sleep(min(delay, 1.0))


LIMITER = RateLimiter(1.0)


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get(sess: requests.Session, url: str, params: dict | None = None, timeout: int = 60, stream: bool = False,
        max_attempts: int = 6):
    """GET with per-host spacing, HTTP 429 / 503 Retry-After honoured, bounded retries. Returns a Response or None."""
    host = urlsplit(url).netloc
    for attempt in range(max_attempts):
        LIMITER.wait(host)
        try:
            r = sess.get(url, params=params, timeout=timeout, stream=stream)
        except requests.RequestException:
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else 15.0 * (attempt + 1)
            except ValueError:
                wait = 15.0 * (attempt + 1)
            r.close()
            time.sleep(min(wait, 300))
            continue
        if 500 <= r.status_code < 600:
            r.close()
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        return r
    return None


def load_species() -> list[dict]:
    """The 357 encyclopedia species with the website's research record (GBIF key when present) merged in."""
    sp = json.load(open(WEB / "species.json"))["species"]
    research = {r["id"]: r for r in json.load(open(WEB / "encyclopedia" / "research.json"))["species"]}
    out = []
    for s in sp:
        r = research.get(s["id"], {})
        out.append({"id": s["id"], "kind": s["kind"], "scientific_name": s["scientific_name"], "name_en": s.get("name_en"),
                    "family_web": s.get("family") or r.get("family"), "order_web": s.get("order") or r.get("order"),
                    "gbif_key_web": r.get("gbif_key"), "wiki_en": (s.get("wiki") or {}).get("en") or (r.get("wiki_titles") or {}).get("en")})
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
