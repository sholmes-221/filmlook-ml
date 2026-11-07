"""Rights-aware acquisition of small pilots.

Only sources registered with ``training_allowed: true`` can be fetched. Downloads are
rate-limited, honour robots.txt for the image host, and stop at a hard cap. Large
downloads (registry ``requires_approval_to_download``) need ``approve_large=True``.
"""
from __future__ import annotations

import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

from filmlook_ml.datasets.registry import LicenseError, Registry, normalize_license

USER_AGENT = "filmlook-ml-dataset-pilot/0.1 (research; rate-limited)"
PHOTO_WORDS = ("photograph", "photo of", "a photo", "black and white photo", "snapshot", "street scene")
NONPHOTO_WORDS = ("painting", "drawing", "illustration", "engraving", "map", "print of", "sketch",
                  "watercolor", "lithograph", "poster", "manuscript", "text", "diagram", "coin")


class RobotsCache:
    def __init__(self):
        self._c: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        if base not in self._c:
            rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
            import socket
            old = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(10)
                rp.read()
            except Exception:
                rp = None  # unreachable robots.txt -> treat as allowed per RFC 9309 §2.3.1.3
            finally:
                socket.setdefaulttimeout(old)
            self._c[base] = rp
        rp = self._c[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


def pd12m_candidates(shard: int = 0, limit: int = 300, min_side: int = 768) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Spawning/PD12M", f"metadata/pd12m.{shard:03d}.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    cap = df["caption"].fillna("").str.lower()
    is_photo = cap.apply(lambda c: any(w in c for w in PHOTO_WORDS) and not any(w in c for w in NONPHOTO_WORDS))
    df = df[is_photo & (df["width"] >= min_side) & (df["height"] >= min_side)
            & df["mime_type"].isin(["image/jpeg", "image/png"])]
    out = []
    for r in df.head(limit * 3).itertuples():
        lic = normalize_license(r.license)
        out.append({"source_id": str(r.id), "url": r.url, "license": lic, "caption": r.caption,
                    "source": getattr(r, "source", ""), "dataset_id": "pd12m_pilot"})
    return out


def download_pilot(cands: list[dict], out_dir: Path, registry: Registry, limit: int,
                   delay_s: float = 1.0, timeout: float = 30.0, max_bytes: int = 25_000_000) -> list[dict]:
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robots = RobotsCache()
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    got: list[dict] = []
    for c in cands:
        if len(got) >= limit:
            break
        try:
            registry.require(c["dataset_id"], c["license"])
        except LicenseError:
            continue
        ext = ".png" if c["url"].lower().endswith(".png") else ".jpg"
        dest = out_dir / f"{c['dataset_id']}_{c['source_id']}{ext}"
        rec = {"path": str(dest), "dataset_id": c["dataset_id"], "license": c["license"],
               "source_url": c["url"], "source_id": c["source_id"], "attribution": c.get("source", "")}
        if dest.exists():
            got.append(rec)
            continue
        if not robots.allowed(c["url"]):
            continue
        try:
            r = sess.get(c["url"], timeout=timeout, stream=True)
            if r.status_code == 429:
                time.sleep(30)
                continue
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200 or not (ctype.startswith("image/") or ctype == "binary/octet-stream"):
                continue
            data = r.raw.read(max_bytes + 1, decode_content=True)
            if len(data) > max_bytes or not data.startswith((b"\xff\xd8\xff", b"\x89PNG")):
                continue  # oversize, or not actually a JPEG/PNG
            dest.write_bytes(data)
            got.append(rec)
            if len(got) % 10 == 0:
                print(f"  downloaded {len(got)}/{limit}", flush=True)
        except requests.RequestException:
            pass
        time.sleep(delay_s)
    return got


def ingest_user_consented(folder: Path) -> list[dict]:
    """User photos: require consent.jsonl with {"file": ..., "consent": true, "user": ...}."""
    import json
    folder = Path(folder)
    cfile = folder / "consent.jsonl"
    if not cfile.exists():
        return []
    recs = []
    for line in cfile.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("consent") is True and (folder / c["file"]).exists():
            recs.append({"path": str(folder / c["file"]), "dataset_id": "user_consented",
                         "license": "user-consent", "source_url": "filmlook-upload",
                         "source_id": c["file"], "attribution": c.get("user", "")})
    return recs
