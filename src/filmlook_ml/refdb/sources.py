"""Discovery from Wikimedia Commons and the Art Institute of Chicago: metadata only, no image bytes.

Etiquette: one request at a time, a descriptive User-Agent without personal contact data, ``maxlag`` on Commons,
Retry-After honoured, >= 1 s between AIC requests. The Library of Congress site answers scripted requests with a
Cloudflare challenge (checked 2026-11-09); that is an access control and is not worked around. Its FSA/OWI scans reach
this pipeline only through the copies on Commons, whose licence templates record their status.
"""
from __future__ import annotations

import html
import re
import time
import unicodedata
from typing import Iterator

import requests

from filmlook_ml.refdb import rights
from filmlook_ml.refdb.manifest import now
from filmlook_ml.refdb.roster import Entry

# Wikimedia's User-Agent policy asks automated clients for contact information; the public project repository is
# given (never a personal e-mail address).
UA = {"User-Agent": "FilmLook-ML-research/0.2 (https://github.com/sholmes-221/filmlook-ml; reference dataset, one request at a time) python-requests",
      "AIC-User-Agent": "filmlook-ml (research; dataset discovery)"}
COMMONS = "https://commons.wikimedia.org/w/api.php"
AIC = "https://api.artic.edu/api/v1/artworks/search"

# Sub-categories that hold something other than the photographer's own photographs.
SKIP_CAT = re.compile(r"(?i)\b(portraits? of|photographs of [^ ]+ [^ ]+$|exhibitions?|books?|signatures?|graves?|memorials?|"
                      r"statues?|works about|in art|stamps?|retouched|colori[sz]ed|derivative|letters?|by other|"
                      r"cameras?|houses?|museum|logos?|documents?|audio|videos?)\b")
# File titles that are not photographs by the photographer.
SKIP_FILE = re.compile(r"(?i)(signature|autograph|letter|book ?cover|title ?page|exhibition|stamp|grave|plaque|"
                       r"retouched|coloriz|colouriz|\bcrop(ped)?\b|detail|poster|advert|newspaper|magazine)")
PORTRAIT_OF = re.compile(r"(?i)(portrait of|photograph of|photo of)\s+([a-zé.\- ]+)")


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def attributed_to(entry: Entry, artist_text: str) -> bool:
    folded = _fold(artist_text)
    return any(_fold(token) in folded for token in entry.match)


class Commons:
    def __init__(self, session: requests.Session | None = None, spacing: float = 1.0):
        self.s, self.spacing = session or requests.Session(), spacing

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "maxlag": 5}
        for attempt in range(6):
            time.sleep(self.spacing)
            # Long title lists go in a POST body (URI length limit); the API accepts POST for read queries.
            r = self.s.post(COMMONS, data=params, headers=UA, timeout=60) if len(str(params)) > 1500 else self.s.get(COMMONS, params=params, headers=UA, timeout=60)
            if r.status_code == 429 or "maxlag" in r.text[:200]:
                wait = float(r.headers.get("Retry-After", 10 * (attempt + 1)))
                time.sleep(min(wait, 300)); continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Commons kept rate limiting")

    def members(self, category: str, kind: str) -> Iterator[str]:
        cont: dict = {}
        while True:
            d = self._get({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + category, "cmtype": kind, "cmlimit": 500, **cont})
            for m in d["query"]["categorymembers"]:
                yield m["title"]
            if "continue" not in d:
                return
            cont = d["continue"]

    def crawl(self, root: str, depth: int = 3) -> tuple[list[str], list[str]]:
        """-> (file titles, categories visited). Breadth first; skipped sub-categories are not entered."""
        seen, files, queue = {root}, [], [(root, 0)]
        while queue:
            cat, d = queue.pop(0)
            files += list(self.members(cat, "file"))
            if d >= depth:
                continue
            for sub in self.members(cat, "subcat"):
                name = sub.split(":", 1)[1]
                if name in seen or SKIP_CAT.search(name):
                    continue
                seen.add(name); queue.append((name, d + 1))
        return list(dict.fromkeys(files)), sorted(seen)

    def info(self, titles: list[str], width: int = 1024) -> Iterator[dict]:
        for i in range(0, len(titles), 50):
            d = self._get({"action": "query", "prop": "imageinfo", "titles": "|".join(titles[i:i + 50]), "iiprop": "url|size|mime|sha1|extmetadata",
                           "iiurlwidth": width, "iiextmetadatafilter": "License|LicenseShortName|LicenseUrl|Copyrighted|UsageTerms|Artist|ObjectName|ImageDescription|DateTimeOriginal|Credit|Categories"})
            for page in d.get("query", {}).get("pages", {}).values():
                if page.get("imageinfo"):
                    yield {"title": page["title"], "pageid": page.get("pageid"), **page["imageinfo"][0]}


def commons_rows(entry: Entry, client: Commons, depth: int = 3) -> tuple[list[dict], dict]:
    files, cats = [], []
    for root in entry.commons:
        f, c = client.crawl(root, depth); files += f; cats += c
    files = list(dict.fromkeys(files))
    rows = []
    for ii in client.info(files):
        em = {k: (v or {}).get("value", "") for k, v in (ii.get("extmetadata") or {}).items()}
        artist = _plain(em.get("Artist", ""))
        cls = rights.classify_commons(em.get("License", ""), em.get("LicenseShortName", ""), em.get("Copyrighted", ""), em.get("UsageTerms", ""))
        title = ii["title"].split(":", 1)[1]
        desc = _plain(em.get("ObjectName", "")) or _plain(em.get("ImageDescription", ""))[:300]
        reason = None
        if ii.get("mime") not in ("image/jpeg", "image/png", "image/tiff"):
            reason = f"not a raster photograph ({ii.get('mime')})"
        elif min(ii.get("width", 0), ii.get("height", 0)) < 400:
            reason = "too small"
        elif SKIP_FILE.search(title) or SKIP_FILE.search(desc[:120]):
            reason = "not a photograph by the photographer (title)"
        elif not attributed_to(entry, artist):
            reason = "attribution does not name the photographer"
        else:
            m = PORTRAIT_OF.search(desc)
            if m and any(_fold(t) in _fold(m.group(2)) for t in entry.match):
                reason = "portrait of the photographer, not by them"
        status = "rejected" if reason else ("eligible" if rights.eligible(cls) else "metadata_only")
        year = re.search(r"\b(18[4-9]\d|19[0-7]\d)\b", em.get("DateTimeOriginal", "") + " " + desc)
        rows.append({
            "work_id": f"commons:{ii['pageid']}", "source": "commons", "photographer_id": entry.id, "artist_text": artist[:300],
            "title": desc[:300] or title, "year": year.group(1) if year else None, "source_url": ii.get("descriptionurl"),
            "image_url": ii.get("thumburl") or ii.get("url"), "license": em.get("LicenseShortName") or em.get("License"),
            "license_evidence_url": ii.get("descriptionurl"), "rights_class": cls, "rights_status": status, "reject_reason": reason,
            "credit": _plain(em.get("Credit", ""))[:300], "retrieved_at": now(),
            "extra": {"file_title": title, "commons_sha1": ii.get("sha1"), "license_url": em.get("LicenseUrl"), "categories": em.get("Categories", "")[:500]},
        })
    return rows, {"categories": cats, "files": len(files)}


def aic_rows(entry: Entry, session: requests.Session | None = None, spacing: float = 1.2) -> list[dict]:
    s, rows = session or requests.Session(), []
    fields = "id,title,artist_title,artist_display,date_display,date_start,is_public_domain,image_id,classification_title,credit_line,copyright_notice"
    for artist in entry.aic:
        page = 1
        while True:
            q = {"query": {"bool": {"must": [{"match_phrase": {"artist_title": artist}}]}}, "fields": fields, "limit": 100, "page": page}
            time.sleep(spacing)
            r = s.post(AIC, json=q, headers=UA, timeout=60); r.raise_for_status(); d = r.json()
            for a in d["data"]:
                if a.get("artist_title") != artist:
                    continue
                cls = rights.classify_museum(a.get("is_public_domain"))
                reason = None
                if not a.get("image_id"):
                    reason = "no image"
                elif "photograph" not in (a.get("classification_title") or "") and "print" not in (a.get("classification_title") or ""):
                    reason = f"classification {a.get('classification_title')}"
                status = "rejected" if reason else ("eligible" if rights.eligible(cls) else "metadata_only")
                rows.append({
                    "work_id": f"aic:{a['id']}", "source": "aic", "photographer_id": entry.id, "artist_text": a.get("artist_display") or artist,
                    "title": a.get("title"), "year": str(a.get("date_start") or "") or None, "source_url": f"https://www.artic.edu/artworks/{a['id']}",
                    "image_url": f"https://www.artic.edu/iiif/2/{a['image_id']}/full/843,/0/default.jpg" if a.get("image_id") else None,
                    "license": "CC0 (AIC public domain)" if cls == "cc0" else (a.get("copyright_notice") or "not public domain"),
                    "license_evidence_url": "https://www.artic.edu/open-access/open-access-images", "rights_class": cls, "rights_status": status,
                    "reject_reason": reason, "credit": a.get("credit_line"), "retrieved_at": now(),
                    "extra": {"date_display": a.get("date_display"), "classification": a.get("classification_title")},
                })
            if page >= d["pagination"]["total_pages"]:
                break
            page += 1
    return rows
