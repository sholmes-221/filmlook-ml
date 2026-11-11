"""C17: how many distinct photographers and which licences sit behind Wikimedia Commons film-stock categories.
Read-only API calls with a generic User-Agent, 1.5 s apart, back-off on 429; no image is downloaded.
    python -m research.c17.commons_census   # -> reports/c17/commons_census.json
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "research-census/1.0 (non-commercial dataset survey)"
CATS = {"kodak_ektar_100": "Category:Photographs taken on Kodak Ektar 100 film", "fuji_superia_xtra_400": "Category:Taken on Fuji Superia X-TRA 400",
        "fuji_velvia": "Category:Taken on Fuji Velvia", "kodak_ultramax_400": "Category:Photographs taken on Kodak UltraMax 400 film",
        "kodak_portra": "Category:Photographs taken on Kodak Portra films", "kodak_ektachrome": "Category:Photographs taken on Ektachrome and Elite Chrome film",
        "kodak_gold_200": "Category:Photographs taken on Kodak Gold 200 film"}


def call(params):
    body = urllib.parse.urlencode({**params, "format": "json"}).encode()
    for attempt in range(6):
        time.sleep(1.5)
        try:
            with urllib.request.urlopen(urllib.request.Request(API, data=body, headers={"User-Agent": UA}), timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (attempt + 1)); continue
            raise
    raise RuntimeError("rate limited")


def members(cat, kind):
    out, cont = [], {}
    while True:
        d = call({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": kind, "cmlimit": 500, **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def main():
    rep = {}
    for key, cat in CATS.items():
        files = set(members(cat, "file"))
        for sub in members(cat, "subcat"):
            files |= set(members(sub, "file"))
        files = sorted(files); users, artists, lic = Counter(), Counter(), Counter()
        for i in range(0, len(files), 50):
            d = call({"action": "query", "prop": "imageinfo", "iiprop": "user|extmetadata", "iiextmetadatafilter": "Artist|LicenseShortName", "titles": "|".join(files[i:i + 50])})
            for p in d["query"]["pages"].values():
                ii = (p.get("imageinfo") or [{}])[0]; md = ii.get("extmetadata") or {}
                users[ii.get("user")] += 1
                artists[re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", "")).strip()[:60]] += 1
                lic[(md.get("LicenseShortName") or {}).get("value", "?")] += 1
        top_share = users.most_common(1)[0][1] / max(1, len(files)) if users else 0
        rep[key] = {"category": cat, "files": len(files), "distinct_uploaders": len(users), "distinct_credited_authors": len(artists),
                    "largest_uploader_share": round(top_share, 2), "uploaders_with_5plus": sum(1 for v in users.values() if v >= 5),
                    "licences": dict(lic.most_common(6))}
        print(key, rep[key]["files"], "files |", rep[key]["distinct_uploaders"], "uploaders | top share", rep[key]["largest_uploader_share"], "|", dict(lic.most_common(3)), flush=True)
    Path("reports/c17").mkdir(parents=True, exist_ok=True); Path("reports/c17/commons_census.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
