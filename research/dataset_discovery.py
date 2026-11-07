"""Multilingual discovery of real analog-film (and curated photography) datasets, with a persistent registry.

  python -m research.dataset_discovery discover [--platforms hf,zenodo,figshare,github] [--ttl-days 7] [--max-queries 0]
  python -m research.dataset_discovery seed                 # record the hand-verified sources (idempotent)
  python -m research.dataset_discovery set <key> --status verified|rejected|quarantined|metadata_only|acquired|deferred
                                             [--auth-class A|B|C|D|E] [--reason "..."]
  python -m research.dataset_discovery report               # tables for docs/REAL_FILM_DATASETS.md (stdout)
  python -m research.dataset_discovery status

Discovery is API-based only (Hugging Face Hub, Zenodo, figshare, GitHub search; no HTML scraping, no logins),
polite (serial requests with spacing, bounded result counts), and remembered: `data/manifests/discovery_registry.json`
keeps every candidate with its status so a rejected source is never re-evaluated and a query is not repeated within
`ttl_days`. Nothing here downloads images or enables training: acquisition is a separate, per-source adapter
(scripts/ingest_*.py) registered in research.data_pipeline, and `configs/datasets.yaml` stays the only training gate.

Authenticity classes (decided from recorded evidence, never from a visual classifier):
  A verified real-film scans (documented negatives/slides, owner or institutional records)
  B probable real-film scans (credible analog metadata: scanner EXIF, documented stock, consistent provenance)
  C unverified analog claims (described as film without supporting evidence)        -> quarantined
  D film simulations (digital captures or renders processed to look like film)        -> synthetic experiments only
  E synthetic film images (AI-generated / procedurally synthesized)                  -> synthetic experiments only
Only A and B may enter the real-film pool, and only with an eligible licence (CC0/PD/CC-BY for training; CC-BY-SA is
evaluation-only, as for commons_film; NC/ND/unknown are never ingested: metadata and links only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "manifests" / "discovery_registry.json"
UA = "filmlook-ml-dataset-discovery/0.1 (research; python-urllib)"
STATUSES = ("candidate", "verified", "rejected", "quarantined", "metadata_only", "acquired", "deferred")
AUTH_CLASSES = ("A", "B", "C", "D", "E")
REAL_POOL_CLASSES = ("A", "B")
WEIGHTS = {"authenticity": 0.30, "licensing": 0.25, "diversity": 0.15, "stock_diversity": 0.10, "quality": 0.10, "metadata": 0.10}

QUERIES = {
    "en": ["real analog film photography dataset", "scanned 35mm film photographs dataset", "analog film scans machine learning dataset",
           "Kodak Portra 400 real scans dataset", "Fujifilm Superia real film scans", "35mm negative film digitization dataset",
           "film stock classification dataset", "analog photography dataset", "film photography dataset", "real film grain dataset",
           "film scan dataset", "film negative restoration benchmark", "film scanning color correction dataset"],
    "zh": ["真实胶片摄影 数据集", "胶片扫描照片 开源数据集", "135胶卷 扫描照片 数据集", "胶卷型号 图像数据集", "柯达 Portra 400 实拍扫描样片",
           "富士 Superia 胶卷 实拍扫描", "胶片摄影 机器学习 数据集", "胶片负片 数字化 数据集", "胶片颗粒 真实样本", "胶片摄影 开放授权 图片",
           "胶片扫描 学术论文 数据集"],
    "ja": ["フィルム写真 データセット", "フィルムスキャン 画像データセット", "35mmフィルム スキャン画像", "ネガフィルム デジタル化 データセット",
           "フィルム写真 オープンデータ", "Kodak Portra 400 作例 スキャン", "富士フイルム SUPERIA 実写 サンプル", "銀塩写真 画像データセット",
           "フィルム粒状性 画像解析", "写真フィルム スキャン 公開データ", "フィルム写真 機械学習 データ"],
}
# short tokens the Hub's substring search understands better than sentences
HF_TOKENS = ["film", "analog", "analogue", "35mm", "negative", "kodak", "fuji", "portra", "superia", "lomo", "grain", "darkroom", "polaroid", "胶片", "フィルム"]

# licence normalisation -> project ids (configs/datasets.yaml allowlist / denylist)
_LIC = [(r"\bpdm\b|public domain mark|publicdomain/mark", "PDM-1.0"), (r"cc0|public ?domain|pddl|cc-pddc|publicdomain/zero", "CC0-1.0"),
        (r"cc[- ]by[- ]nc[- ]nd", "CC-BY-NC-ND-4.0"), (r"cc[- ]by[- ]nc[- ]sa", "CC-BY-NC-SA-4.0"), (r"cc[- ]by[- ]nc", "CC-BY-NC-4.0"),
        (r"cc[- ]by[- ]nd", "CC-BY-ND-4.0"), (r"cc[- ]by[- ]sa[- ]?(2\.0)", "CC-BY-SA-2.0"), (r"cc[- ]by[- ]sa", "CC-BY-SA-4.0"),
        (r"cc[- ]by[- ]?2\.0|licenses/by/2\.0", "CC-BY-2.0"), (r"cc[- ]by\b|licenses/by/|attribution 4", "CC-BY-4.0"),
        (r"apache", "Apache-2.0"), (r"\bmit\b|mit-license", "MIT"), (r"cdla-permissive", "CDLA-Permissive-2.0")]
TRAIN_OK = {"CC0-1.0", "PDM-1.0", "CDLA-Permissive-2.0", "Apache-2.0", "CC-BY-4.0", "CC-BY-2.0"}
EVAL_ONLY = {"CC-BY-SA-4.0", "CC-BY-SA-2.0"}


def normalize_license(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return "unknown"
    for pat, lic in _LIC:
        if re.search(pat, s):
            return lic
    return "unknown"


def license_eligibility(raw: str | None) -> dict:
    lic = normalize_license(raw)
    return {"license": lic, "training_allowed": lic in TRAIN_OK, "evaluation_allowed": lic in TRAIN_OK or lic in EVAL_ONLY,
            "ingest_allowed": lic in TRAIN_OK or lic in EVAL_ONLY}


# ------------------------------------------------------------------------------------------- authenticity (evidence-based)
_E_RE = re.compile(r"ai[- ]generated|generated by|diffusion|synthetic|stable diffusion|pixart|flux|midjourney|procedural", re.I)
_D_RE = re.compile(r"simulat|emulat|preset|\blut\b|film look|film[- ]style|style transfer|recipe|capture one|lightroom|vsco|filter", re.I)
_A_RE = re.compile(r"negative|slide|contact sheet|dng|scanner|scanned on|emulsion|developed|c-?41|e-?6|archive|museum|library|estate", re.I)
_C_RE = re.compile(r"film photograph|analog|analogue|35 ?mm|120 film|medium format|shot on film|フィルム|胶片|胶卷|銀塩", re.I)


def classify_authenticity(evidence: dict) -> tuple[str, str]:
    """evidence: {text, physical_provenance (owner/institution records of negatives or scans), analog_metadata (scanner EXIF,
    documented stock, scan process), synthetic, simulation}. Flags win over text; text alone never yields A or B."""
    if evidence.get("synthetic"):
        return "E", "declared synthetic / AI-generated"
    if evidence.get("simulation"):
        return "D", "declared film simulation of digital captures"
    text = evidence.get("text") or ""
    if evidence.get("physical_provenance"):
        return "A", "documented physical origin: " + str(evidence["physical_provenance"])
    if evidence.get("analog_metadata"):
        return "B", "credible analog metadata: " + str(evidence["analog_metadata"])
    if _E_RE.search(text):
        return "E", "text describes generated images"
    if _D_RE.search(text) and not _A_RE.search(text):
        return "D", "text describes simulated film looks"
    if _C_RE.search(text) or _A_RE.search(text):
        return "C", "analog claim without recorded evidence (quarantine until verified)"
    return "C", "no evidence recorded"


def rank(rec: dict) -> float:
    """Discovery heuristic (brief weights); hard licence/authenticity requirements are checked separately."""
    s = rec.get("scores") or {}
    auth = {"A": 1.0, "B": 0.7, "C": 0.2, "D": 0.0, "E": 0.0}.get(rec.get("auth_class") or "C", 0.2)
    el = license_eligibility(rec.get("license_raw") or rec.get("license"))
    lic = 1.0 if el["training_allowed"] else 0.6 if el["evaluation_allowed"] else 0.0
    if rec.get("auth_class") in ("D", "E") or not el["ingest_allowed"]:
        return 0.0                                               # hard requirements: no real-pool value whatever the other scores
    return round(WEIGHTS["authenticity"] * auth + WEIGHTS["licensing"] * lic + WEIGHTS["diversity"] * s.get("diversity", 0.3)
                 + WEIGHTS["stock_diversity"] * s.get("stock_diversity", 0.2) + WEIGHTS["quality"] * s.get("quality", 0.5) + WEIGHTS["metadata"] * s.get("metadata", 0.3), 3)


def eligible_for_real_pool(rec: dict) -> tuple[bool, str]:
    el = license_eligibility(rec.get("license_raw") or rec.get("license"))
    if rec.get("auth_class") not in REAL_POOL_CLASSES:
        return False, f"authenticity class {rec.get('auth_class')} (only A/B enter the real-film pool)"
    if not el["ingest_allowed"]:
        return False, f"licence {el['license']} not ingestible (metadata and links only)"
    return True, "eligible" + ("" if el["training_allowed"] else " (evaluation-only licence)")


# ------------------------------------------------------------------------------------------------------ registry
def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"version": 1, "sources": {}, "searches": {}}


def save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".json.part"); tmp.write_text(json.dumps(reg, indent=1, ensure_ascii=False, sort_keys=True)); os.replace(tmp, REGISTRY)


def upsert(reg: dict, rec: dict) -> dict:
    """Add a candidate; never downgrade a decided status (verified/rejected/... are human-or-adapter decisions)."""
    key = rec["key"]; cur = reg["sources"].get(key)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if cur is None:
        rec = {"status": "candidate", "first_seen": now, "last_checked": now, "queries": [], **rec}
        rec.setdefault("auth_class", None); rec.setdefault("evidence", {}); rec.setdefault("license", normalize_license(rec.get("license_raw")))
        reg["sources"][key] = rec; return rec
    for q in rec.get("queries", []):
        if q not in cur.setdefault("queries", []):
            cur["queries"].append(q)
    for k in ("name", "url", "publisher", "description", "license_raw", "size_hint", "platform"):
        if rec.get(k) and not cur.get(k):
            cur[k] = rec[k]
    cur["last_checked"] = now
    return cur


def set_status(key: str, status: str, auth_class: str | None = None, reason: str | None = None, evidence: dict | None = None, **fields) -> dict:
    assert status in STATUSES, status
    reg = load(); rec = reg["sources"].get(key)
    if rec is None:
        rec = upsert(reg, {"key": key, "platform": key.split(":", 1)[0], **fields})
    rec["status"] = status; rec["decided"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if auth_class:
        assert auth_class in AUTH_CLASSES; rec["auth_class"] = auth_class
    if reason:
        rec["reason"] = reason
    if evidence:
        rec["evidence"] = {**(rec.get("evidence") or {}), **evidence}
    rec.update(fields); rec["rank"] = rank(rec); save(reg); return rec


# ------------------------------------------------------------------------------------------------------ adapters
def _get(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 40):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search_hf(query: str, limit: int = 50) -> list[dict]:
    items = _get("https://huggingface.co/api/datasets?" + urllib.parse.urlencode({"search": query, "limit": limit}))
    out = []
    for it in items:
        tags = it.get("tags") or []
        if "modality:image" not in tags:
            continue
        lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
        out.append({"key": f"hf:{it['id']}", "platform": "hf", "name": it["id"], "url": f"https://huggingface.co/datasets/{it['id']}", "publisher": it["id"].split("/")[0],
                    "license_raw": lic, "size_hint": next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), None),
                    "downloads": it.get("downloads"), "gated": it.get("gated"), "queries": [query]})
    return out


def search_zenodo(query: str, limit: int = 15) -> list[dict]:
    d = _get("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": query, "size": limit, "type": "dataset"}))
    out = []
    for h in d.get("hits", {}).get("hits", []):
        m = h.get("metadata", {}); lic = m.get("license"); lic = lic.get("id") if isinstance(lic, dict) else lic
        out.append({"key": f"zenodo:{h['id']}", "platform": "zenodo", "name": m.get("title"), "url": h.get("links", {}).get("self_html") or f"https://zenodo.org/records/{h['id']}",
                    "publisher": ", ".join(c.get("name", "") for c in m.get("creators", [])[:3]), "license_raw": lic, "description": (m.get("description") or "")[:600],
                    "size_hint": str(len(h.get("files", []))) + " files", "queries": [query]})
    return out


def search_figshare(query: str, limit: int = 10) -> list[dict]:
    items = _get("https://api.figshare.com/v2/articles/search", data=json.dumps({"search_for": query, "page_size": limit, "item_type": 3}).encode(), headers={"Content-Type": "application/json"})
    return [{"key": f"figshare:{it['id']}", "platform": "figshare", "name": it.get("title"), "url": it.get("url_public_html"), "publisher": None, "license_raw": None,
             "queries": [query]} for it in items]


def search_github(query: str, limit: int = 10) -> list[dict]:
    d = _get("https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": limit, "sort": "stars"}))
    return [{"key": f"github:{it['full_name']}", "platform": "github", "name": it["full_name"], "url": it["html_url"], "publisher": it["owner"]["login"],
             "license_raw": (it.get("license") or {}).get("spdx_id"), "description": (it.get("description") or "")[:300], "queries": [query]} for it in d.get("items", [])]


ADAPTERS = {"hf": (search_hf, 0.8), "zenodo": (search_zenodo, 1.5), "figshare": (search_figshare, 1.5), "github": (search_github, 8.0)}
_KW = re.compile(r"film|analog|analogue|35 ?mm|negativ|kodak|fuji|portra|superia|grain|darkroom|slide|lomo|胶片|胶卷|フィルム|銀塩", re.I)


def discover(platforms=("hf", "zenodo", "figshare", "github"), ttl_days: float = 7, max_queries: int = 0, sleep=time.sleep, adapters=None, log=print) -> dict:
    """Run every (platform, query) pair not searched within ttl_days; keep candidates whose name/description mention film."""
    adapters = adapters or ADAPTERS; reg = load(); now = time.time(); n_q, n_new, errors = 0, 0, []
    queries = [(p, q) for p in platforms for lang in QUERIES for q in (HF_TOKENS if p == "hf" and lang == "en" else QUERIES[lang]) if not (p == "hf" and lang != "en")]
    for p, q in queries:
        sk = f"{p}|{q}"; last = reg["searches"].get(sk, {}).get("ts", 0)
        if now - last < ttl_days * 86400:
            continue
        if max_queries and n_q >= max_queries:
            break
        fn, spacing = adapters[p]
        try:
            res = fn(q)
        except Exception as ex:
            errors.append({"platform": p, "query": q, "error": repr(ex)[:200]}); reg["searches"][sk] = {"ts": now, "error": repr(ex)[:200]}; sleep(spacing); n_q += 1; continue
        kept = 0
        for rec in res:
            if not _KW.search(f"{rec.get('name', '')} {rec.get('description', '')}"):
                continue
            before = rec["key"] in reg["sources"]; upsert(reg, rec); kept += 1; n_new += 0 if before else 1
        reg["searches"][sk] = {"ts": now, "results": len(res), "kept": kept}; n_q += 1
        log(f"[{p}] {q!r}: {len(res)} results, {kept} film-related"); sleep(spacing)
    save(reg)
    return {"queries_run": n_q, "new_candidates": n_new, "errors": errors, "sources": len(reg["sources"])}


# ------------------------------------------------------------------------------------------------- hand-verified seed
def verified_seed() -> list[dict]:
    """Sources checked in the 2026-11-07/08 session through their APIs or published terms (evidence quoted from them)."""
    S = []
    def add(key, status, auth, **f):
        S.append({"key": key, "platform": key.split(":")[0], "status": status, "auth_class": auth, **f})
    add("hf:common-canvas/commoncatalog-cc-by#film_scan_stock_tags", "verified", "B", name="CommonCatalog CC-BY film scans with uploader film-stock tags",
        url="https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by", publisher="common-canvas (YFCC100M / Flickr)", license_raw="CC-BY-2.0 per row (licenseurl)",
        evidence={"analog_metadata": "EXIF capture device is a film scanner or minilab (Noritsu QSS, Fuji Frontier SP, Coolscan, Epson, Plustek...) AND the uploader tagged a film stock (portra, superia, tri-x, ...)",
                  "text": "metadata-only scan of 60 shards: 1,605 film-scan rows from 468 uploaders, 365 with stock-like tags from 105 uploaders"},
        scores={"diversity": 0.9, "stock_diversity": 0.8, "quality": 0.7, "metadata": 0.7}, size_hint="~5,500 parquet shards; thousands of film scans", dataset_id="commoncatalog_film",
        reason="official Hub parquet range reads (no Flickr scraping); stock = uploader claim; uploaders used for training are excluded; evaluation-only snapshot")
    add("figshare:21803304", "verified", "A", name="ARTeFACT: authentically damaged & manually restored 35mm film scans (Ivanova et al., Eurographics 2023)",
        url="https://figshare.com/articles/dataset/Authentically_damaged_film_scans/21803304", publisher="Daniela Ivanova et al.", license_raw="CC BY 4.0",
        evidence={"physical_provenance": "paper documents 44 scanned 35mm slides and negatives of named colour emulsions (file names: cinestill50d, cinestill800t, ...) with manual restorations", "text": "88 JPEG files ~5 MB each"},
        scores={"diversity": 0.2, "stock_diversity": 0.6, "quality": 0.9, "metadata": 0.6}, size_hint="88 files, ~450 MB", dataset_id="artefact_film",
        reason="single photographer/scanner; damage artefacts present; evaluation-only (authenticity + per-emulsion retrieval sanity)")
    add("figshare:21803292", "rejected", "A", name="Scans of film damage (empty Lomochrome Purple frames)", url="https://figshare.com/articles/dataset/Scans_of_film_damage/21803292",
        license_raw="CC BY 4.0", evidence={"physical_provenance": "20 empty developed frames, Plustek OpticFilm 7600i, 300 dpi"}, reason="no photographic content (artefact-only frames)")
    add("github:DhaliwalX/gold200-dataset", "rejected", "A", name="Kodak Gold 200 film scans (RM120 R/G/B) and matching Sony A7C II raws", url="https://github.com/DhaliwalX/gold200-dataset",
        license_raw="CC BY-SA 4.0", evidence={"physical_provenance": "README documents 3 frames of one Gold 200 roll scanned per LED channel plus digital captures of the same scenes"},
        reason="3 frames only; recorded for paired film-vs-digital reference, not ingested")
    add("zenodo:22170495", "rejected", "A", name="grainifier frame statistics dataset 2026.08 (197 frames Double-X 5222)", url="https://zenodo.org/records/22170495", license_raw="cc-by-4.0",
        evidence={"physical_provenance": "per-frame grain statistics from Nikon Coolscan V ED 4000 dpi linear scans"}, reason="statistics only (npz), no image pixels")
    add("zenodo:14283642", "deferred", "A", name="Photographs by Robert Breitinger (1886-1910, Zentralbibliothek Zuerich)", url="https://zenodo.org/records/14283642", license_raw="cc-pddc",
        evidence={"physical_provenance": "complete photographic estate archived by the Zentralbibliothek Zuerich, scans with metadata"}, size_hint="3.3 GB zip",
        reason="institutional, public domain; 19th-century plate era, no stock labels; candidate for cross-era authenticity checks later")
    add("zenodo:10233853", "rejected", None, name="An Ode to Film (3D model)", url="https://zenodo.org/records/10233853", license_raw="cc-by-4.0", reason="3D model (Objaverse), not photographs")
    add("zenodo:19263125", "rejected", None, name="Catalogue of 35mm colour photographs of Saudi Arabia (KAFRA)", url="https://zenodo.org/records/19263125", license_raw="cc-by-4.0", reason="catalogue (xlsx/pdf) without images")
    add("zenodo:6184281", "rejected", "A", name="HISTORIAN: historical film dataset (98 digitised WWII reels)", url="https://zenodo.org/records/6184281", license_raw="CC BY 4.0",
        reason="motion-picture frames, not still photography; out of scope for the film-aesthetics pool")
    add("hf:pslabs/35mm_32k", "quarantined", "C", name="pslabs/35mm_32k", url="https://huggingface.co/datasets/pslabs/35mm_32k", license_raw=None,
        reason="no licence, no dataset card, two zip files: provenance unknown; not ingested")
    add("hf:Nbardy/film_stock", "quarantined", "C", name="Nbardy/film_stock", url="https://huggingface.co/datasets/Nbardy/film_stock", license_raw=None, reason="no licence or card (n<1K parquet); not ingested")
    add("hf:wornleto/grain", "quarantined", "C", name="wornleto/grain", url="https://huggingface.co/datasets/wornleto/grain", license_raw=None, reason="no licence or card; not ingested")
    add("hf:nuhmanpk/darkroom-pinhole-18k", "rejected", "E", name="Darkroom: synthetic training data for Pinhole", url="https://huggingface.co/datasets/nuhmanpk/darkroom-pinhole-18k", license_raw="openrail++",
        evidence={"synthetic": True}, reason="PixArt-generated icons/pixel art; not photographs")
    add("kaggle:xuhangc/filmset", "metadata_only", "D", name="FilmSet (IJCAI 2023): 5,285 photos x 3 Capture One film recipes", url="https://www.kaggle.com/datasets/xuhangc/filmset", license_raw=None,
        evidence={"simulation": True, "text": "arXiv 2301.08880: film recipes applied in Capture One to in-the-wild digital photos; sourcing described as licence-free samples, no formal licence"},
        reason="film simulation (class D); licence unclear; never mixed into real-film evaluation")
    add("hf:prs-eth/BlueNeg", "acquired", "A", name="BlueNeg: 491 35mm colour negatives (ICCV 2025)", url="https://huggingface.co/datasets/prs-eth/BlueNeg", license_raw="CC-BY derivative (credit Tien-Tsin Wong)",
        evidence={"physical_provenance": "owner's negatives scanned to 16-bit DNG with roll records"}, dataset_id="blueneg", reason="already registered; test-only in benchmark v1")
    add("wikimedia:Category:Photographs_by_film_brand", "acquired", "B", name="Wikimedia Commons film-brand categories", url="https://commons.wikimedia.org/wiki/Category:Photographs_by_film_brand",
        license_raw="per-file CC0/PD/CC-BY/CC-BY-SA", evidence={"analog_metadata": "uploader-asserted stock categories, roll/scanner categories for some, EXIF when present"}, dataset_id="commons_film",
        reason="partial (266/5,743): per-client HTTP 429 throttling honoured with backoff; deprioritised")
    add("artic:photography_public_domain", "verified", "A", name="Art Institute of Chicago public-domain photographs (CC0, IIIF)", url="https://api.artic.edu/docs/", publisher="Art Institute of Chicago",
        license_raw="CC0 (is_public_domain)", evidence={"physical_provenance": "museum catalogue records of the physical print process (medium_display), artist and date; 3,777 works with images, mostly 1840-1920s"},
        scores={"diversity": 0.8, "stock_diversity": 0.1, "quality": 0.8, "metadata": 0.9}, dataset_id="aic_photographs", group="curated_photography",
        reason="curated-photography group (not analog film stock): process labels are owner records; CC0 permits training; pilot capped per artist")
    add("metmuseum:photographs_public_domain", "deferred", "A", name="The Met Photographs department, public domain (CC0)", url="https://metmuseum.github.io/", license_raw="CC0",
        reason="objects endpoint works; the search endpoint returned HTTP 410 on 2026-11-07; needs the open-access CSV route")
    add("smithsonian:open_access", "deferred", "A", name="Smithsonian Open Access (CC0 subset, S3 bulk metadata)", url="https://registry.opendata.aws/smithsonian-open-access/", license_raw="CC0 (per record)",
        reason="public S3 metadata listed; needs per-unit filtering for photographic negatives (e.g. nmah, npg, aaa); large")
    add("condenast:photovogue", "metadata_only", None, name="PhotoVogue (Conde Nast)", url="https://www.vogue.com/photovogue/photos", publisher="Conde Nast", license_raw="all rights reserved (photographers)",
        evidence={"text": "Conde Nast User Agreement: no automated access, and no use of the Service in connection with the development, training or fine-tuning of any machine-learning model without prior written consent; robots.txt disallows query URLs and */c_limit/ image paths"},
        group="curated_photography", reason="not crawled, no pixels, no embeddings; public landing URL recorded only")
    add("iconicartist:dehazed", "metadata_only", None, name="DEHAZED (dehazed.eu, ICONIC ARTIST, Munich)", url="https://www.dehazed.eu/", publisher="ICONIC ARTIST",
        license_raw="all rights reserved (submitting photographers)", evidence={"text": "paid-submission online gallery/magazine; Squarespace robots.txt lists AI crawlers; no licence covering ML use"},
        group="curated_photography", reason="not crawled; metadata only")
    add("iconicartist:gallery", "metadata_only", None, name="Iconic Artist (iconicartist.eu)", url="https://www.iconicartist.eu/", publisher="ICONIC ARTIST",
        license_raw="all rights reserved", evidence={"text": "print-sales gallery; Squarespace robots.txt lists AI crawlers; no licence covering ML use"}, group="curated_photography", reason="not crawled; metadata only")
    return S


def seed() -> int:
    reg = load(); n = 0
    for rec in verified_seed():
        cur = reg["sources"].get(rec["key"])
        if cur and cur.get("status") not in (None, "candidate"):
            continue                                                     # keep later decisions
        upsert(reg, {k: v for k, v in rec.items() if k not in ("status",)}); r = reg["sources"][rec["key"]]
        r.update({k: v for k, v in rec.items()}); r["rank"] = rank(r); r["decided"] = time.strftime("%Y-%m-%dT%H:%M:%S"); n += 1
    save(reg); return n


# --------------------------------------------------------------------------------------------------------- report
def report(reg: dict | None = None) -> str:
    reg = reg or load(); rows = sorted(reg["sources"].values(), key=lambda r: (-(r.get("rank") or 0), r["key"]))
    def row(r):
        el = license_eligibility(r.get("license_raw") or r.get("license"))
        return f"| `{r['key']}` | {r.get('name') or ''} | {r.get('status')} | {r.get('auth_class') or '-'} | {el['license']} | {'train' if el['training_allowed'] else 'eval' if el['evaluation_allowed'] else 'none'} | {r.get('rank') or '-'} | {(r.get('reason') or '')[:160]} |"
    head = "| key | source | status | class | licence | use | rank | note |\n|---|---|---|---|---|---|---|---|"
    decided = [r for r in rows if r.get("status") != "candidate"]; cands = [r for r in rows if r.get("status") == "candidate"]
    by = {}
    for r in rows:
        by[r.get("status")] = by.get(r.get("status"), 0) + 1
    out = [f"Registry: {len(rows)} sources ({', '.join(f'{k} {v}' for k, v in sorted(by.items()))}); {len(reg.get('searches', {}))} platform/query pairs searched.", "",
           "### Decided sources", head, *[row(r) for r in decided], "", f"### Undecided candidates ({len(cands)})", head, *[row(r) for r in cands[:60]]]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["discover", "seed", "set", "report", "status"]); ap.add_argument("key", nargs="?")
    ap.add_argument("--platforms", default="hf,zenodo,figshare,github"); ap.add_argument("--ttl-days", type=float, default=7); ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--status", default=None); ap.add_argument("--auth-class", default=None); ap.add_argument("--reason", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "discover":
        print(json.dumps(discover(tuple(p for p in a.platforms.split(",") if p), a.ttl_days, a.max_queries), indent=1))
    elif a.cmd == "seed":
        print("seeded", seed())
    elif a.cmd == "set":
        print(json.dumps(set_status(a.key, a.status, a.auth_class, a.reason), indent=1, ensure_ascii=False))
    elif a.cmd == "report":
        print(report())
    else:
        reg = load(); print(json.dumps({"sources": len(reg["sources"]), "by_status": {s: sum(1 for r in reg["sources"].values() if r.get("status") == s) for s in STATUSES},
                                        "searches": len(reg["searches"])}, indent=1))


if __name__ == "__main__":
    main()
