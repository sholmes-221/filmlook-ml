"""Hugging Face Hub dataset discovery.

Discovery NEVER enables training: results land in a candidates file with
``training_allowed: false``. A human must verify terms and copy an entry into
configs/datasets.yaml.
"""
from __future__ import annotations

COMMERCIAL_OK = {"cc0-1.0", "cc-by-4.0", "cc-by-2.0", "apache-2.0", "mit", "cdla-permissive-2.0",
                 "odc-by", "pddl", "cc-by-sa-4.0"}
QUERIES = ["film photography", "analog film", "film scan", "photo aesthetics", "image aesthetics",
           "film grain", "public domain photographs", "street photography", "unsplash"]


def discover_hf(queries: list[str] = QUERIES, limit_per_query: int = 15) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    seen: dict[str, dict] = {}
    for q in queries:
        try:
            results = list(api.list_datasets(search=q, limit=limit_per_query, full=True))
        except Exception as e:  # network / rate limit
            seen[f"__error__{q}"] = {"query": q, "error": str(e)}
            continue
        for d in results:
            if d.id in seen:
                continue
            tags = d.tags or []
            lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "unknown")
            seen[d.id] = {
                "id": d.id, "source_url": f"https://huggingface.co/datasets/{d.id}", "query": q,
                "license": lic, "license_commercial_compatible": lic.lower() in COMMERCIAL_OK,
                "downloads": getattr(d, "downloads", None),
                "size_category": next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), None),
                "gated": getattr(d, "gated", None),
                "training_allowed": False,
                "status": "unverified - check card, upstream image rights, and terms before use",
            }
    return sorted(seen.values(), key=lambda r: (not r.get("license_commercial_compatible", False),
                                                 -(r.get("downloads") or 0)))
