"""Select the licensed source photographs for LUT-based synthetic training data.

Writes research/lut_library/source_photos.json: one record per image with id, absolute path, licence,
source dataset, attribution, split, capture medium, image size and a *heuristic* scene-category guess.
No hosted model is called; categories come from existing uploader tags (CommonCatalog `subjects`), the
dataset's own nature (PD12M iNaturalist = plants/animals) and cheap pixel statistics (brightness, contrast,
saturation, sky/green/skin fractions). They are guesses for balancing a generator, not labels.

Pool and gates (configs/datasets.yaml is the only data gate; nothing here overrides it):
  * commoncatalog_ccby  (training_allowed: true, CC-BY-2.0 per row)  - digital_camera and smartphone originals
    only (tier 1). Film scans and filtered uploads are listed in tier 2 (a film LUT on a film scan or a filtered
    phone shot stacks two looks); the generator decides whether to use them.
  * pd12m_pilot        (training_allowed: true, CC0)                 - tier 1.
  * PD12M iNaturalist images used by the photo gate (CC0; data/raw/photo_gate/pd12m_inaturalist) - tier 1, the
    only local source of birds / flowers / wildlife. They were the photo gate's *test* images, so the field
    `photo_gate_split` is kept and the gate must never be evaluated on synthetic derivatives of them.
  * excluded: met_photographs and artefact_film (evaluation-only), commoncatalog_film* (evaluation-only),
    aic_photographs (CC0 but 'never mixed into synthetic sets without an explicit experiment definition'),
    photographer_refdb (historical B&W reproductions), commons_film (evaluation-only), the quarantined image
    (configs/quarantine.yaml) and its family, and every original in the project's *test* split.

Run from the repository root:
    .venv/bin/python -I research/lut_library/source_photos.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "source_photos.json"
MAX_SIDE = 1600
TODAY = "2026-11-10"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def quarantined_ids() -> set[str]:
    try:
        import yaml  # type: ignore
        q = yaml.safe_load((ROOT / "configs" / "quarantine.yaml").read_text()) or {}
        return {str(x.get("source_id")) for x in q.get("images", [])}
    except Exception:
        return {"9733492355"}  # the one recorded on 2026-11-07


def pixel_stats(path: Path) -> dict:
    """Cheap statistics on a 256 px thumbnail (sRGB assumed)."""
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("RGB")
        im.thumbnail((256, 256))
        a = np.asarray(im, dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    top = slice(0, a.shape[0] // 3)
    sky = ((b[top] > r[top] + 0.05) & (b[top] > g[top]) & (luma[top] > 0.45)).mean()
    green = ((g > r + 0.04) & (g > b + 0.04) & (sat > 0.15)).mean()
    # crude skin-tone mask (works poorly on strongly tinted images; recorded as a heuristic only)
    skin = (r > g) & (g > b) & (r - b > 0.06) & (r - g < 0.35) & (sat > 0.12) & (sat < 0.7) & (luma > 0.15)
    skin_frac = float(skin.mean())
    skin_luma = float(luma[skin].mean()) if skin_frac > 0.01 else None
    p5, p95 = np.percentile(luma, [5, 95])
    return {
        "width": w, "height": h, "max_side": max(w, h), "needs_downscale": max(w, h) > MAX_SIDE,
        "mean_luma": round(float(luma.mean()), 3), "luma_std": round(float(luma.std()), 3),
        "p5_p95_range": round(float(p95 - p5), 3), "mean_saturation": round(float(sat.mean()), 3),
        "monochrome_estimate": bool(sat.mean() < 0.06), "sky_fraction": round(float(sky), 3),
        "green_fraction": round(float(green), 3), "skin_fraction": round(skin_frac, 3),
        "skin_luma": round(skin_luma, 3) if skin_luma is not None else None,
        "dark_fraction": round(float((luma < 0.1).mean()), 3), "bright_fraction": round(float((luma > 0.9).mean()), 3),
    }


def categorise(tags: list[str], st: dict, source_kind: str) -> tuple[str, list[str]]:
    cats: list[str] = []
    tags = [t.lower() for t in tags]
    for t in ("portrait", "landscape", "architecture", "street", "night"):
        if t in tags:
            cats.append(t)
    if source_kind == "inaturalist":
        cats.append("nature_wildlife")  # iNaturalist observations: plants (flowers) and animals (incl. birds)
        if st["green_fraction"] > 0.35:
            cats.append("foliage_flowers")
    if st["mean_luma"] < 0.22 or st["dark_fraction"] > 0.45:
        cats.append("low_light")
        if "night" not in cats and st["bright_fraction"] > 0.01:
            cats.append("night_lights")
    if st["p5_p95_range"] > 0.85 or st["luma_std"] > 0.30:
        cats.append("high_contrast")
    if st["sky_fraction"] > 0.25 and "landscape" not in cats and "architecture" not in cats:
        cats.append("outdoor_sky")
    if st["green_fraction"] > 0.30 and "landscape" not in cats and source_kind != "inaturalist":
        cats.append("foliage_greenery")
    if st["skin_fraction"] > 0.18 and "portrait" not in cats:
        cats.append("people_skin")
    if st["monochrome_estimate"]:
        cats.append("monochrome")
    if st["mean_saturation"] > 0.45:
        cats.append("high_saturation")
    if not cats:
        cats.append("indoor_or_other" if st["sky_fraction"] < 0.02 and st["green_fraction"] < 0.05 else "general_outdoor")
    return cats[0], cats


def skin_bucket(st: dict) -> str | None:
    """Light / medium / deep buckets from the mean luminance of skin-mask pixels (very crude, exposure-dependent)."""
    if st["skin_fraction"] < 0.08 or st["skin_luma"] is None:
        return None
    l = st["skin_luma"]
    return "deep" if l < 0.38 else ("medium" if l < 0.6 else "light")


def main() -> int:
    quarantine = quarantined_ids()
    processed = load_jsonl(ROOT / "data" / "manifests" / "manifest.jsonl")
    split_by_source = {r["source_id"]: r["split"] for r in processed if r.get("edit_type") == "original"}
    records = []

    # 1) CommonCatalog CC-BY
    for r in load_jsonl(ROOT / "data" / "manifests" / "raw_commoncatalog.jsonl"):
        sid = str(r["source_id"])
        if sid in quarantine:
            continue
        split = split_by_source.get(sid, "unassigned")
        if split == "test":
            continue
        tier = 1 if r.get("medium") in ("digital_camera", "smartphone") else 2
        p = Path(r["path"])
        if not p.exists():
            continue
        st = pixel_stats(p)
        primary, cats = categorise(r.get("subjects") or [], st, "flickr")
        records.append({
            "id": f"cc_{sid}", "path": str(p), "source": "commoncatalog_ccby", "license": r["license"],
            "license_url": r["license_url"], "attribution": r["attribution"], "source_url": r["source_url"],
            "author": r.get("author"), "group_id": r.get("group_id"), "split": split, "medium": r.get("medium"),
            "camera": r.get("camera"), "uploader_tags": r.get("subjects") or [], "monochrome_tag": r.get("monochrome_tag"),
            "tier": tier, "category_primary": primary, "categories": cats, "skin_tone_bucket": skin_bucket(st), "stats": st,
        })

    # 2) PD12M pilot (CC0)
    for r in load_jsonl(ROOT / "data" / "manifests" / "raw_pd12m.jsonl"):
        p = Path(r["path"])
        if not p.exists():
            continue
        sid = r["source_id"]
        split = split_by_source.get(sid, "unassigned")
        if split == "test":
            continue
        st = pixel_stats(p)
        primary, cats = categorise([], st, "pd12m")
        records.append({
            "id": f"pd12m_{sid}", "path": str(p), "source": "pd12m_pilot", "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/", "attribution": r.get("attribution"),
            "source_url": r["source_url"], "author": r.get("attribution"), "group_id": f"pd12m:{r.get('attribution')}",
            "split": split, "medium": "archive_photograph", "camera": None, "uploader_tags": [], "monochrome_tag": None,
            "tier": 1, "category_primary": primary, "categories": cats, "skin_tone_bucket": skin_bucket(st), "stats": st,
        })

    # 3) PD12M iNaturalist (CC0) from the photo gate
    inat = ROOT / "data" / "raw" / "photo_gate" / "pd12m_inaturalist" / "manifest_pd12m.jsonl"
    if inat.exists():
        for r in load_jsonl(inat):
            p = ROOT / "data" / r["path"]
            if not p.exists():
                continue
            st = pixel_stats(p)
            primary, cats = categorise([], st, "inaturalist")
            records.append({
                "id": f"pd12m_inat_{p.stem}", "path": str(p), "source": "pd12m_inaturalist_photo_gate", "license": "CC0-1.0",
                "license_url": r.get("licence", "https://creativecommons.org/publicdomain/zero/1.0/"),
                "attribution": "PD12M (Spawning) / iNaturalist CC0 observation", "source_url": "https://huggingface.co/datasets/Spawning/PD12M",
                "author": None, "group_id": "pd12m:inaturalist", "split": "unassigned", "photo_gate_split": r.get("split"),
                "medium": "digital_camera_or_phone", "camera": None, "uploader_tags": [], "monochrome_tag": None,
                "tier": 1, "category_primary": primary, "categories": cats, "skin_tone_bucket": skin_bucket(st), "stats": st,
            })

    tier1 = [r for r in records if r["tier"] == 1]
    summary = {
        "generated": TODAY,
        "n_total": len(records), "n_tier1": len(tier1), "n_tier2_film_or_filtered": len(records) - len(tier1),
        "max_side_px": MAX_SIDE, "n_needing_downscale": sum(r["stats"]["needs_downscale"] for r in records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_license": dict(Counter(r["license"] for r in records)),
        "by_split": dict(Counter(r["split"] for r in records)),
        "tier1_by_primary_category": dict(Counter(r["category_primary"] for r in tier1).most_common()),
        "tier1_category_membership": dict(Counter(c for r in tier1 for c in r["categories"]).most_common()),
        "tier1_skin_tone_buckets": dict(Counter(r["skin_tone_bucket"] or "none" for r in tier1)),
        "tier2_by_primary_category": dict(Counter(r["category_primary"] for r in records if r["tier"] == 2).most_common()),
        "excluded": ["project test split (133 originals)", "quarantine.yaml image + family", "met_photographs (eval-only)",
                     "artefact_film (eval-only)", "commoncatalog_film / _hd (eval-only)", "aic_photographs (needs an experiment definition)",
                     "photographer_refdb (historical reproductions)", "commons_film (eval-only)"],
        "category_method": "uploader tags (CommonCatalog subjects) + dataset nature (iNaturalist) + pixel statistics; heuristic guesses, no model, no human labels",
    }
    OUT.write_text(json.dumps({"summary": summary, "images": records}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
