"""Dataset registry integrity and generator provenance records."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "data_registry" / "film_datasets.json"


def test_registry_entries_carry_license_evidence_and_status():
    reg = json.loads(REG.read_text())
    assert reg["datasets"], "registry is empty"
    for d in reg["datasets"]:
        assert d["id"] and d["source_url"].startswith("http"), d["id"]
        assert d["status"] in {"eligible", "not_eligible", "not_pursued", "follow_up"}, d["id"]
        if d["status"] == "eligible":
            assert d["license"]["name"] and d.get("commercial_ml_training"), d["id"]
            assert d.get("provenance"), d["id"]


def test_not_eligible_datasets_are_never_cached():
    reg = json.loads(REG.read_text())
    cache = ROOT / "data" / "hf_cache"
    if not cache.exists():
        pytest.skip("no HF cache")
    cached = {p.name.lower() for p in cache.glob("datasets--*")}
    for d in reg["datasets"]:
        if d["status"] != "eligible" and "huggingface.co/datasets/" in d["source_url"]:
            repo = d["source_url"].split("huggingface.co/datasets/")[1].strip("/")
            assert f"datasets--{repo.replace('/', '--')}".lower() not in cached, f"{d['id']} is cached but not eligible"


def test_sdxl_generation_records_full_provenance():
    ann = ROOT / "data" / "synthetic" / "sdxl_i2i_v1" / "annotations.jsonl"
    if not ann.exists():
        pytest.skip("no SDXL outputs")
    rows = [json.loads(l) for l in ann.read_text().splitlines()]
    for r in rows[:50]:
        assert r["is_real_film_scan"] is False and r["film_stock_claim"] is None
        assert r["label_source"] == "diffusion_image_to_image"
        assert "stabilityai/stable-diffusion-xl-base-1.0" in r["models"]
        for m in r["models"].values():
            assert m["revision"] and m["license"], r["variant_id"]
        for k in ("prompt", "negative_prompt", "seed", "strength", "steps", "guidance_scale", "split", "group_id", "sha256"):
            assert k in r, k
        assert r["source_license"] in ("CC-BY-2.0", "CC0-1.0", "PDM-1.0"), r["source_license"]
