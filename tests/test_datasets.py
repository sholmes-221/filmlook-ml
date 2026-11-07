import json

import numpy as np
import pytest
from PIL import Image

from filmlook_ml.datasets.preprocess import (ImageValidationError, group_split, hamming, load_image, phash,
                                             prepare)
from filmlook_ml.datasets.registry import LicenseError, Registry, normalize_license


def test_corrupt_image_rejected(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    with pytest.raises(ImageValidationError):
        load_image(p)


def test_truncated_image_rejected(tmp_path, fixture_dir):
    src = (fixture_dir / "scene_0.png").read_bytes()
    p = tmp_path / "trunc.png"
    p.write_bytes(src[: len(src) // 3])
    with pytest.raises(ImageValidationError):
        load_image(p)


def test_too_small_rejected(tmp_path):
    p = tmp_path / "small.png"
    Image.new("RGB", (100, 100)).save(p)
    with pytest.raises(ImageValidationError):
        load_image(p)


def test_exif_orientation_applied(tmp_path):
    im = Image.new("RGB", (400, 300), (10, 20, 30))
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    p = tmp_path / "o.jpg"
    im.save(p, exif=exif)
    assert load_image(p).size == (300, 400)


def test_grayscale_and_rgba_normalised(tmp_path):
    for mode in ("L", "RGBA"):
        p = tmp_path / f"{mode}.png"
        Image.new(mode, (300, 300)).save(p)
        assert load_image(p).mode == "RGB"


def test_registry_allowlist():
    reg = Registry.load()
    assert reg.check_record("synthetic_fixture", "synthetic")[0]
    assert reg.check_record("pd12m_pilot", "CC0-1.0")[0]
    assert not reg.check_record("pd12m_pilot", "CC-BY-NC-4.0")[0]       # denylisted license
    assert not reg.check_record("ava", "unknown")[0]                    # quarantined dataset
    assert not reg.check_record("unsplash_full", "Unsplash-Full-Dataset-Terms")[0]
    assert not reg.check_record("not_registered", "CC0-1.0")[0]
    with pytest.raises(LicenseError):
        reg.require("pd12m_pilot", None)


def test_normalize_license():
    assert normalize_license("https://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"
    assert normalize_license("https://creativecommons.org/publicdomain/mark/1.0/") == "PDM-1.0"
    assert normalize_license("https://creativecommons.org/licenses/by-nc/4.0/") == "CC-BY-NC-4.0"


def test_prepare_quarantines_and_dedupes(tmp_path, raw_records):
    recs = list(raw_records)
    recs.append({**raw_records[0], "source_id": "exact_dup"})                       # exact duplicate
    near = tmp_path / "near.jpg"
    Image.open(raw_records[1]["path"]).save(near, quality=85)                      # near duplicate
    recs.append({**raw_records[1], "path": str(near), "source_id": "near_dup"})
    recs.append({**raw_records[2], "dataset_id": "ava", "license": "unknown"})     # license fail
    man, rej = prepare(recs, Registry.load(), tmp_path / "out", ["posterized", "film_emulation"])
    reasons = sorted(r["reason"].split(":")[0].split(" ")[0] for r in rej)
    assert reasons == ["exact", "license", "near"]
    assert len(man) == len(raw_records) * 3
    fam = {}
    for r in man:
        fam.setdefault(r["family_id"], set()).add(r["edit_type"])
    assert all(v == {"original", "posterized", "film_emulation"} for v in fam.values())


def test_phash_near_duplicate(fixture_dir):
    a = Image.open(fixture_dir / "scene_0.png")
    b = a.resize((320, 224))
    c = Image.open(fixture_dir / "scene_3.png")
    assert hamming(phash(a), phash(b)) <= 4
    assert hamming(phash(a), phash(c)) > 10


def test_group_split_no_family_leak():
    man = [{"image_id": f"{f}_{e}", "family_id": f} for f in range(30) for e in ("orig", "hdr", "film")]
    split = group_split(man, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    per_family = {}
    for r in man:
        per_family.setdefault(r["family_id"], set()).add(split[r["image_id"]])
    assert all(len(s) == 1 for s in per_family.values())
    assert {"train", "val", "test"} <= set(split.values())
    assert group_split(man, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1) == split  # deterministic


def test_large_jpeg_bounded_decode(tmp_path, fixture_dir):
    big = Image.open(fixture_dir / "scene_0.png").resize((4000, 2800))
    p = tmp_path / "big.jpg"
    big.save(p, quality=90)
    im = load_image(p, max_side=1280)
    assert max(im.size) == 1280 and im.mode == "RGB"
    assert load_image(p).size == (4000, 2800)  # no max_side -> untouched


def test_decompression_bomb_guard(tmp_path, monkeypatch):
    import filmlook_infer.imageio as pp
    monkeypatch.setattr(pp, "MAX_PIXELS", 1000)
    p = tmp_path / "x.png"
    Image.new("RGB", (300, 300)).save(p)
    with pytest.raises(ImageValidationError):
        pp.load_image(p)


def test_prepare_parallel_matches_sequential(tmp_path, raw_records):
    from filmlook_ml.datasets.edits import EDITS
    from filmlook_ml.datasets.preprocess import prepare
    from filmlook_ml.datasets.registry import Registry
    recs = [{**r, "group_id": "flickr:u1", "medium": "film_scan"} for r in raw_records[:3]]
    a, _ = prepare(recs, Registry.load(), tmp_path / "a", list(EDITS)[:2], max_side=512, workers=1)
    b, _ = prepare(recs, Registry.load(), tmp_path / "b", list(EDITS)[:2], max_side=512, workers=2)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "path"} for r in rows]  # noqa: E731
    assert strip(a) == strip(b) and len(a) == 9
    assert all(r["group_id"] == "flickr:u1" and r["medium"] == "film_scan" for r in a)
