"""Photographer reference database: rights rules, attribution filter, removal, renditions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from filmlook_ml.refdb import manifest, rights
from filmlook_ml.refdb.roster import BY_ID, ROSTER
from filmlook_ml.refdb.sources import PORTRAIT_OF, SKIP_CAT, SKIP_FILE, attributed_to

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_rights_classes_and_eligibility():
    assert rights.classify_commons("pd", "Public domain", "False") == "public_domain"
    assert rights.classify_commons("cc0", "CC0", "False") == "cc0"
    assert rights.classify_commons("cc-by-sa-4.0", "CC BY-SA 4.0", "True") == "licensed"
    assert rights.classify_commons("cc-by-nc-2.0", "CC BY-NC 2.0", "True") == "restricted"
    assert rights.classify_commons("", "", "") == "unverified"
    assert rights.classify_museum(True) == "cc0" and rights.classify_museum(False) == "restricted"
    assert rights.eligible("public_domain") and rights.eligible("cc0")
    assert not any(rights.eligible(c) for c in ("licensed", "unverified", "restricted"))  # Only PD and CC0 are processed.


def test_magnum_and_living_photographers_are_never_collected():
    for pid in ("henri_cartier_bresson", "robert_capa", "david_seymour", "george_rodger", "masahisa_fukase", "rinko_kawauchi", "saul_leiter", "daido_moriyama"):
        assert BY_ID[pid].status == "metadata_only" and not BY_ID[pid].commons and not BY_ID[pid].aic


def test_attribution_and_portrait_filters():
    lange = BY_ID["dorothea_lange"]
    assert attributed_to(lange, "Lange, Dorothea, 1895-1965, photographer")
    assert not attributed_to(lange, "Rondal Partridge")  # A portrait OF Lange by someone else is not her work.
    m = PORTRAIT_OF.search("Portrait of Dorothea Lange with her Graflex")
    assert m and "lange" in m.group(2).lower()
    assert SKIP_CAT.search("Portraits of Dorothea Lange") and SKIP_CAT.search("Exhibitions of Walker Evans")
    assert not SKIP_CAT.search("Farm Security Administration photographs by Dorothea Lange")
    assert SKIP_FILE.search("Lange signature.jpg") and SKIP_FILE.search("Migrant Mother (retouched).jpg")


def test_removal_deletes_file_and_embeddings_and_is_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "IMAGES", tmp_path / "img")
    db = tmp_path / "m.sqlite"
    with manifest.connect(db) as c:
        manifest.upsert(c, {"work_id": "commons:1", "source": "commons", "photographer_id": "dorothea_lange", "source_url": "u", "rights_class": "public_domain",
                            "rights_status": "eligible", "retrieved_at": manifest.now()})
        p = manifest.image_path("commons:1"); p.parent.mkdir(parents=True); p.write_bytes(b"x")
        c.execute("INSERT INTO embeddings VALUES ('commons:1', 'dinov2s_cls_224', 'v', 384, ?, 't')", (np.zeros(384, np.float32).tobytes(),))
        manifest.remove(c, "commons:1", "rights holder request")
        assert not p.exists() and c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
        assert c.execute("SELECT rights_status FROM works").fetchone()[0] == "removed"
        assert c.execute("SELECT reason FROM removals").fetchone()[0] == "rights holder request"
        manifest.upsert(c, {"work_id": "commons:1", "source": "commons", "photographer_id": "dorothea_lange", "source_url": "u", "rights_class": "public_domain",
                            "rights_status": "eligible", "retrieved_at": manifest.now()})
        assert c.execute("SELECT rights_status FROM works").fetchone()[0] == "removed"  # Rediscovery never resurrects it.


def test_unscaled_originals_are_requested_as_standard_thumbnails():
    import refdb_download as d
    urls = d.renditions("https://upload.wikimedia.org/wikipedia/commons/3/33/A_b.jpg?utm_source=x")
    assert urls[0] == "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/A_b.jpg/960px-A_b.jpg" and len(urls) == 3
    assert d.renditions("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/A_b.jpg/1280px-A_b.jpg") == ["https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/A_b.jpg/1280px-A_b.jpg"]


def test_roster_ids_are_unique():
    assert len({e.id for e in ROSTER}) == len(ROSTER)
