#!/usr/bin/env python
"""Composite images for the photo gate, plus real UI screenshots of FilmAnalysis's own pages.

  python scripts/photo_gate_composites.py [--n 120] [--site http://127.0.0.1:8766]

Training composites (family ``a``) are built from *colour* CommonCatalog photographs; the existing hard-case test set
(photo_gate_make_synthetic.py) uses black-and-white FSA scans, so the test also measures transfer across photo types:
- poster_with_photo_a (label 0): a photograph used as a poster background with large display text and shapes;
- ui_with_photo_a (label 0): an app/web UI in which a photograph is one card among text and controls;
- photo_with_caption_a (label 1): a photograph with a small caption burned in (must stay a photograph).
Test set ``ui_real`` (label 0): screenshots of FilmAnalysis's own pages (our own content) at phone and desktop widths.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import photo_gate_make_synthetic as syn  # noqa: E402

from filmlook_ml import paths as P  # noqa: E402

OUT = P.DATA / "raw" / "photo_gate" / "composites"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); ap.add_argument("--site", default="")
    a = ap.parse_args()
    rng = random.Random(7)
    photos = sorted(glob.glob(str(P.DATA / "raw" / "commoncatalog" / "**" / "*.jp*g"), recursive=True))
    rng.shuffle(photos)
    OUT.mkdir(parents=True, exist_ok=True); man, jobs = [], []
    work = Path.home() / "snap" / "firefox" / "common" / "photo-gate"; work.mkdir(parents=True, exist_ok=True)
    for i, ph in enumerate(photos[: a.n]):
        p = OUT / f"poster_with_photo_a_{i:03d}.jpg"
        if not p.exists():
            syn.poster(rng, "a", Image.open(ph)).save(p, quality=90)
        man.append({"path": str(p.relative_to(P.DATA)), "kind": "poster_with_photo", "family": "a", "label": 0, "split": "train", "generator": "local:poster+commoncatalog"})
    for i, ph in enumerate(photos[a.n: 2 * a.n]):
        p = OUT / f"photo_with_caption_a_{i:03d}.jpg"
        if not p.exists():
            im = Image.open(ph).convert("RGB"); im.thumbnail((1280, 1280)); d = ImageDraw.Draw(im)
            d.text((16, im.height - 56), syn.words(rng, rng.randint(2, 5), rng.random() < 0.5), fill=(255, 255, 255), font=syn.font(rng, rng.randint(22, 36), True), stroke_width=2, stroke_fill=(0, 0, 0))
            im.save(p, quality=90)
        man.append({"path": str(p.relative_to(P.DATA)), "kind": "photo_with_caption", "family": "a", "label": 1, "split": "train", "generator": "local:commoncatalog+caption"})
    for i, ph in enumerate(photos[2 * a.n: 3 * a.n]):
        p = OUT / f"ui_with_photo_a_{i:03d}.jpg"
        if not p.exists():
            shutil.copy(ph, work / f"cphoto{i}.jpg")
            jobs.append((syn.ui_html(rng, rng.choice("ab"), photo_src=f"cphoto{i}.jpg"), rng.choice(["500x1000", "1280x800"]), p))
        man.append({"path": str(p.relative_to(P.DATA)), "kind": "ui_with_photo", "family": "a", "label": 0, "split": "train", "generator": "local:ui+commoncatalog"})
    if a.site:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service
        import io
        import time
        o = Options(); o.add_argument("-headless")
        dr = webdriver.Firefox(service=Service("/snap/bin/firefox.geckodriver"), options=o)
        pages = ["/", "/about/", "/pricing/", "/privacy/", "/guides/", "/guides/highlight-roll-off/", "/guides/developing-a-visual-style/", "/guides/film-grain-vs-digital-noise/"]
        try:
            for path in pages:
                for w, h in ((500, 1000), (1280, 900)):
                    for theme in ("light", "dark"):
                        p = OUT / f"ui_real_{path.strip('/').replace('/', '_') or 'home'}_{w}_{theme}.jpg"
                        if not p.exists():
                            dr.set_window_size(w, h); dr.get(a.site + "/about/"); dr.execute_script("localStorage.setItem('filmlook_theme', arguments[0])", theme)
                            dr.get(a.site + path); time.sleep(2)
                            Image.open(io.BytesIO(dr.get_screenshot_as_png())).convert("RGB").save(p, quality=90)
                        man.append({"path": str(p.relative_to(P.DATA)), "kind": "ui_real", "family": "real", "label": 0, "split": "test", "generator": "filmanalysis site screenshot"})
        finally:
            dr.quit()
    if jobs:
        syn.screenshots(jobs)
    (OUT / "manifest_composites.jsonl").write_text("\n".join(json.dumps(m) for m in man) + "\n")
    print(len(man))


if __name__ == "__main__":
    main()
