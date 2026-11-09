#!/usr/bin/env python
"""Locally rendered non-photograph images for the photo gate: documents, charts, posters, UI screenshots.

  python scripts/photo_gate_make_synthetic.py [--n 400] [--seed 0]

Everything is generated here (fonts installed on this system, random text, random data), so the images carry no
third-party rights. Two *families* per kind are produced, ``a`` and ``b``, with different layouts, fonts and
palettes; training uses family ``a`` and the held-out test uses family ``b``, so the test measures generalisation to
an unseen generator rather than memorisation of one.

Hard cases written separately (evaluation only, never training):
- ``ui_with_photo``: UI screenshots that contain a real public-domain photograph (refdb) in a card;
- ``poster_with_photo``: posters whose background is a photograph with large text;
- ``photo_with_overlay_text``: a photograph with a caption burned in (should stay a photograph).
Each file is listed in data/raw/photo_gate/manifest.jsonl with kind, family, split and generator.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import random
import string
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmlook_ml import paths as P  # noqa: E402

OUT = P.DATA / "raw" / "photo_gate"
ZH_WORDS = "摄影 胶片 风格 光线 构图 色彩 分析 上传 照片 设置 账号 登录 注册 今天 明天 价格 方案 订单 消息 好友 关注 收藏 评论 分享 发现 城市 街头 黑白 颗粒 夜晚 咖啡 地图 天气 时间 设置 通知 隐私 帮助".split()
EN_WORDS = ("photo film style light colour frame upload account settings login price plan order message friend follow share discover city street "
            "black white grain night coffee map weather time notice privacy help report total revenue growth quarter user data model score").split()


def fonts() -> tuple[list[str], list[str]]:
    files = sorted(set(glob.glob("/usr/share/fonts/**/*.tt[fc]", recursive=True) + glob.glob("/usr/share/fonts/**/*.otf", recursive=True)))
    cjk = [f for f in files if "CJK" in f or "Noto Sans CJK" in f]
    import re
    latin = [f for f in files if re.search(r"/(DejaVu|Liberation|Ubuntu|NotoSans-|NotoSerif-|NotoSansMono-|NotoSansDisplay-|FreeSans|FreeSerif|FreeMono)[^/]*$", f)]
    return latin, cjk


LATIN, CJK = fonts()


def font(rng: random.Random, size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    pool = CJK if cjk and CJK else LATIN
    for _ in range(5):
        try:
            return ImageFont.truetype(rng.choice(pool), size)
        except OSError:
            continue
    return ImageFont.load_default()


def words(rng: random.Random, n: int, zh: bool) -> str:
    if zh:
        return "".join(rng.choice(ZH_WORDS) for _ in range(n))
    return " ".join(rng.choice(EN_WORDS) for _ in range(n)).capitalize()


def pastel(rng, light=True):
    base = 200 if light else 20
    return tuple(rng.randint(base, base + 55) for _ in range(3))


def document(rng: random.Random, family: str) -> Image.Image:
    w, h = (rng.choice([(900, 1200), (1080, 1440), (800, 1100)]) if family == "a" else rng.choice([(1200, 900), (1240, 1754), (600, 900)]))
    bg = (255, 255, 255) if family == "a" else pastel(rng)
    im = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(im)
    zh = rng.random() < 0.5; y = rng.randint(40, 90)
    d.text((60, y), words(rng, 3, zh), fill=(20, 20, 20), font=font(rng, rng.randint(34, 56), zh)); y += 90
    size = rng.randint(18, 30); f = font(rng, size, zh)
    while y < h - 80:
        if family == "b" and rng.random() < 0.15:  # table rows / receipt lines
            d.line([(60, y), (w - 60, y)], fill=(160, 160, 160), width=1)
        d.text((60, y), words(rng, rng.randint(6, 14), zh), fill=(30, 30, 30), font=f); y += int(size * 1.6)
        if rng.random() < 0.1:
            y += size
    if rng.random() < 0.3:
        im = im.rotate(rng.uniform(-4, 4), expand=True, fillcolor=(90, 90, 90))  # a scanned page
    return im


def chart(rng: random.Random, family: str) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = ["default", "ggplot", "bmh"] if family == "a" else ["seaborn-v0_8-whitegrid", "dark_background", "fivethirtyeight", "Solarize_Light2"]
    with plt.style.context(rng.choice(styles)):
        fig, ax = plt.subplots(figsize=(rng.uniform(5, 9), rng.uniform(3.5, 6)), dpi=110)
        kind = rng.choice(["line", "bar", "scatter", "pie", "area"] if family == "a" else ["hist", "barh", "step", "pie", "box"])
        x = np.arange(rng.randint(5, 30)); y = np.cumsum(np.random.default_rng(rng.randint(0, 10**6)).normal(size=len(x)))
        if kind == "line": ax.plot(x, y); ax.plot(x, y * 0.6)
        elif kind == "bar": ax.bar(x, np.abs(y))
        elif kind == "scatter": ax.scatter(y, np.roll(y, 3), s=12)
        elif kind == "area": ax.fill_between(x, np.abs(y))
        elif kind == "hist": ax.hist(np.random.default_rng(rng.randint(0, 99)).normal(size=500), bins=rng.randint(10, 40))
        elif kind == "barh": ax.barh(x[:10], np.abs(y[:10]))
        elif kind == "step": ax.step(x, y)
        elif kind == "box": ax.boxplot([np.random.default_rng(i).normal(size=60) for i in range(rng.randint(3, 7))])
        else: ax.pie(np.abs(y[:5]) + 0.1)
        ax.set_title(words(rng, 3, False))
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return Image.open(buf).convert("RGB")


def poster(rng: random.Random, family: str, photo: Image.Image | None = None) -> Image.Image:
    w, h = (1080, 1520) if family == "a" else rng.choice([(1240, 1754), (900, 1600)])
    if photo is not None:
        im = photo.convert("RGB").resize((w, h))
    else:
        top, bot = pastel(rng, rng.random() < 0.5), pastel(rng, rng.random() < 0.5)
        g = np.linspace(0, 1, h)[:, None, None]
        im = Image.fromarray((np.array(top) * (1 - g) + np.array(bot) * g).astype("uint8").repeat(w, axis=1))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(2, 6) if family == "a" else rng.randint(4, 10)):
        x0, y0 = rng.randint(0, w), rng.randint(0, h); s = rng.randint(80, 500); col = tuple(rng.randint(0, 255) for _ in range(3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x0 + s, y0 + s], fill=col)
    zh = rng.random() < 0.5
    d.text((60, rng.randint(80, 300)), words(rng, 2, zh), fill=(255, 255, 255), font=font(rng, rng.randint(90, 160), zh), stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((60, h - 260), words(rng, 6, zh), fill=(20, 20, 20), font=font(rng, rng.randint(36, 60), zh))
    return im


UI_TEMPLATES = {
    "a": """<html><body style="margin:0;font-family:{font};background:{bg}"><div style="height:56px;background:{bar};color:#fff;display:flex;align-items:center;padding:0 16px;font-size:20px">{title}</div>
<div style="padding:16px">{cards}</div></body></html>""",
    "b": """<html><body style="margin:0;font-family:{font};background:{bg};display:flex"><div style="width:220px;background:{bar};color:#eee;padding:16px;min-height:100vh">{side}</div>
<div style="flex:1;padding:24px"><h2>{title}</h2>{cards}</div></body></html>""",
}


def ui_html(rng: random.Random, family: str, photo_src: str | None = None) -> str:
    zh = rng.random() < 0.5
    cards = []
    for _ in range(rng.randint(3, 8)):
        kind = rng.choice(["card", "chat", "table", "form", "code"])
        if kind == "card":
            img = f'<img src="{photo_src}" style="width:100%;max-height:260px;object-fit:cover;border-radius:8px">' if photo_src and rng.random() < 0.6 else ""
            cards.append(f'<div style="background:#fff;border-radius:12px;padding:14px;margin:10px 0;box-shadow:0 1px 4px #0002">{img}<b>{words(rng, 3, zh)}</b><p>{words(rng, 12, zh)}</p><button style="padding:6px 14px">{words(rng, 1, zh)}</button></div>')
        elif kind == "chat":
            cards.append("".join(f'<div style="max-width:70%;margin:6px {"auto 6px 0" if i % 2 else "0 6px auto"};background:{"#9be27f" if i % 2 else "#fff"};padding:8px 12px;border-radius:14px">{words(rng, rng.randint(3, 10), zh)}</div>' for i in range(rng.randint(2, 6))))
        elif kind == "table":
            rows = "".join("<tr>" + "".join(f"<td style='border:1px solid #ccc;padding:4px'>{rng.randint(0, 9999)}</td>" for _ in range(4)) + "</tr>" for _ in range(rng.randint(3, 8)))
            cards.append(f"<table style='border-collapse:collapse;background:#fff;margin:10px 0'>{rows}</table>")
        elif kind == "form":
            cards.append(f'<div style="margin:10px 0">{words(rng, 2, zh)}<br><input style="width:80%;padding:8px" value="{words(rng, 2, zh)}"></div>')
        else:
            code = "<br>".join("&nbsp;" * rng.randint(0, 8) + "".join(rng.choice(string.ascii_lowercase + "(){}=;.") for _ in range(rng.randint(10, 50))) for _ in range(rng.randint(4, 12)))
            cards.append(f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:8px">{code}</pre>')
    side = "".join(f"<p>{words(rng, 2, zh)}</p>" for _ in range(8))
    return UI_TEMPLATES[family].format(font=rng.choice(["sans-serif", "serif", "monospace", "Noto Sans CJK SC"]), bg=rng.choice(["#f4f4f7", "#ffffff", "#eef3ee", "#1f2124"]),
                                       bar=rng.choice(["#2b5797", "#333", "#7a3e48", "#0f7b6c"]), title=words(rng, 3, zh), cards="".join(cards), side=side)


def screenshots(jobs: list[tuple[str, str, Path]]) -> None:
    """Render HTML jobs (html, size 'WxH', destination) with headless Firefox (snap profile dir for file access)."""
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    work = Path.home() / "snap" / "firefox" / "common" / "photo-gate"; work.mkdir(parents=True, exist_ok=True)
    o = Options(); o.add_argument("-headless")
    d = webdriver.Firefox(service=Service("/snap/bin/firefox.geckodriver"), options=o)
    try:
        for i, (html, size, dst) in enumerate(jobs):
            w, h = map(int, size.split("x"))
            d.set_window_size(w, h)
            page = work / f"p{i}.html"; page.write_text(html, encoding="utf-8")
            d.get(page.as_uri())
            png = d.get_screenshot_as_png()
            Image.open(io.BytesIO(png)).convert("RGB").save(dst, quality=92)
            page.unlink()
    finally:
        d.quit()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    photos = sorted(glob.glob(str(P.DATA / "raw" / "refdb" / "commons" / "*.jpg")))[:300]
    man = []
    for kind, fn in (("document", document), ("chart", chart), ("poster", poster)):
        for fam in ("a", "b"):
            out = OUT / kind / fam; out.mkdir(parents=True, exist_ok=True)
            for i in range(a.n if fam == "a" else a.n // 3):
                p = out / f"{kind}_{fam}_{i:04d}.jpg"
                if not p.exists():
                    fn(rng, fam).convert("RGB").save(p, quality=rng.choice([75, 85, 92]))
                man.append({"path": str(p.relative_to(P.DATA)), "kind": kind, "family": fam, "label": 0, "split": "train" if fam == "a" else "test", "generator": f"local:{kind}_{fam}"})
    jobs = []
    work = Path.home() / "snap" / "firefox" / "common" / "photo-gate"; work.mkdir(parents=True, exist_ok=True)
    for fam in ("a", "b"):
        out = OUT / "ui" / fam; out.mkdir(parents=True, exist_ok=True)
        for i in range(a.n if fam == "a" else a.n // 3):
            p = out / f"ui_{fam}_{i:04d}.jpg"
            size = rng.choice(["500x1000", "540x1100"] if fam == "a" else ["1280x800", "1440x900", "1024x768"])
            if not p.exists():
                jobs.append((ui_html(rng, fam), size, p))
            man.append({"path": str(p.relative_to(P.DATA)), "kind": "ui", "family": fam, "label": 0, "split": "train" if fam == "a" else "test", "generator": f"local:ui_{fam}"})
    # Hard cases (evaluation only).
    hard = OUT / "hard"; hard.mkdir(parents=True, exist_ok=True)
    for i, ph in enumerate(photos[:80]):
        dst = hard / f"ui_with_photo_{i:03d}.jpg"
        if not dst.exists():
            import shutil
            shutil.copy(ph, work / f"photo{i}.jpg")
            jobs.append((ui_html(rng, "a", photo_src=f"photo{i}.jpg"), rng.choice(["500x1000", "1280x800"]), dst))
        man.append({"path": str(dst.relative_to(P.DATA)), "kind": "ui_with_photo", "family": "hard", "label": 0, "split": "hard", "generator": "local:ui+refdb_photo"})
    for i, ph in enumerate(photos[80:140]):
        dst = hard / f"poster_with_photo_{i:03d}.jpg"
        if not dst.exists():
            poster(rng, "b", Image.open(ph)).save(dst, quality=90)
        man.append({"path": str(dst.relative_to(P.DATA)), "kind": "poster_with_photo", "family": "hard", "label": 0, "split": "hard", "generator": "local:poster+refdb_photo"})
    for i, ph in enumerate(photos[140:200]):
        dst = hard / f"photo_with_caption_{i:03d}.jpg"
        if not dst.exists():
            im = Image.open(ph).convert("RGB"); d = ImageDraw.Draw(im)
            d.text((20, im.height - 70), words(rng, 4, rng.random() < 0.5), fill=(255, 255, 255), font=font(rng, 40, True), stroke_width=2, stroke_fill=(0, 0, 0))
            im.save(dst, quality=90)
        man.append({"path": str(dst.relative_to(P.DATA)), "kind": "photo_with_caption", "family": "hard", "label": 1, "split": "hard", "generator": "local:refdb_photo+caption"})
    if jobs:
        screenshots(jobs)
    (OUT / "manifest_synthetic.jsonl").write_text("\n".join(json.dumps(m) for m in man) + "\n")
    print(len(man), "synthetic records;", sum(1 for m in man if m["split"] == "train"), "train")


if __name__ == "__main__":
    main()
