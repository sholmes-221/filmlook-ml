"""Map every LUT style to a direction of the website's film catalog and write zh/en colour descriptions.

    python -m research.lut.catalog_map --out research/lut/db/catalog_map.json

The catalog (film-evaluation/filmlook/film_catalog.py, read-only) = 10 hand-written directions + one generated
direction per `subjective_profile` stock of filmlook/data/reference_knowledge.json (same slug rule as the website, so
the ids written here are the website's ids; the script verifies the count against the catalog file).
Mapping rule: exact when the preset emulates that catalog stock (name match); otherwise the closest direction by
category, speed and documented look, flagged `exact: false` with a note. Descriptions are templated from the LUT
descriptor database (research/lut/db/lut_db.json): cast of shadows / midtones / highlights, saturation, black lift,
midtone contrast -- every statement traces to a stored number.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FE = Path("/home/shanmo/misc/film-evaluation")
HANDWRITTEN = [  # id, name, reference_name, category, film_speed (from film_catalog.py, 2026-11-10)
    ("cinestill_like", "CineStill-like", "CineStill 800T", "color_negative", 800),
    ("ektar_like", "Ektar-like", "Kodak Ektar 100", "color_negative", 100),
    ("gold_like", "Gold-like", "Kodak Gold 200", "color_negative", 200),
    ("hp5_like", "HP5-like", "Ilford HP5 Plus 400", "black_and_white", 400),
    ("kodachrome_like", "Kodachrome-like", "Kodachrome 64", "color_slide", 64),
    ("portra_like", "Portra-like", "Kodak Portra 400", "color_negative", 400),
    ("pro_400h_like", "Pro 400H-like", "Fujifilm Pro 400H", "color_negative", 400),
    ("superia_like", "Superia-like", "Fujifilm Superia 400", "color_negative", 400),
    ("tri_x_like", "Tri-X-like", "Kodak Tri-X 400", "black_and_white", 400),
    ("velvia_like", "Velvia-like", "Fujifilm Velvia 50", "color_slide", 50),
]
ALIASES = {"Fuji Pro 400H", "Fuji Superia X-TRA 400", "Fuji Velvia 50", "Ilford HP5 Plus"}


def catalog() -> list[dict]:
    items = [{"id": i, "name": n, "reference_name": r, "category": c, "film_speed": s, "generated": False} for i, n, r, c, s in HANDWRITTEN]
    existing = {it["reference_name"].casefold() for it in items}
    k = json.loads((FE / "filmlook/data/reference_knowledge.json").read_text(encoding="utf-8"))
    for stock in k["stocks"]:
        if stock["name"] in ALIASES or stock["status"] != "subjective_profile" or stock["name"].casefold() in existing:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", stock["name"].lower()).strip("-")
        items.append({"id": slug.replace("-", "_") + "_like", "name": stock["name"] + "-like", "reference_name": stock["name"],
                      "category": stock["kind"], "film_speed": stock["iso"], "generated": True})
    items.sort(key=lambda it: it["name"])
    return items


# style id -> (catalog reference_name, exact, note)
MAPPING = {
    "kodak_portra_400": ("Kodak Portra 400", True, ""),
    "kodak_portra_160": ("Kodak Portra 160", True, ""),
    "kodak_ektar_100": ("Kodak Ektar 100", True, ""),
    "kodak_gold_200": ("Kodak Gold 200", True, ""),
    "kodak_color_plus_200": ("Kodak ColorPlus 200", True, ""),
    "kodak_ultramax_400": ("Kodak UltraMax 400", True, ""),
    "fujifilm_200": ("Fuji C200", True, "the preset is named 'Fujifilm 200'; Fujifilm 200 is the current C200-derived consumer stock"),
    "cinestill_50d": ("Cinestill 50D", True, ""),
    "lomochrome_92_400": ("LomoChrome Color 92", True, "the preset name 'LomoChrome 92' 400' is LomoChrome Color '92 (ISO 400)"),
    "phoenix_200": ("HARMAN Phoenix 200", True, ""),
    "candido_800t": ("CineStill 800T", False, "Candido 800T is a tungsten-balanced 800 cine-derived colour negative with no catalog entry; nearest direction by type and speed"),
    "film_one_hundred_100": ("Fuji Industrial 100", False, "Film One Hundred (ISO 100 colour negative, muted palette) has no catalog entry; nearest ISO-100 consumer colour-negative direction"),
    "lomography_color_400": ("Kodak UltraMax 400", False, "Lomography Color Negative 400 has no catalog entry; nearest ISO-400 consumer colour-negative direction"),
}


def hue_word_en(h):
    h %= 360
    # CIELAB hue angles: red ~40, orange ~65, yellow ~95, green ~135, cyan ~195, blue ~300, magenta ~330
    for lim, w in ((20, "magenta"), (50, "red"), (78, "orange"), (112, "yellow"), (145, "yellow-green"), (180, "green"), (235, "teal"), (290, "blue"), (322, "purple"), (361, "magenta")):
        if h < lim:
            return w


def hue_word_zh(h):
    h %= 360
    for lim, w in ((20, "偏品红"), (50, "偏红"), (78, "偏橙"), (112, "偏黄"), (145, "偏黄绿"), (180, "偏绿"), (235, "偏青"), (290, "偏蓝"), (322, "偏紫"), (361, "偏品红")):
        if h < lim:
            return w


def describe_style(st: dict) -> dict:
    d = st["descriptors"]
    mid, sh, hi = d["midtone_colour_bias"], d["shadow_colour_bias"], d["highlight_colour_bias"]
    sat = d["gamut_compression"]["chroma_mean_ratio"]
    black = d["clipping"]["black_lift_8bit"]
    mc = d["luminance_response"]["midtone_contrast"]
    warm = mid["b"] > 1.5 and 20 <= mid["hue_deg"] <= 110
    cool = mid["b"] < -1.5 or 180 <= mid["hue_deg"] <= 300 and mid["chroma"] > 1.5
    # English
    en = []
    if mid["chroma"] >= 1.5:
        hw = hue_word_en(mid['hue_deg'])
        en.append(f"{'Warm' if warm else ('Cool' if cool else 'Tinted')} midtones with {'an' if hw[0] in 'aeiou' else 'a'} {hw} cast")
    else:
        en.append("Neutral midtones")
    if sh["chroma"] >= 1.5:
        en.append(f"{hue_word_en(sh['hue_deg'])} shadows")
    if hi["chroma"] >= 1.5:
        en.append(f"{hue_word_en(hi['hue_deg'])} highlights")
    s1 = ", ".join(en) + "."
    sat_w = "muted colours" if sat < 0.88 else ("slightly muted colours" if sat < 0.97 else ("natural saturation" if sat < 1.05 else "boosted saturation"))
    blk = "lifted, matte blacks" if black >= 20 else ("softly lifted blacks" if black >= 8 else "deep blacks")
    con = "low midtone contrast" if mc < 0.97 else ("normal midtone contrast" if mc < 1.03 else "punchy midtone contrast")
    s2 = f"{sat_w.capitalize()}, {blk}, {con}."
    # Chinese
    zh = []
    if mid["chroma"] >= 1.5:
        zh.append(f"中间调{'偏暖' if warm else ('偏冷' if cool else '带色')}，{hue_word_zh(mid['hue_deg'])}")
    else:
        zh.append("中间调基本中性")
    if sh["chroma"] >= 1.5:
        zh.append(f"暗部{hue_word_zh(sh['hue_deg'])}")
    if hi["chroma"] >= 1.5:
        zh.append(f"亮部{hue_word_zh(hi['hue_deg'])}")
    z1 = "，".join(zh) + "。"
    sat_z = "色彩明显收敛" if sat < 0.88 else ("色彩略微收敛" if sat < 0.97 else ("饱和度接近自然" if sat < 1.05 else "饱和度有所提升"))
    blk_z = "黑位明显抬高、呈哑光感" if black >= 20 else ("黑位略微抬高" if black >= 8 else "黑位扎实")
    con_z = "中间调反差偏低" if mc < 0.97 else ("中间调反差正常" if mc < 1.03 else "中间调反差偏高")
    z2 = f"{sat_z}，{blk_z}，{con_z}。"
    return {"en": f"{s1} {s2}", "zh": f"{z1}{z2}",
            "numbers": {"midtone_bias_ab": [round(mid["a"], 1), round(mid["b"], 1)], "shadow_bias_ab": [round(sh["a"], 1), round(sh["b"], 1)],
                        "highlight_bias_ab": [round(hi["a"], 1), round(hi["b"], 1)], "chroma_ratio": round(sat, 3), "black_lift_8bit": round(black, 1),
                        "midtone_contrast": round(mc, 3)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="research/lut/db/lut_db.json")
    ap.add_argument("--out", default="research/lut/db/catalog_map.json")
    a = ap.parse_args()
    cat = catalog()
    by_ref = {it["reference_name"].casefold(): it for it in cat}
    src = (FE / "filmlook/film_catalog.py").read_text()
    n_hand = src.count("FilmAesthetic(") - 1
    assert n_hand == len(HANDWRITTEN), f"hand-written entries changed in film_catalog.py ({n_hand})"
    db = json.loads(Path(a.db).read_text())
    out = {"catalog_size": len(cat), "catalog_source": "film-evaluation/filmlook/film_catalog.py + filmlook/data/reference_knowledge.json (read 2026-11-10)",
           "label_basis": "Every style is a LUT-derived aesthetic (an approximate re-implementation of a Lightroom preset); the catalog direction is the "
                          "closest named look, not a statement about the film that exposed the photo.", "styles": {}}
    for st in db["styles"]:
        ref, exact, note = MAPPING[st["id"]]
        it = by_ref[ref.casefold()]
        out["styles"][st["id"]] = {"style_name": st["name"], "catalog_id": it["id"], "catalog_name": it["name"], "catalog_reference_name": it["reference_name"],
                                   "category": it["category"], "film_speed": it["film_speed"], "exact": exact, "note": note or None,
                                   "label_basis": "LUT-derived aesthetic", "description": describe_style(st)}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"catalog {len(cat)} directions; mapped {len(out['styles'])} styles -> {a.out}")
    for k, v in out["styles"].items():
        print(f"{k:22} -> {v['catalog_id']:28} exact={v['exact']}\n   en: {v['description']['en']}\n   zh: {v['description']['zh']}")


if __name__ == "__main__":
    main()
