"""Film-stock claims from free-form uploader tags (Flickr/CommonCatalog) or category names.

A claim is only ever *uploader-asserted*: this module never looks at pixels. Unknown stays unknown (no negative
label); a brand without a stock line yields `film_brand` only.
"""
from __future__ import annotations

import re
import urllib.parse

# (regex on the normalised tag, canonical stock, brand, film_type). Order matters: specific lines before generic brand lines.
_LINES = [
    (r"portra", "kodak_portra", "kodak", "colour_negative"), (r"ektar", "kodak_ektar", "kodak", "colour_negative"),
    (r"ultramax|kodak ?max|kodakmax", "kodak_ultramax", "kodak", "colour_negative"), (r"colou?r ?plus", "kodak_colorplus", "kodak", "colour_negative"),
    (r"kodak ?gold|\bgold ?(100|200|400)\b", "kodak_gold", "kodak", "colour_negative"), (r"pro ?image", "kodak_proimage", "kodak", "colour_negative"),
    (r"kodak ?(400|160) ?(vc|nc)|\b(160|400)(vc|nc)\b", "kodak_portra_vc_nc", "kodak", "colour_negative"),
    (r"ektachrome|\be100\b", "kodak_ektachrome", "kodak", "colour_reversal"), (r"kodachrome", "kodak_kodachrome", "kodak", "colour_reversal"),
    (r"tri-?x|\btrix", "kodak_tri_x", "kodak", "bw_negative"), (r"t-?max|tmax|tmx|tmy", "kodak_tmax", "kodak", "bw_negative"),
    (r"plus-?x", "kodak_plus_x", "kodak", "bw_negative"), (r"vision ?3|5219|5207|5203|250d|500t|50d(?!$)", "kodak_vision3", "kodak", "colour_negative_cine"),
    (r"advantix", "kodak_advantix", "kodak", "colour_negative"),
    (r"superia|x-?tra|xtra", "fuji_superia", "fuji", "colour_negative"), (r"\bc ?200\b|fujicolor ?200|fuji ?200", "fuji_c200", "fuji", "colour_negative"),
    (r"pro ?400 ?h|400h", "fuji_pro400h", "fuji", "colour_negative"), (r"pro ?160 ?(s|ns|c)|160ns|160s\b", "fuji_pro160", "fuji", "colour_negative"),
    (r"natura", "fuji_natura", "fuji", "colour_negative"), (r"reala", "fuji_reala", "fuji", "colour_negative"), (r"velvia", "fuji_velvia", "fuji", "colour_reversal"),
    (r"provia", "fuji_provia", "fuji", "colour_reversal"), (r"astia", "fuji_astia", "fuji", "colour_reversal"), (r"sensia", "fuji_sensia", "fuji", "colour_reversal"),
    (r"fujichrome", "fuji_fujichrome", "fuji", "colour_reversal"), (r"acros", "fuji_acros", "fuji", "bw_negative"), (r"neopan", "fuji_neopan", "fuji", "bw_negative"),
    (r"hp5", "ilford_hp5", "ilford", "bw_negative"), (r"fp4", "ilford_fp4", "ilford", "bw_negative"), (r"delta ?(100|400|3200)|ilford ?delta", "ilford_delta", "ilford", "bw_negative"),
    (r"pan ?f\b|panf", "ilford_panf", "ilford", "bw_negative"), (r"xp2", "ilford_xp2", "ilford", "bw_negative_c41"), (r"\bsfx\b", "ilford_sfx", "ilford", "bw_negative"),
    (r"ilford ?(400|100|125)", "ilford_unspecified", "ilford", "bw_negative"),
    (r"vista ?(100|200|400|800)?|agfa ?vista", "agfa_vista", "agfa", "colour_negative"), (r"\bapx\b|agfapan", "agfa_apx", "agfa", "bw_negative"),
    (r"ct ?precisa|agfachrome", "agfa_ctprecisa", "agfa", "colour_reversal"), (r"agfa ?(ultra|optima)", "agfa_optima", "agfa", "colour_negative"),
    (r"lomochrome ?purple", "lomography_lomochrome_purple", "lomography", "colour_negative_special"), (r"lomochrome", "lomography_lomochrome", "lomography", "colour_negative_special"),
    (r"lady ?grey|earl ?grey", "lomography_bw", "lomography", "bw_negative"), (r"lomo(graphy)? ?(cn|colou?r ?negative|colou?r) ?(100|400|800)|lomo ?(100|400|800)\b", "lomography_colour_negative", "lomography", "colour_negative"),
    (r"cinestill ?800|800t", "cinestill_800t", "cinestill", "colour_negative_cine"), (r"cinestill ?50|50d\b", "cinestill_50d", "cinestill", "colour_negative_cine"),
    (r"fomapan", "foma_fomapan", "foma", "bw_negative"), (r"kentmere", "kentmere", "kentmere", "bw_negative"), (r"rollei ?(retro|rpx|superpan|infrared)", "rollei_bw", "rollei", "bw_negative"),
    (r"konica ?(centuria|vx|minolta)", "konica_colour", "konica", "colour_negative"), (r"ferrania", "ferrania", "ferrania", "colour_reversal"),
    (r"\badox\b", "adox", "adox", "bw_negative"), (r"bergger", "bergger", "bergger", "bw_negative"), (r"\befke\b", "efke", "efke", "bw_negative"), (r"svema", "svema", "svema", "unknown"),
]
_BRANDS = [(r"kodak", "kodak"), (r"fuji", "fuji"), (r"ilford", "ilford"), (r"agfa", "agfa"), (r"lomo", "lomography"), (r"cinestill", "cinestill"), (r"foma", "foma"),
           (r"rollei", "rollei"), (r"konica", "konica"), (r"kentmere", "kentmere")]
_SPEED = re.compile(r"(?<!\d)(25|50|64|80|100|125|160|200|250|320|400|500|800|1600|3200)(?!\d)")
_NEG = re.compile(r"camera|lens|rolleiflex|rolleicord|kodak ?(retina|brownie|instamatic|ektra|signet|pony|duaflex)|fuji(film)? ?(x-?t\d|x-?t\b|x-?pro|x100|x-?e\d|x-?a\d|gfx|finepix|s\d)|xf ?\d|instax|film ?simulation|"
                  r"natura ?(classica|s\b|black|ns)|klasse|tiara|cardia|silvi|zoom", re.I)
_COMPILED = [(re.compile(p, re.I), s, b, t) for p, s, b, t in _LINES]
_BRANDC = [(re.compile(p, re.I), b) for p, b in _BRANDS]


def normalise_tag(tag: str) -> str:
    return urllib.parse.unquote_plus(tag or "").strip().lower()


def stock_claim(tags: str | list[str] | None) -> dict:
    """Return {film_stock, film_stock_speed, film_brand, film_type, evidence_tags} from comma-separated uploader tags.
    `film_stock` is None when only a brand (or nothing) is claimed; camera-model tags (Rolleiflex, Fuji X-T) are ignored."""
    raw = tags.split(",") if isinstance(tags, str) else (tags or [])
    found = []
    for t in (normalise_tag(x) for x in raw):
        if not t or _NEG.search(t):
            continue
        for rx, stock, brand, ftype in _COMPILED:
            if rx.search(t):
                m = _SPEED.search(t); found.append((stock, brand, ftype, m.group(1) if m else None, t)); break
    if found:
        stocks = sorted({f[0] for f in found})
        if len(stocks) > 1:                                      # conflicting claims: record all, assert none
            return {"film_stock": None, "film_stock_speed": None, "film_brand": None, "film_type": None, "evidence_tags": [f[4] for f in found], "conflict": stocks}
        stock, brand, ftype, _, _ = found[0]; speed = next((f[3] for f in found if f[3]), None)
        named = f"{stock}_{speed}" if speed and not re.search(r"\d", stock) else stock      # lines that already carry a speed (cinestill_800t, fuji_c200) stay as they are
        return {"film_stock": named, "film_stock_line": stock, "film_stock_speed": speed, "film_brand": brand, "film_type": ftype,
                "evidence_tags": [f[4] for f in found]}
    for t in (normalise_tag(x) for x in raw):
        if _NEG.search(t):
            continue
        for rx, brand in _BRANDC:
            if rx.search(t):
                return {"film_stock": None, "film_stock_speed": None, "film_brand": brand, "film_type": None, "evidence_tags": [t]}
    return {"film_stock": None, "film_stock_speed": None, "film_brand": None, "film_type": None, "evidence_tags": []}
