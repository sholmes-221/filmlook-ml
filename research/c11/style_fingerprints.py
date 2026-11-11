"""Reference colour fingerprints of the 13 film-style looks (LUT simulations), for the website's Colour Fingerprint.
    python -m research.c11.style_fingerprints   # -> ~/misc/film-evaluation/filmlook/data/quant/style-fingerprints-v1.json

Each look (the owner's preset, as the live engine models it) is applied at full strength to the same 100 licensed
digital originals (tier-1 CC0 / public-domain sources, every 6th by id), saved as JPEG q92, prepared like an upload and
measured with the website's own colour study; the vectorscope shares and summary values are averaged. The untouched
originals give the `none` fingerprint. This is what the simulation does to varied content, not a film scan and not a
measurement of any film emulsion.
"""
from __future__ import annotations

import io
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

WEB = Path.home() / "misc" / "film-evaluation"
sys.path.insert(0, str(WEB))
from filmlook.image_utils import prepare_image  # noqa: E402
from filmlook.quant import color_study as cs  # noqa: E402

from research.lut.engine.lut3d import LUT3D  # noqa: E402
from research.lut.forward_model import load_presets  # noqa: E402
from research.lut.synthetic_library import load_photo, tier1_photos  # noqa: E402

OUT = WEB / "filmlook" / "data" / "quant" / "style-fingerprints-v1.json"
_L = {}


def norm_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _init():
    _L.update({name: LUT3D(p.lattice(33)) for name, p in load_presets().items()})


def one(path: str):
    src = load_photo(path, 1024)
    out = {}
    for name, lut in [("none", None)] + list(_L.items()):
        x = src if lut is None else lut.apply_u8(src)
        buf = io.BytesIO(); Image.fromarray(x).save(buf, "JPEG", quality=92)
        s = cs.study(prepare_image(buf.getvalue()).data)
        out[name] = {"shares": np.array(s["vectorscope"]["shares"]), "neutral": s["vectorscope"]["neutral_share"], "sat": s["colour"]["saturation_mean"],
                     "colourfulness": s["colour"]["colourfulness"], "contrast": s["tone"]["contrast_std_L"], "mean_b": s["colour"]["mean_b"], "mean_a": s["colour"]["mean_a"],
                     "shadow_b": (s["colour"]["shadow_bias"] or {}).get("b"), "highlight_b": (s["colour"]["highlight_bias"] or {}).get("b")}
    return out


def main():
    rows = sorted(tier1_photos(), key=lambda r: r.get("source_id") or r["id"])[::6][:100]
    with ProcessPoolExecutor(16, initializer=_init) as ex:
        res = list(ex.map(one, [r["path"] for r in rows], chunksize=2))
    looks = {}
    for name in res[0]:
        acc = [r[name] for r in res]
        mean = lambda k: round(float(np.mean([a[k] for a in acc if a[k] is not None])), 3)  # noqa: E731
        looks[norm_id(name)] = {"label": name.replace("_", " "), "shares": np.round(np.mean([a["shares"] for a in acc], 0), 5).tolist(), "neutral_share": mean("neutral"),
                                "saturation_mean": mean("sat"), "colourfulness": mean("colourfulness"), "contrast_std_L": mean("contrast"), "mean_a": mean("mean_a"), "mean_b": mean("mean_b"),
                                "shadow_b": mean("shadow_b"), "highlight_b": mean("highlight_b")}
    OUT.write_text(json.dumps({"version": cs.VERSION, "kind": "LUT simulation",
                               "method": "each look applied at full strength to the same 100 licensed digital photographs (CC0 / public domain), measured with the site's colour study and averaged; 'none' is the untouched originals",
                               "not": "not a film scan and not a measurement of any film emulsion", "n_photographs": len(rows), "looks": looks}, ensure_ascii=False))
    print("looks", len(looks), "->", OUT)


if __name__ == "__main__":
    main()
