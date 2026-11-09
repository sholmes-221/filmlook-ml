"""Run spektrafilm (GPL-3.0, third_party/, not vendored) over a folder of sRGB JPEGs for several stocks. Resumable.

    PYTHONPATH=third_party/stubs:third_party/spektrafilm/src .venv/bin/python simulator/run_spektrafilm.py <manifest> <img dir> <out dir> [long_side]

Default chain: negative → Portra Endura print → scan (spektrafilm's default), neutral print filters from its database,
no grain (the family features below are colour and tone). One PNG per stock and image under <out dir>/<stock>/.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from spektrafilm import init_params, simulate

STOCKS = ["kodak_gold_200", "kodak_ultramax_400", "fujifilm_xtra_400", "fujifilm_c200", "kodak_ektar_100", "kodak_portra_400"]


def main(manifest: Path, imgdir: Path, out: Path, long_side: int = 320) -> None:
    items = [i for i in json.loads(manifest.read_text()) if i["label"] == "digital"]
    log = open(out / "run.log", "a")
    for stock in STOCKS:
        (out / stock).mkdir(parents=True, exist_ok=True)
        params = init_params(film_profile=stock)
        params.io.input_color_space = "sRGB"
        params.io.input_cctf_decoding = True
        params.film_render.grain.active = False
        for it in items:
            name = it.get("file") or hashlib.sha1(it["title"].encode()).hexdigest()[:16] + ".jpg"
            target = out / stock / (Path(name).stem + ".png")
            if target.exists() or not (imgdir / name).exists():
                continue
            im = Image.open(imgdir / name).convert("RGB"); im.thumbnail((long_side, long_side))
            t = time.perf_counter()
            try:
                res = simulate(np.asarray(im, dtype=np.float64) / 255.0, params)
            except Exception as exc:  # noqa: BLE001
                print(stock, name, "error", repr(exc)[:200], file=log, flush=True)
                continue
            Image.fromarray((np.clip(res, 0, 1) * 255).astype(np.uint8)).save(target)
            print(stock, name, f"{time.perf_counter() - t:.2f}s", file=log, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 320)
