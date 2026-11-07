"""python -m filmlook_infer.cli IMAGE [--lang en|zh] [--model DIR]  -> JSON on stdout.

Exit codes: 0 ok, 2 invalid input (JSON error object on stdout), 1 unexpected error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="filmlook_infer")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--model", default=os.environ.get("FILMLOOK_MODEL_DIR",
                                                      str(Path(__file__).resolve().parent.parent / "model")))
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    from filmlook_infer.analyzer import FilmLookAnalyzer, FilmLookError

    an = FilmLookAnalyzer(a.model, threads=a.threads)
    rc = 0
    for p in a.images:
        try:
            out = an.analyze(p, a.lang)
        except FilmLookError as e:
            out, rc = {"error": e.code, "message": str(e), "image": p}, 2
        print(json.dumps(out, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
