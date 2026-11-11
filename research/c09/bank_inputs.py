"""C09: production-faithful local inputs for every bank work (the site's own preparation path; research.c08.prod_inputs):
36 film statistics and DINOv2-small blocks 4/8/12 [CLS, mean], each part unit length.
    python -m research.c09.bank_inputs   # -> data/c09/bank_inputs.npz (work_ids, F36, layers)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from filmlook_ml.refdb import manifest
from research.c08.film_head_v2 import parts_normalised
from research.c08.prod_inputs import compute
from research.photographers.evaluate import load_rows

OUT = Path("data/c09/bank_inputs.npz")


def main():
    ids = [r["work_id"] for r in load_rows()]
    F, L = compute([str(manifest.image_path(w)) for w in ids], workers=20)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, work_ids=np.array(ids), F36=np.nan_to_num(F), layers=parts_normalised(L))
    print("saved", len(ids))


if __name__ == "__main__":
    main()
