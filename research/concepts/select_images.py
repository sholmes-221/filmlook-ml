"""Choose the image subset and the per-head label matrix from Open Images V7 human-verified image-level labels.

  python -m research.concepts.select_images ids      -> train-ids-needed.txt (for the streamed train metadata pass)
  python -m research.concepts.select_images build    -> data/concepts/selection.parquet

Per head: up to POS_CAP verified positives (at most PER_AUTHOR per Flickr author), up to VNEG_CAP verified negatives
(raters said "no" to one of the head's classes: the hardest negatives the source offers), plus GENERIC random
validation/test-set photographs shared by every head. Label per (image, head): 1 verified positive, 0 negative,
-1 excluded (unknown for a train image not otherwise labelled, or positive for an IGNORE class of that head).

Split by Flickr author (sha256 of the author string): 60 % train, 20 % val, 20 % test, so no photographer is in two
splits.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from research.concepts.common import IGNORE, META, DATA, class_maps
from research.concepts.concept_map import CONCEPTS

POS_CAP, VNEG_CAP, PER_AUTHOR, GENERIC = 650, 300, 4, 8000
HEADS = list(CONCEPTS)


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def load_labels() -> pd.DataFrame:
    parts = []
    for f, split in [("oidv7-val-annotations-human-imagelabels.csv", "validation"), ("oidv7-test-annotations-human-imagelabels.csv", "test"),
                     ("train-labels-concepts.csv", "train")]:
        d = pd.read_csv(META / f, usecols=["ImageID", "LabelName", "Confidence"])
        d["oi_split"] = split
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def head_sets(L: pd.DataFrame):
    n2m, _ = class_maps()
    pos = L[L.Confidence == 1]
    neg = L[L.Confidence == 0]
    out = {}
    for head, (_, names) in CONCEPTS.items():
        mids = {n2m[n] for n in names if n in n2m}
        ign = {n2m[n] for n in IGNORE.get(head, []) if n in n2m}
        P = set(pos[pos.LabelName.isin(mids)].ImageID)
        I = set(pos[pos.LabelName.isin(ign)].ImageID) - P
        V = set(neg[neg.LabelName.isin(mids)].ImageID) - P - I
        out[head] = (P, V, I)
    return out


def meta() -> pd.DataFrame:
    cols = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License", "Author", "AuthorProfileURL", "Rotation"]
    parts = [pd.read_csv(META / f, usecols=cols) for f in ["validation-images-with-rotation.csv", "test-images-with-rotation.csv"]]
    if (META / "train-images-needed.csv").exists():
        parts.append(pd.read_csv(META / "train-images-needed.csv", usecols=cols))
    return pd.concat(parts, ignore_index=True).drop_duplicates("ImageID").set_index("ImageID")


def capped(ids, cap, author=None, salt=""):
    ids = sorted(ids, key=lambda i: h(salt + i))
    if author is None:
        return ids[:cap]
    out, per = [], {}
    for i in ids:
        a = author.get(i, i)
        if per.get(a, 0) >= PER_AUTHOR:
            continue
        per[a] = per.get(a, 0) + 1
        out.append(i)
        if len(out) >= cap:
            break
    return out


def ids() -> None:
    """Train image ids that could be selected (no author info yet: take up to 3x the cap, by hash)."""
    L = load_labels()
    sets = head_sets(L)
    vt = set(L[L.oi_split != "train"].ImageID)
    need = set()
    for head, (P, V, _) in sets.items():
        need |= set(capped([i for i in P if i not in vt], 3 * POS_CAP, salt=head))
        need |= set(capped([i for i in V if i not in vt], VNEG_CAP, salt=head + "n"))
    pd.Series(sorted(need)).to_csv(META / "train-ids-needed.txt", index=False, header=False)
    print(f"{len(need):,} train ids needed")


def build() -> None:
    L = load_labels()
    sets = head_sets(L)
    M = meta()
    author = M["Author"].fillna("").astype(str).to_dict()
    M = M[M.License.str.contains("creativecommons.org/licenses/by/2.0", na=False)]
    M = M[M.Rotation.fillna(0) == 0]   # the ~1 % needing rotation are dropped rather than guessing the convention
    known = set(M.index)
    vt = set(L[L.oi_split != "train"].ImageID) & known
    chosen = set()
    for head, (P, V, _) in sets.items():
        chosen |= set(capped([i for i in P if i in known], POS_CAP, author, head))
        chosen |= set(capped([i for i in V if i in known], VNEG_CAP, author, head + "n"))
    generic = set(capped(sorted(vt), GENERIC, author, "generic"))
    chosen |= generic
    chosen = sorted(chosen)
    rows = pd.DataFrame({"image_id": chosen})
    rows["author"] = rows.image_id.map(lambda i: author.get(i, ""))
    r = rows.author.map(lambda a: h("split:" + a) % 10)
    rows["split"] = np.where(r < 6, "train", np.where(r < 8, "val", "test"))
    rows["generic"] = rows.image_id.isin(generic)
    rows["oi_split"] = rows.image_id.map(lambda i: "train" if i not in vt else M.loc[i, "Subset"])
    for c in ["OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL", "Rotation"]:
        rows[c] = rows.image_id.map(M[c])
    full = rows.oi_split != "train"   # all labels known for validation/test images; train only for the wanted classes
    for head, (P, V, I) in sets.items():
        y = np.where(rows.image_id.isin(P), 1, np.where(rows.image_id.isin(I), -1, 0))
        # A train image carries only the wanted classes; if it was chosen for another head, "no positive for this head"
        # is as reliable as for a validation image only because every head class and its IGNORE classes were kept.
        rows[head] = y.astype(np.int8)
    rows.to_parquet(DATA / "selection.parquet")
    print(rows.split.value_counts().to_dict(), len(rows))
    print(pd.DataFrame({hd: [(rows[hd] == 1).sum(), ((rows[hd] == 1) & (rows.split == "test")).sum()] for hd in HEADS}, index=["pos", "pos_test"]).T.to_string())


if __name__ == "__main__":
    {"ids": ids, "build": build}[sys.argv[1]]()
