"""C12a: open-set look data, every feature computed through the WEBSITE's own code path.

    OMP_NUM_THREADS=1 python -m research.c12.build all --workers 6     # -> data/c12/<group>/{meta.jsonl, desc.npy, col312.npy, dino.npy, engine_d.npy}

Path per image (read-only import of ~/misc/film-evaluation): pixels -> PNG -> filmlook.image_utils.prepare_image (what
an upload goes through) -> filmlook.filmstyle.engine.load_image -> filmlook.filmstyle.descriptors.describe (205 values,
the production descriptor) -> FilmStyleEngine.analyze(desc=...) (method D ranking and its `uncertain` flag, the live
engine). Also the 312 research colour values (research.lut.features; NOT computed by the website, ablation only) and
DINOv2-small blocks 4/8/12 [CLS, mean] on the photo-gate crop of the prepared JPEG (fp32, = production ONNX before its
per-chunk L2 normalisation).

Groups:
  pairs     the C08 sources and their source split (data/c08/pairs/sources.json). Known classes: the 13 owner presets +
            `none`, clean + 2 varied renders (train) / clean + 1 varied (val, test), the same seeds as C08. Unknown edits:
            every other analysable library LUT (colour film emulations, creative looks, B&W), split by FAMILY (film stock,
            or creative-look stem) into train / val / test families; a source draws unknown LUTs only from its own
            split's families. Test sources: the varied known renders and 8 unknown renders again under 7 perturbations.
  pool      Common Catalog preset renders (data/lut/synth) of scenes that are NOT C08 sources at all (60 scenes; the
            C08 "unseen pool" had 52 of its 81 scenes among the C08 training sources), plus test-family unknown LUTs
            applied to those scenes' untouched renders.
  real      film/digital set (data/lut/film_likeness, the claude_annotations film_eval set), test split only, with the
            same 7 perturbations.
  edited    110 edited uploads (data/lut/synth/negatives).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

WEB = Path.home() / "misc" / "film-evaluation"
OUT = Path("data/c12")
PERTURB = ("jpeg50", "half", "crop60", "wb+8", "wb-8", "ev+1", "ev-1")
N_UNK = {"train": 20, "val": 24, "test": 24}
N_UNK_ROBUST = 8
_S: dict = {}


def _web():
    if str(WEB) not in sys.path:
        sys.path.insert(0, str(WEB))


# ------------------------------------------------------------------------------------------------ LUT families
def family_of(L: dict) -> str:
    import re
    if L.get("film_stock"):
        return f"stock:{L['film_stock']}"
    return f"{L['source']}:{re.sub(r'[0-9 _-]+$', '', L['name']).lower()}"


def unknown_luts(luts: list[dict]) -> list[dict]:
    """Every analysable library LUT that is not one of the 13 owner presets (and not a technical transform)."""
    from research.c08.common import lut_kind
    out = []
    for L in luts:
        if L["source"] == "owner_presets" or L["lut"] is None:
            continue
        k = lut_kind(L)
        if k == "technical":
            continue
        L = dict(L); L["kind"] = k; L["family"] = family_of(L)
        out.append(L)
    return out


def split_families(unk: list[dict]) -> dict[str, str]:
    """Per kind, families ordered by a fixed hash: first 50 % train, next 20 % val, rest test."""
    by_kind = defaultdict(set)
    for L in unk:
        by_kind[L["kind"]].add(L["family"])
    out = {}
    for k, fams in by_kind.items():
        fams = sorted(fams, key=lambda f: hashlib.sha1(f"c12:{f}".encode()).hexdigest())
        n = len(fams); a, b = round(0.5 * n), round(0.7 * n)
        for i, f in enumerate(fams):
            out[f] = "train" if i < a else "val" if i < b else "test"
    return out


def lut_table():
    from research.c08.common import all_luts
    from research.c08.pairs import select_luts
    known = [L for L in select_luts() if L["source"] in ("owner_presets", "identity")]
    unk = unknown_luts(all_luts())
    fam_split = split_families(unk)
    for L in unk:
        L["unk_split"] = fam_split[L["family"]]
    return known, unk


# ------------------------------------------------------------------------------------------------ production path
def _init(known, unk):
    _web()
    from filmlook.filmstyle.engine import FilmStyleEngine
    _S["known"] = {L["id"]: L for L in known}
    _S["unk"] = {L["id"]: L for L in unk}
    _S["engine"] = FilmStyleEngine()


def _png(img: np.ndarray) -> bytes:
    b = io.BytesIO(); Image.fromarray(img).save(b, "PNG", compress_level=1); return b.getvalue()


def perturb(img: np.ndarray, kind: str | None) -> bytes:
    """Perturbation of the photo as uploaded; returns the file bytes production would receive."""
    from research.lut.forward_model import srgb_decode, srgb_encode
    if kind is None:
        return _png(img)
    if kind == "jpeg50":
        b = io.BytesIO(); Image.fromarray(img).save(b, "JPEG", quality=50); return b.getvalue()
    if kind == "half":
        im = Image.fromarray(img); return _png(np.asarray(im.resize((max(16, im.width // 2), max(16, im.height // 2)), Image.LANCZOS)))
    if kind == "crop60":
        h, w = img.shape[:2]; ch, cw = round(0.6 * h), round(0.6 * w); t, l = (h - ch) // 2, (w - cw) // 2
        return _png(np.ascontiguousarray(img[t:t + ch, l:l + cw]))
    lin = srgb_decode(img / 255.0)
    if kind.startswith("wb"):
        s = 0.08 if kind == "wb+8" else -0.08
        lin = lin * np.array([1 + s, 1.0, 1 - s])
    elif kind.startswith("ev"):
        lin = lin * (2.0 if kind == "ev+1" else 0.5)
    return _png(np.clip(srgb_encode(np.clip(lin, 0, 1)) * 255, 0, 255).round().astype(np.uint8))


def features(raw: bytes):
    """Exactly the website's path from upload bytes to descriptor, engine ranking and encoder crop."""
    from filmlook.filmstyle.descriptors import describe
    from filmlook.filmstyle.engine import load_image
    from filmlook.image_utils import prepare_image
    from research.c08.pairs import dino_crop
    from research.lut.features import colour_features_from_array
    jpeg = prepare_image(raw).data
    img = load_image(jpeg)
    desc = describe(img)
    eng = _S["engine"]
    r = eng.analyze(img, top_k=14, desc=desc)
    d = np.array([r["distances"][i] for i in eng.db.ids], np.float32)
    return img, desc.astype(np.float32), d, bool(r["uncertain"]), r["uncertainty_reason"], colour_features_from_array(img).astype(np.float32), dino_crop(img)


def _emit(rows, arrs, raw, meta):
    img, desc, d, unc, reason, col, crop = features(raw)
    rows.append({**meta, "engine_uncertain": unc, "engine_reason": reason})
    arrs["desc"].append(desc); arrs["engine_d"].append(d); arrs["col312"].append(col); arrs["crop"].append(crop)
    return img


def _new():
    return [], {"desc": [], "engine_d": [], "col312": [], "crop": []}


def job_pairs(job):
    from research.c08.pairs import _vary, render
    from research.lut.synthetic_library import load_photo
    src = load_photo(job["path"], 512)
    rows, arrs = _new()
    base = {"source_id": job["source_id"], "split": job["split"], "category": job["category"]}

    def one(L, k, cls, extra, robust):
        seed = int(hashlib.sha1(f"{job['source_id']}:{L['id']}:{k}".encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        v = None if k == 0 else _vary(rng)
        x = render(src, L["lut"], v, rng)
        meta = {**base, "lut": L["id"], "cls": cls, "variant": k, "clean": v is None, "params": v, **extra}
        _emit(rows, arrs, perturb(x, None), {**meta, "perturb": None})
        if robust:
            for p in PERTURB:
                _emit(rows, arrs, perturb(x, p), {**meta, "perturb": p})

    nvar = 2 if job["split"] == "train" else 1
    for L in _S["known"].values():
        for k in range(1 + nvar):
            one(L, k, L["cls"], {"kind": "known"}, robust=job["split"] == "test" and k == 1)
    pool = [L for L in _S["unk"].values() if L["unk_split"] == job["split"]]
    fams = sorted({L["family"] for L in pool})
    rng = np.random.default_rng(int(hashlib.sha1(f"c12unk:{job['source_id']}".encode()).hexdigest(), 16) % (2 ** 32))
    pick_f = rng.choice(fams, N_UNK[job["split"]], replace=len(fams) < N_UNK[job["split"]])
    chosen = []
    for f in pick_f:
        cand = sorted([L["id"] for L in pool if L["family"] == f and L["id"] not in chosen])
        if cand:
            chosen.append(cand[int(rng.integers(len(cand)))])
    for n, lid in enumerate(chosen):
        L = _S["unk"][lid]
        one(L, 1, "unknown", {"kind": L["kind"], "family": L["family"], "unk_split": L["unk_split"]}, robust=job["split"] == "test" and n < N_UNK_ROBUST)
    return rows, arrs


def job_file(job):
    """A photograph on disk (pool render, real photo, edited upload); optional LUT renders on it; optional perturbations."""
    from research.c08.pairs import _vary, render
    rows, arrs = _new()
    raw = Path(job["path"]).read_bytes()
    img = _emit(rows, arrs, raw, {**job["meta"], "perturb": None})
    for p in job.get("perturb", []):
        _emit(rows, arrs, perturb(img, p), {**job["meta"], "perturb": p})
    for lid in job.get("luts", []):
        L = _S["unk"][lid]
        seed = int(hashlib.sha1(f"{job['meta']['scene_id']}:{lid}:1".encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        x = render(img, L["lut"], _vary(rng), rng)
        _emit(rows, arrs, perturb(x, None), {**job["meta"], "lut": lid, "cls": "unknown", "kind": L["kind"], "family": L["family"], "perturb": None})
    return rows, arrs


# ------------------------------------------------------------------------------------------------ jobs
def jobs_for(group: str, known, unk) -> list[dict]:
    if group == "pairs":
        return [("pairs", j) for j in json.loads(Path("data/c08/pairs/sources.json").read_text())]
    if group == "pool":
        srcs = json.loads(Path("data/c08/pairs/sources.json").read_text())
        used_ids = {s["source_id"] for s in srcs}; used_groups = {s["group_id"] for s in srcs}
        rows = [json.loads(l) for l in open("data/lut/synth/manifest.jsonl")]
        rows = [r for r in rows if r["split"] != "negatives" and f"cc_{r['scene_id']}" not in used_ids and r["group_id"] not in used_groups]
        test_unk = sorted(L["id"] for L in unk if L["unk_split"] == "test")
        out = []
        for r in rows:
            cls = "none" if r["preset"] == "none" else next(L["cls"] for L in known if L["name"] == r["preset"])
            luts = []
            if r["preset"] == "none":
                rng = np.random.default_rng(int(hashlib.sha1(f"c12pool:{r['scene_id']}".encode()).hexdigest(), 16) % (2 ** 32))
                luts = list(rng.choice(test_unk, N_UNK["test"], replace=False))
            out.append(("file", {"path": r["path"], "luts": luts, "meta": {"scene_id": r["scene_id"], "group_id": r["group_id"], "cls": cls, "kind": "known",
                                                                           "variant": r["variant"], "synth_split": r["split"]}}))
        return out
    if group == "real":
        rows = [json.loads(l) for l in open("data/lut/film_likeness/manifest.jsonl")]
        return [("file", {"path": r["path"], "perturb": list(PERTURB), "meta": {k: r[k] for k in ("path", "label", "uploader", "dataset", "film_type", "colour")}})
                for r in rows if r["split"] == "test"]
    if group == "edited":
        return [("file", {"path": str(p), "meta": {"path": str(p), "cls": "unknown", "kind": "edited"}}) for p in sorted(Path("data/lut/synth/negatives").glob("*.jpg"))]
    raise ValueError(group)


def _run(job):
    kind, j = job
    return job_pairs(j) if kind == "pairs" else job_file(j)


def build(group: str, workers: int, limit: int) -> None:
    from research.c08.experiments import norm_id
    from research.c08.pairs import DinoLayers
    known, unk = lut_table()
    for L in known:
        L["cls"] = norm_id(L["name"])
    jobs = jobs_for(group, known, unk)[: limit or None]
    out = OUT / group
    out.mkdir(parents=True, exist_ok=True)
    (OUT / "luts.json").write_text(json.dumps({"known": [{"id": L["id"], "cls": L["cls"]} for L in known],
                                               "unknown": [{k: L[k] for k in ("id", "name", "source", "kind", "family", "unk_split", "film_stock")} for L in unk]}, indent=1))
    dino = DinoLayers()
    rows, A, D, buf = [], defaultdict(list), [], []
    with Pool(workers, initializer=_init, initargs=(known, unk)) as pool:
        for n, (r, arrs) in enumerate(pool.imap(_run, jobs, chunksize=1)):
            rows += r
            for k in ("desc", "engine_d", "col312"):
                A[k] += arrs[k]
            buf += arrs["crop"]
            if len(buf) >= 512:
                D.append(dino(np.stack(buf))); buf = []
            if n % 20 == 0:
                print(f"{group}: {n + 1}/{len(jobs)} jobs, {len(rows)} images", flush=True)
    if buf:
        D.append(dino(np.stack(buf)))
    np.save(out / "dino.npy", np.concatenate(D))
    for k in ("desc", "engine_d", "col312"):
        np.save(out / f"{k}.npy", np.stack(A[k]).astype(np.float32))
    with open(out / "meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(group, "done", len(rows), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=["pairs", "pool", "real", "edited", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for g in (["edited", "pool", "real", "pairs"] if a.group == "all" else [a.group]):
        build(g, a.workers, a.limit)


if __name__ == "__main__":
    main()
