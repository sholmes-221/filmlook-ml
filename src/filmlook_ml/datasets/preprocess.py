"""Validation, normalisation, de-duplication, manifests and group-aware splits."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import imagehash
from PIL import Image

from filmlook_ml.datasets.edits import apply_edit
from filmlook_ml.datasets.registry import Registry

from filmlook_infer.imageio import MAX_PIXELS, MIN_SIDE, ImageValidationError, limit_size, load_image  # noqa: F401


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(img: Image.Image) -> str:
    return str(imagehash.phash(img, hash_size=8))


def hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def save_processed(img: Image.Image, out_dir: Path, stem: str, lossless: bool = False) -> Path:
    """JPEG q96 with 4:4:4 chroma by default (~10x smaller than PNG; artefacts are far
    below grain amplitude). Use lossless=True for grain-specific studies."""
    if lossless:
        p = out_dir / f"{stem}.png"
        img.save(p, compress_level=3)
    else:
        p = out_dir / f"{stem}.jpg"
        img.save(p, "JPEG", quality=96, subsampling=0)
    return p


def _make_family(args) -> list[dict]:
    """Worker: generate + save all variants of one accepted original (runs in a spawn pool)."""
    rec, family_id, ph, src_path, max_side, edits, seed, lossless, out_dir = args
    img = limit_size(load_image(Path(src_path), max_side=max_side), max_side)
    rows = []
    for k, edit in enumerate(["original", *edits]):
        var = apply_edit(img, edit, seed=int(family_id[:8], 16) + k + seed)
        stem = f"{family_id}_{edit}"
        p = save_processed(var, Path(out_dir), stem, lossless)
        rows.append({
            "image_id": stem, "path": str(p), "sha256": sha256_file(p),
            "phash": phash(var) if edit == "original" else ph, "family_id": family_id,
            "edit_type": edit, "width": var.width, "height": var.height,
            "dataset_id": rec["dataset_id"], "license": rec.get("license"),
            "source_url": rec.get("source_url", ""), "source_id": rec.get("source_id", ""),
            "attribution": rec.get("attribution", ""),
            # provenance / metadata carried from acquisition (medium is EXIF-derived, not a judgement)
            "group_id": rec.get("group_id") or family_id,
            **{k2: rec[k2] for k2 in ("medium", "subjects", "camera", "author", "license_url", "monochrome_tag")
               if k2 in rec},
        })
    return rows


def prepare(raw_records: list[dict], registry: Registry, out_dir: Path, edits: list[str],
            max_side: int = 1280, near_dup_threshold: int = 4, seed: int = 0, lossless: bool = False,
            workers: int = 1) -> tuple[list[dict], list[dict]]:
    """Return (manifest_records, rejected_records).

    * license-gates every raw record (quarantine on failure)
    * exact dedupe (sha256 of decoded pixels) + near-dup (pHash) on originals - sequential, deterministic
    * generates edit variants sharing the original's family_id - parallel when workers > 1
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected, jobs = [], []
    seen_sha: set[str] = set()
    seen_ph: list[tuple[str, str]] = []
    for rec in raw_records:
        ok, why = registry.check_record(rec["dataset_id"], rec.get("license"))
        if not ok:
            rejected.append({**rec, "reason": f"license: {why}"})
            continue
        try:
            img = load_image(Path(rec["path"]), max_side=max_side)  # same path as inference
        except ImageValidationError as e:
            rejected.append({**rec, "reason": f"invalid: {e}"})
            continue
        img = limit_size(img, max_side)
        pix_sha = hashlib.sha256(img.tobytes()).hexdigest()
        if pix_sha in seen_sha:
            rejected.append({**rec, "reason": "exact duplicate"})
            continue
        ph = phash(img)
        dup_of = next((fid for fid, h in seen_ph if hamming(ph, h) <= near_dup_threshold), None)
        if dup_of is not None:
            rejected.append({**rec, "reason": f"near duplicate of {dup_of}"})
            continue
        family_id = pix_sha[:16]
        seen_sha.add(pix_sha)
        seen_ph.append((family_id, ph))
        jobs.append((rec, family_id, ph, rec["path"], max_side, list(edits), seed, lossless, str(out_dir)))
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            fams = list(ex.map(_make_family, jobs, chunksize=2))
    else:
        fams = [_make_family(j) for j in jobs]
    return [row for fam in fams for row in fam], rejected


def group_split(manifest: list[dict], fractions: dict[str, float], seed: int,
                group_key: str = "family_id") -> dict[str, str]:
    """Assign every *group* to exactly one split -> no edit leaks across splits."""
    groups = sorted({r[group_key] for r in manifest})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_val = max(1, round(n * fractions["val"])) if n >= 3 else 0
    n_test = max(1, round(n * fractions["test"])) if n >= 3 else 0
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return {r["image_id"]: assign[r[group_key]] for r in manifest}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
