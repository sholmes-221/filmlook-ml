"""Feature extraction + pseudo-label generation over a manifest."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from filmlook_ml.compliance.policy import claude_training_permission
from filmlook_ml.compliance.provenance import make_provenance
from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.datasets.edits import EDIT_PAIRS, EDIT_TARGETS
from filmlook_ml.labeling.features import FEATURE_NAMES, compute_features
from filmlook_ml.labeling.schema import ImageLabel, PairLabel, TeacherOutput
from filmlook_ml.labeling.teachers import (
    BudgetExceeded,
    ClaudeTeacher,
    HeuristicTeacher,
    check_label,
    film_score_from_attributes,
    training_authorization,
)


def _feat_one(args):
    from filmlook_infer.imageio import load_image
    path, crop, max_side = args
    return compute_features(load_image(path, max_side=max_side), crop)


def extract_features(manifest: list[dict], out_path: Path, crop: int = 512, workers: int = 8,
                     max_side: int = 1280) -> dict[str, dict]:
    """Cached on disk keyed by sha256 (re-running is cheap). Uses the SAME loader as inference."""
    out_path = Path(out_path)
    cache: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [r for r in manifest if r["sha256"] not in cache]
    if todo:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for r, f in zip(todo, ex.map(_feat_one, [(r["path"], crop, max_side) for r in todo], chunksize=4)):
                cache[r["sha256"]] = f
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cache))
    return {r["image_id"]: cache[r["sha256"]] for r in manifest}


def feature_matrix(ids: list[str], feats: dict[str, dict]) -> np.ndarray:
    return np.array([[feats[i][k] for k in FEATURE_NAMES] for i in ids], dtype=np.float32)


def heuristic_labels(manifest: list[dict], feats: dict[str, dict], prompt_version: str) -> list[ImageLabel]:
    t = HeuristicTeacher()
    return [ImageLabel(image_id=r["image_id"], sha256=r["sha256"], source="heuristic", teacher_model=t.model_id,
                       prompt_version=prompt_version, label=t.label(Path(r["path"]), feats[r["image_id"]]),
                       provenance=make_provenance("heuristic", r, model_id=t.model_id))
            for r in manifest]


def synthetic_labels(manifest: list[dict]) -> tuple[list[ImageLabel], list[PairLabel]]:
    labels, pairs = [], []
    by_family: dict[str, dict[str, str]] = {}
    rec = {r["image_id"]: r for r in manifest}
    for r in manifest:
        by_family.setdefault(r["family_id"], {})[r["edit_type"]] = r["image_id"]
        tgt = EDIT_TARGETS.get(r["edit_type"])
        if not tgt:
            continue
        attrs = {a: tgt.get(a, 0.5) for a in ATTRIBUTES}
        labels.append(ImageLabel(
            image_id=r["image_id"], sha256=r["sha256"], source="synthetic_ordering",
            teacher_model="edit-family-v2", prompt_version="n/a",
            label=TeacherOutput(film_score=film_score_from_attributes(attrs), attributes=attrs,
                                visual_evidence=f"known edit: {r['edit_type']}", label_confidence=0.7),
            supervised_attributes=sorted(tgt), provenance=make_provenance("synthetic_ordering", r, model_id="edit-family-v2")))
    for fam, ed in by_family.items():
        for win, lose, attr in EDIT_PAIRS:
            if win in ed and lose in ed:
                pairs.append(PairLabel(image_a=ed[win], image_b=ed[lose], attribute=attr, winner="a",
                                       source="synthetic_ordering",
                                       provenance=make_provenance("synthetic_ordering", rec[ed[win]], model_id="edit-family-v2")))
    return labels, pairs


# ----------------------------------------------------------------------------- Claude
PAIR_EDITS = ["film_emulation", "fake_grain_filter", "hdr_overprocessed", "oversharpened"]


def claude_selection(manifest: list[dict], consented: set[str], n_images: int, n_pairs: int,
                     seed: int = 0) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """Diverse, budget-sized selection. Priority: real photos (balanced by medium x subject x split),
    then a few edited variants (so the teacher sees filters), then original-vs-edit pairs."""
    rng = random.Random(seed)
    ok = [r for r in manifest if r["dataset_id"] in consented]
    origs = [r for r in ok if r["edit_type"] == "original" and r["dataset_id"] != "synthetic_fixture"]
    buckets = defaultdict(list)
    for r in origs:
        buckets[(r.get("medium", r["dataset_id"]), (r.get("subjects") or ["other"])[0])].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    n_orig = int(n_images * 0.75)
    picked: list[dict] = []
    while len(picked) < n_orig and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(picked) < n_orig:
                picked.append(buckets[k].pop())
    fam_of = defaultdict(dict)
    for r in ok:
        fam_of[r["family_id"]][r["edit_type"]] = r
    fams = [r["family_id"] for r in picked]
    rng.shuffle(fams)
    for k, fam in enumerate(fams):
        if len(picked) >= n_images:
            break
        e = fam_of[fam].get(PAIR_EDITS[k % len(PAIR_EDITS)])
        if e:
            picked.append(e)
    pairs = []
    for k, fam in enumerate(fams[: n_pairs]):
        o, e = fam_of[fam].get("original"), fam_of[fam].get(PAIR_EDITS[(k + 1) % len(PAIR_EDITS)])
        if o and e:
            pairs.append((o, e) if rng.random() < 0.5 else (e, o))  # randomise A/B position
    return picked, pairs


def claude_labels(selection: list[dict], pairs: list[tuple[dict, dict]], teacher: ClaudeTeacher,
                  consented: set[str], feats: dict[str, dict]) -> tuple[list[ImageLabel], list[PairLabel], dict]:
    auth = training_authorization(teacher.cfg)
    permission, basis = claude_training_permission(teacher.cfg)
    usable = permission == "permitted"          # policy section 2: uncertain/not_permitted never trains
    out, out_pairs, skipped, stopped = [], [], 0, None
    try:
        for r in selection:
            if r["dataset_id"] not in consented:
                skipped += 1
                continue
            res = teacher.label(Path(r["path"]), sha256=r["sha256"], image_id=r["image_id"], consent_ok=True)
            if res is None:  # dry run
                continue
            flags = check_label(res, feats.get(r["image_id"]))
            out.append(ImageLabel(image_id=r["image_id"], sha256=r["sha256"], source="claude",
                                  teacher_model=teacher.model_id, prompt_version=teacher.prompt_version,
                                  label=res, flags=flags, usable_for_training=usable,
                                  review_status="needs_review" if flags else "unreviewed",
                                  authorization=auth,
                                  provenance=make_provenance("claude", r, model_id=teacher.model_id,
                                                             claude_cfg=teacher.cfg)))
        for a, b in pairs:
            res = teacher.compare(Path(a["path"]), Path(b["path"]), a["sha256"], b["sha256"],
                                  (a["image_id"], b["image_id"]), consent_ok=True)
            if res is None:
                continue
            for attr in ["film_score", *ATTRIBUTES]:
                w = getattr(res, attr)
                out_pairs.append(PairLabel(image_a=a["image_id"], image_b=b["image_id"], attribute=attr,
                                           winner={"A": "a", "B": "b"}.get(w, "tie"), source="claude",
                                           confidence=res.confidence,
                                           provenance=make_provenance("claude", a, model_id=teacher.model_id,
                                                                      claude_cfg=teacher.cfg)))
    except BudgetExceeded as e:
        stopped = str(e)
    report = {"labelled_images": len(out), "pair_judgements": len(out_pairs), "skipped_no_rights": skipped,
              "flagged": sum(bool(x.flags) for x in out), "spent_this_run_usd": round(teacher.cost.spent_this_run, 4),
              "spent_total_usd": round(teacher.cost.spent, 4), "input_tokens": teacher.cost.input_tokens,
              "output_tokens": teacher.cost.output_tokens, "calls": teacher.cost.calls,
              "training_authorization": auth, "training_permission": permission, "permission_basis": basis,
              "usable_for_training": usable, "dry_run": teacher.dry_run, "stopped_by_budget": stopped}
    return out, out_pairs, report
