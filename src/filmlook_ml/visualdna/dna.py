"""Visual DNA knowledge matching, the fusion ranker and the bilingual explanation.

Knowledge matching scores a photographer from the knowledge graph only: which of the photographer's documented motifs
(objects, scenes) and documented style traits (tonality, light, composition) appear in a photograph's detected and
measured tags. Motif weight follows the evidence level (signature > repeated > series-specific > tendency; weak
associations never score) times the stated confidence. Subject evidence and style evidence are kept as two separate
sub-scores so an explanation can say "the subject matches but the rendering does not".

Fusion combines up to three signals per photographer: DINOv2 visual similarity to reference images, SigLIP semantic
similarity to reference images (whole-image content), and the knowledge score. Each signal is first turned into a
probability over the candidate photographers with its own temperature (fitted on validation data), so raw cosine
values from different models are never added. Weights are configurable (initial 0.45 / 0.35 / 0.20 from the brief) and
tuned on validation data. A missing signal (for example no licensed reference images for an in-copyright
photographer) is dropped and the remaining weights renormalised; the result then carries ``signals_used`` and can never
be labelled ``strong``.

Strength labels: ``strong`` needs at least two signals that each rank the photographer first; ``subject_only`` when
only knowledge subject evidence supports it; ``style_only`` when only rendering evidence does; otherwise ``weak``.
No label ever claims that a photographer made, inspired or influenced the user's photograph.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from filmlook_ml.visualdna.build import DB, KNOWLEDGE

LEVEL_WEIGHT = {"signature": 1.0, "repeated": 0.6, "series_specific": 0.5, "tendency": 0.35, "weak": 0.0}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4, None: 0.5}
DEFAULT_WEIGHTS = {"dinov2": 0.45, "semantic": 0.35, "dna": 0.20}
SUBJECT_KINDS = ("object", "scene")
# Context tags are too common to identify anyone on their own. Inside a motif they count only when a distinctive tag
# of the same motif is present; as style evidence they are down-weighted.
CONTEXT_TAGS = frozenset({"street", "sky", "colour", "monochrome", "crowd", "child", "worker", "posed portrait", "close-up", "tree", "interior"})
CONTEXT_WEIGHT = 0.3
PARENT = {"crow": "bird", "mother and child": "child", "city street at night": "street", "farm field": "farm"}
STYLE_KINDS = ("attribute", "composition")


@dataclass
class Evidence:
    kind: str           # subject | style
    tag: str
    strength: float     # detection score (0-1) or 1.0 for a measured fact
    weight: float
    element: str        # the motif or trait it supports
    level: str
    series: list = field(default_factory=list)


@lru_cache(maxsize=1)
def knowledge(db: Path = DB) -> dict:
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    vocab = dict(c.execute("SELECT tag, kind FROM vocabulary").fetchall())
    out = {}
    for p in c.execute("SELECT * FROM photographers"):
        pid = p["id"]
        motifs = []
        for m in c.execute("SELECT * FROM motifs WHERE photographer_id = ?", (pid,)):
            tags = [r[0] for r in c.execute("SELECT tag FROM motif_tags WHERE photographer_id = ? AND motif_idx = ?", (pid, m["idx"]))]
            motifs.append({**dict(m), "tags": tags, "series": json.loads(m["series_json"])})
        styles = [{**dict(s), "tags": json.loads(s["tags_json"])} for s in c.execute("SELECT * FROM style_traits WHERE photographer_id = ?", (pid,))]
        series = {s["id"]: dict(s) for s in c.execute("SELECT * FROM series WHERE photographer_id = ?", (pid,))}
        books = {b["id"]: dict(b) for b in c.execute("SELECT * FROM photobooks WHERE photographer_id = ?", (pid,))}
        links = [dict(r) for r in c.execute("SELECT * FROM series_photobooks WHERE photographer_id = ?", (pid,))]
        out[pid] = {"profile": dict(p), "motifs": motifs, "styles": styles, "series": series, "books": books, "series_books": links}
    c.close()
    return {"vocab": vocab, "photographers": out}


def knowledge_score(pid: str, detected: dict[str, float], measured: set[str], kb: dict | None = None) -> dict:
    """-> {subject, style, total, evidence[]}. ``detected``: tag -> zero-shot score above threshold."""
    kb = kb or knowledge()
    vocab, ph = kb["vocab"], kb["photographers"][pid]
    present = {**{t: float(s) for t, s in detected.items()}, **{t: 1.0 for t in measured}}
    ev: list[Evidence] = []
    used: set[tuple[str, str]] = set()  # (kind, tag): one tag counts once per photographer (no double counting)
    for m in ph["motifs"]:
        w = LEVEL_WEIGHT[m["level"]] * CONFIDENCE_WEIGHT.get(m["confidence"], 0.5)
        if w == 0:
            continue
        distinctive = [t for t in m["tags"] if t not in CONTEXT_TAGS and t in present]
        if not distinctive and any(t not in CONTEXT_TAGS for t in m["tags"]):
            continue  # Only context tags matched (e.g. "street" alone): not evidence for this motif.
        for t in m["tags"]:
            kind = "subject" if vocab.get(t) in SUBJECT_KINDS else "style"
            if t in present and (kind, t) not in used:
                wt = w * (CONTEXT_WEIGHT if t in CONTEXT_TAGS else 1.0)
                used.add((kind, t)); ev.append(Evidence(kind, t, present[t], wt, m["element"], m["level"], m["series"]))
    for s in ph["styles"]:
        for t in s["tags"]:
            if t in present and ("style", t) not in used:
                base = 0.4 if s["kind"] == "observable" else 0.2
                used.add(("style", t)); ev.append(Evidence("style", t, present[t], base * (CONTEXT_WEIGHT if t in CONTEXT_TAGS else 1.0), s["value"], "tendency", []))
    subject = sum(e.strength * e.weight for e in ev if e.kind == "subject")
    style = sum(e.strength * e.weight for e in ev if e.kind == "style")
    # Saturating combination: a single strong subject cue cannot exceed ~0.63 on its own.
    total = 1 - np.exp(-(subject + style))
    return {"subject": float(subject), "style": float(style), "total": float(total), "evidence": ev}


def to_probs(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature; z = z - z.max(); e = np.exp(z)
    return e / e.sum()


def fuse(candidates: list[str], signals: dict[str, np.ndarray | None], temperatures: dict[str, float], weights: dict[str, float] | None = None) -> dict:
    """signals: name -> raw score vector over ``candidates`` (None when unavailable). Returns ranked results."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    probs = {k: to_probs(np.asarray(v, float), temperatures[k]) for k, v in signals.items() if v is not None}
    total_w = sum(weights[k] for k in probs)
    fused = sum(weights[k] / total_w * p for k, p in probs.items())
    order = np.argsort(-fused)
    firsts = {k: int(np.argmax(p)) for k, p in probs.items()}
    ranked = []
    for i in order:
        agree = [k for k, j in firsts.items() if j == i]
        ranked.append({"photographer_id": candidates[i], "fused": float(fused[i]), "signals": {k: float(p[i]) for k, p in probs.items()}, "first_in": agree})
    return {"ranked": ranked, "signals_used": sorted(probs), "weights": {k: weights[k] / total_w for k in probs}}


ABSTAIN_BELOW = 0.30  # knowledge total below this: no meaningful match (provisional; tuned on validation data)
EN_PHRASE = {"through glass": "view through glass", "layered frame": "layered foreground", "negative space": "empty space", "dark tones": "dark tonality"}


def strength_label(result: dict, k: dict) -> str:
    """Never 'strong' from one signal; subject-only and style-only are said as such; low evidence abstains."""
    if len(result["first_in"]) >= 2:
        return "strong"
    if k["total"] < ABSTAIN_BELOW:
        return "no_match"
    if k["subject"] > 0 and k["style"] > 0:
        return "subject_and_style"
    if k["subject"] > 0 and k["style"] == 0:
        return "subject_only"
    if k["style"] > 0 and k["subject"] == 0:
        return "style_only"
    return "weak"


def _series_book(ph: dict, ev: list[Evidence]) -> tuple[str | None, str | None, bool]:
    """-> (series title, book title for display, whether the book is that series' own book)."""
    for e in ev:
        for sid in e.series:
            s = ph["series"].get(sid)
            if s:
                bid = next((l["photobook_id"] for l in ph["series_books"] if l["series_id"] == sid and l["photobook_id"] in ph["books"]), None)
                book = ph["books"].get(bid) if bid else None
                title = (book.get("title_zh") or book.get("title_original") or book.get("title_en")) if book else None
                return s["title"], title, bid == sid
    return None, None, False


def explain(pid: str, k: dict, label: str, detected_order: list[str], kb: dict | None = None) -> dict:
    """Short zh/en text from verified knowledge only. Returns {zh, en, found_zh, found_en, related_work}."""
    kb = kb or knowledge()
    zh_label = json.loads((KNOWLEDGE / "_vocabulary.json").read_text(encoding="utf-8"))["labels_zh"]
    ph = kb["photographers"][pid]; prof = ph["profile"]
    name_zh, name_en = prof["name_zh"] or prof["name_en"], prof["name_en"]
    found = detected_order[:5]
    tags_present = {e.tag for e in k["evidence"]}
    rank = lambda es: sorted(es, key=lambda e: (e.tag in CONTEXT_TAGS, -(e.strength * e.weight)))  # noqa: E731
    drop = {PARENT[t] for t in tags_present if t in PARENT}
    subj = [e for e in rank(k["evidence"]) if e.kind == "subject" and e.tag not in drop]
    sty = [e for e in rank(k["evidence"]) if e.kind == "style" and e.tag not in drop]
    if any(e.tag not in CONTEXT_TAGS for e in subj):
        subj = [e for e in subj if e.tag not in CONTEXT_TAGS]
    if any(e.tag not in CONTEXT_TAGS for e in sty):
        sty = [e for e in sty if e.tag not in CONTEXT_TAGS]
    series, book, book_is_series = _series_book(ph, subj or sty)
    work_zh = (f"的《{book}》系列" if book_is_series else f"的《{book}》") if book else (f"的「{series}」" if series else "")
    work_en = series or ""
    s_zh = "和".join(zh_label.get(e.tag, e.tag) for e in subj[:2]); s_en = " and ".join(e.tag for e in subj[:2])
    t_zh = "和".join(zh_label.get(e.tag, e.tag) for e in sty[:2]); t_en = " and ".join(EN_PHRASE.get(e.tag, e.tag) for e in sty[:2])
    verb = "recall" if len(subj[:2]) > 1 else "recalls"
    if label == "no_match":
        return {"zh": "这张照片没有和资料库中的摄影师形成明确的关联。", "en": "This photograph has no clear link to a photographer in the database.",
                "found_zh": " · ".join(zh_label.get(t, t) for t in found), "found_en": " · ".join(found), "related_work_zh": None, "related_work_en": None, "label": label}
    if label == "subject_only":
        zh = f"照片里的{s_zh}是{name_zh}{work_zh}中反复出现的题材，但画面的影调与{name_zh}的风格差别较大，这只是题材上的联想。"
        en = f"The {s_en} in this photograph is a recurring subject in {name_en}'s {work_en or 'work'}, but the rendering differs; this is a subject association only."
        verb = verb
    elif label == "style_only":
        zh = f"画面的{t_zh}与{name_zh}的影像语言有相近之处，题材上没有直接对应。"
        en = f"Its {t_en} echo {name_en}'s visual language; the subject has no direct counterpart."
    elif subj and sty:
        zh = f"照片中的{s_zh}让人联想到{name_zh}{work_zh}，{t_zh}的处理也形成了一定的视觉呼应。"
        en = f"The {s_en} in this photograph {verb} {name_en}'s {work_en or 'work'}; the {t_en} add a visual echo."
    else:
        zh = f"整体视觉上与{name_zh}的作品有一定相似，但没有明确的题材或风格依据。"
        en = f"Visually somewhat close to {name_en}'s work, without a clear subject or style cue."
    if prof["profile_status"] != "complete":
        zh += "（该摄影师资料尚不完整，仅作参考。）"; en += " (This photographer's profile is incomplete; treat as a loose reference.)"
    return {"zh": zh, "en": en, "found_zh": " · ".join(zh_label.get(t, t) for t in found), "found_en": " · ".join(found),
            "related_work_zh": work_zh or None, "related_work_en": work_en or None, "label": label}
