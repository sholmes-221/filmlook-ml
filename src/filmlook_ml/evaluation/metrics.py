"""Evaluation metrics + HTML report.

Three kinds of evidence are kept strictly apart:
1. teacher agreement  - imitation of heuristic / Claude labels (NOT human accuracy)
2. controlled pairs   - within-photo orderings from known edits (synthetic) or teacher/human pair judgements
3. capture medium     - EXIF/scanner metadata: real film scans vs digital captures. Independent of every
                        teacher, but it measures "looks like a real film scan", which is related to - not
                        identical with - human judgement of film aesthetics.
Human agreement is reported only when human labels exist; otherwise it is marked unverified.
"""
from __future__ import annotations

import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.datasets.edits import EVAL_CATEGORY
from filmlook_ml.labeling.store import DIGITAL_MEDIA, FILM_MEDIA
from filmlook_ml.training.train import expected_calibration_error


def regression(pred: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> dict:
    m = w > 0
    if m.sum() < 3:
        return {"n": int(m.sum())}
    p, t = pred[m], tgt[m]
    ok = p.std() > 0 and t.std() > 0
    return {"n": int(m.sum()), "mae": float(np.abs(p - t).mean()),
            "pearson": float(stats.pearsonr(p, t)[0]) if ok else None,
            "spearman": float(stats.spearmanr(p, t)[0]) if ok else None}


def auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ci(score, label, groups=None, n_boot=1000, seed=0) -> dict:
    """AUROC with a cluster bootstrap over groups (uploaders) when given."""
    a = auroc(score, label)
    if a is None:
        return {"auroc": None, "n_pos": int((label == 1).sum()), "n_neg": int((label == 0).sum())}
    rng = np.random.default_rng(seed)
    groups = np.arange(len(score)) if groups is None else np.asarray(groups)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug))])
        v = auroc(score[ix], label[ix])
        if v is not None:
            bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (None, None)
    return {"auroc": a, "ci95": [float(lo), float(hi)] if bs else None, "n_pos": int((label == 1).sum()),
            "n_neg": int((label == 0).sum())}


def pairwise(score: np.ndarray, attrs: np.ndarray, pairs) -> dict:
    ia, ib, at, sign, _ = pairs
    if len(ia) == 0:
        return {"n": 0}
    va = np.where(at < 0, score[ia], attrs[ia, np.maximum(at, 0)])
    vb = np.where(at < 0, score[ib], attrs[ib, np.maximum(at, 0)])
    ok = (va - vb) * sign > 0
    per = {("film_score" if a < 0 else ATTRIBUTES[a]): {"n": int((at == a).sum()), "acc": float(ok[at == a].mean())}
           for a in np.unique(at)}
    return {"n": int(len(ia)), "acc": float(ok.mean()), "per_attribute": per}


def _split_pairs(pairs_by_source: dict, split: np.ndarray, name: str):
    out = {}
    for src, (ia, ib, at, sign, w) in pairs_by_source.items():
        m = (split[ia] == name) & (split[ib] == name) if len(ia) else np.zeros(0, bool)
        out[src] = (ia[m], ib[m], at[m], sign[m], w[m])
    return out


def medium_eval(pred_film: np.ndarray, pred_score: np.ndarray, rows: list[dict], mask: np.ndarray,
                extra_scores: dict | None = None) -> dict:
    """Film-scan vs digital on held-out ORIGINALS. Also the hard subset: film scans vs digital
    photos that their owners tagged as filtered (VSCO/Instagram/vintage...)."""
    sel = [i for i in np.where(mask)[0] if rows[i]["edit_type"] == "original"
           and rows[i].get("medium") in FILM_MEDIA | DIGITAL_MEDIA]
    if not sel:
        return {"n": 0}
    sel = np.array(sel)
    y = np.array([1 if rows[i]["medium"] in FILM_MEDIA else 0 for i in sel])
    grp = [rows[i].get("group_id", rows[i]["family_id"]) for i in sel]
    out = {"film_capture_head": auroc_ci(pred_film[sel], y, grp),
           "film_score_output": auroc_ci(pred_score[sel], y, grp),
           "film_capture_ece": expected_calibration_error(pred_film[sel], y)}
    for k, v in (extra_scores or {}).items():
        out[k] = auroc_ci(v[sel], y, grp)
    hard = np.array([rows[i]["medium"] in FILM_MEDIA | {"digital_filtered", "smartphone_filtered"} for i in sel])
    if hard.any():
        out["hard_subset_film_vs_filtered_digital"] = {
            "film_capture_head": auroc_ci(pred_film[sel][hard], y[hard], np.array(grp)[hard]),
            "film_score_output": auroc_ci(pred_score[sel][hard], y[hard], np.array(grp)[hard])}
    by = defaultdict(list)
    for i in sel:
        by[rows[i]["medium"]].append(pred_score[i])
    out["mean_film_score_by_medium"] = {k: round(100 * float(np.mean(v)), 1) for k, v in sorted(by.items())}
    return out


def evaluate(pred: dict, data, manifest: list[dict], split: str = "test", pairs_by_source: dict | None = None,
             label_sources: list[str] | None = None, extra_medium_scores: dict | None = None) -> dict:
    t = data.t
    m = data.split == split
    src = np.array(t.source)
    res = {"split": split, "n_images": int(m.sum()), "teacher_agreement": {}}
    for s in sorted(set(src[m]) - {"none"}):
        mm = m & (src == s)
        res["teacher_agreement"][s] = {
            "film_score": regression(pred["score"][mm], t.score[mm], t.score_w[mm]),
            "attributes": {a: regression(pred["attrs"][mm, j], t.attrs[mm, j], t.attrs_w[mm, j])
                           for j, a in enumerate(ATTRIBUTES)}}
    res["pairwise"] = {s: pairwise(pred["score"], pred["attrs"], p)
                       for s, p in _split_pairs(pairs_by_source or {}, data.split, split).items()}
    res["capture_medium"] = medium_eval(pred["film_capture"], pred["score"], manifest, m, extra_medium_scores)
    cat, subj = defaultdict(list), defaultdict(list)
    for i in np.where(m)[0]:
        r = manifest[i]
        cat[EVAL_CATEGORY.get(r["edit_type"], r["edit_type"])].append(pred["score"][i])
        if r["edit_type"] == "original":
            for s_ in (r.get("subjects") or ["other"]):
                subj[s_].append(pred["score"][i])
    res["mean_film_score_by_edit_category"] = {k: round(100 * float(np.mean(v)), 1)
                                               for k, v in sorted(cat.items(), key=lambda kv: -np.mean(kv[1]))}
    res["mean_film_score_by_subject_originals"] = {k: {"n": len(v), "score": round(100 * float(np.mean(v)), 1)}
                                                   for k, v in sorted(subj.items())}
    if "score_members" in pred and pred["score_members"].shape[0] > 1:
        sd = pred["score_members"][:, m].std(0) * 100
        res["seed_stability_points"] = {"mean_std": float(sd.mean()), "p95_std": float(np.percentile(sd, 95))}
    res["human_agreement"] = ("unverified: no human labels" if "human" not in (label_sources or [])
                              else "see human_eval section")
    return res


def failure_cases(pred: dict, data, manifest: list[dict], split: str = "test", k: int = 8) -> list[dict]:
    """Largest disagreements with the highest-priority available target (for inspection)."""
    t = data.t
    m = (data.split == split) & (t.score_w > 0)
    idx = np.where(m)[0]
    err = np.abs(pred["score"][idx] - t.score[idx])
    out = []
    for i in idx[np.argsort(-err)[:k]]:
        r = manifest[i]
        out.append({"image_id": r["image_id"], "path": r["path"], "edit_type": r["edit_type"],
                    "medium": r.get("medium"), "pred": round(100 * float(pred["score"][i]), 1),
                    "target": round(100 * float(t.score[i]), 1), "target_source": t.source[i]})
    return out


def _thumb(path: str, side: int = 160) -> str:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


CSS = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--bd:#ddd;--hd:#f4f4f6}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#9a9aa0;--bd:#333;--hd:#1f1f23}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}
table{border-collapse:collapse;margin:8px 0 20px;width:100%}td,th{border:1px solid var(--bd);padding:4px 8px;text-align:left}
th{background:var(--hd)}.mut{color:var(--mut)}.cards{display:flex;flex-wrap:wrap;gap:10px}
.card{border:1px solid var(--bd);padding:6px;width:170px;font-size:12px}pre{white-space:pre-wrap;overflow-x:auto}"""


def html_report(sections: dict, out_path: Path, failures: list[dict] | None = None, title: str = "FilmLook eval") -> None:
    def tbl(d, depth=0):
        if not isinstance(d, dict):
            return html.escape(json.dumps(d) if not isinstance(d, str) else d)
        rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{tbl(v, depth + 1)}</td></tr>" for k, v in d.items())
        return f"<table>{rows}</table>"

    body = [f"<h1>{html.escape(title)}</h1>",
            "<p class=mut>Teacher agreement is NOT human accuracy. Capture-medium AUROC uses EXIF/scanner metadata "
            "(independent of teachers). Human agreement is unverified unless stated.</p>"]
    for name, sec in sections.items():
        body.append(f"<h2>{html.escape(name)}</h2>{tbl(sec)}")
    if failures:
        body.append("<h2>Largest disagreements (test)</h2><div class=cards>")
        for f in failures:
            body.append(f"<div class=card><img src='{_thumb(f['path'])}' width=160><br>{html.escape(f['edit_type'])} "
                        f"{html.escape(str(f.get('medium') or ''))}<br>pred {f['pred']} / target {f['target']} "
                        f"({html.escape(f['target_source'])})</div>")
        body.append("</div>")
    doc = (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
           f"<title>FilmLook Evaluation</title><style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc)
    Path(out_path).with_suffix(".json").write_text(json.dumps({"sections": sections, "failures": failures}, indent=1,
                                                              default=str))
