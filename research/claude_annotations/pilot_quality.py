"""Pilot quality beyond schema validity: does the annotation carry information that an independent label confirms?

  python -m research.claude_annotations.pilot_quality   -> reports/claude/pilot_quality.json

1. filmset rows: Claude's 0-4 "film_like_impression" and its "film_rendering" flags against the recorded capture medium
   (film scan vs digital, from EXIF and owner records; never shown to the model). AUROC with a bootstrap over uploaders,
   next to the shipped 36-statistic estimate on the same rows.
2. bank rows: does the annotation vector separate photographers at all? Leave-one-out nearest-neighbour photographer
   accuracy among the annotated works (chance = 1 / photographers), against the same reading for the production embedding
   on the same rows. This is a sanity reading on pilot rows, not the benchmark (which uses the leakage-controlled splits).
3. consistency: how often fields that must agree do (monochrome palette <-> saturation none <-> colour temperature n/a).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research.claude_annotations import pipeline as P  # noqa: E402
from research.claude_annotations import schema as S  # noqa: E402

OUT = ROOT / "reports" / "claude" / "pilot_quality.json"


def main() -> None:
    cheap = P.cfg()["models"]["cheap"]
    report: dict = {}
    # 1. film / digital
    fs = P.annotated("filmset", cheap)
    rows = [json.loads(l) for l in (ROOT / "data/lut/film_likeness/manifest.jsonl").read_text().splitlines()]
    F = np.nan_to_num(np.load(ROOT / "data/lut/film_likeness/features.npy").astype(np.float32))
    ids = [f"filmset:{r['dataset']}:{Path(r['path']).name}" for r in rows]
    sel = [i for i, k in enumerate(ids) if k in fs]
    if len(sel) >= 40:
        y = np.array([rows[i]["label"] for i in sel]); g = np.array([rows[i]["uploader"] for i in sel])
        imp = np.array([int(fs[ids[i]]["annotation"]["film_like_impression"]) for i in sel], dtype=float)
        names = S.feature_names()
        grain = np.array([("visible_grain" in fs[ids[i]]["annotation"]["film_rendering"]) for i in sel], dtype=float)
        clean = np.array([("clean_digital_rendering" in fs[ids[i]]["annotation"]["film_rendering"]) for i in sel], dtype=float)
        # the shipped estimate's own logit on the same rows (weights from the vendored artefact)
        art = json.loads((ROOT.parent / "film-evaluation/filmlook/data/filmstyle/film_likeness.json").read_text())
        z = (F[sel] - np.array(art["mean"])) / np.array(art["std"]); logit = z @ np.array(art["weights"]) + art["bias"]

        def ci(s):
            rng = np.random.default_rng(0); ug = np.unique(g); vals = []
            for _ in range(500):
                idx = np.concatenate([np.flatnonzero(g == u) for u in rng.choice(ug, len(ug))])
                if len(set(y[idx])) == 2: vals.append(roc_auc_score(y[idx], s[idx]))
            return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
        report["filmset"] = {"n": len(sel), "film": int(y.sum()), "digital": int((1 - y).sum()), "uploaders": int(len(set(g))),
                             "claude_impression_auroc": round(float(roc_auc_score(y, imp)), 3), "claude_impression_ci": ci(imp),
                             "claude_grain_flag_auroc": round(float(roc_auc_score(y, grain - clean)), 3),
                             "shipped_36_statistics_auroc": round(float(roc_auc_score(y, logit)), 3), "shipped_ci": ci(logit),
                             "impression_by_label": {"film": dict(Counter(imp[y == 1].astype(int).tolist())), "digital": dict(Counter(imp[y == 0].astype(int).tolist()))}}
    # 2. photographers
    bk = P.annotated("bank", cheap)
    from filmlook_ml.refdb import manifest
    with manifest.connect() as c:
        who = {r["work_id"]: r["photographer_id"] for r in c.execute("SELECT work_id, photographer_id FROM works")}
    keys = sorted(bk); labels = np.array([who[k] for k in keys])
    if len(keys) >= 40:
        A = np.stack([S.vectorise(bk[k]["annotation"]) for k in keys]); A = A - A.mean(0); A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        E = np.load(ROOT / "data/embeddings/photographers/prod.npz"); pos = {w: i for i, w in enumerate(E["work_ids"].tolist())}
        X = E["X"][[pos[k] for k in keys]]; X /= np.linalg.norm(X, axis=1, keepdims=True)

        def loo(M):
            Sm = M @ M.T; np.fill_diagonal(Sm, -9)
            return float((labels[Sm.argmax(1)] == labels).mean())
        report["bank"] = {"n": len(keys), "photographers": int(len(set(labels))), "chance": round(1 / len(set(labels)), 3),
                          "annotation_vector_loo_nn": round(loo(A), 3), "production_embedding_loo_nn": round(loo(X), 3),
                          "both_concatenated_loo_nn": round(loo(np.hstack([X, A]) / np.sqrt(2)), 3)}
    # 3. consistency
    allrows = list(bk.values()) + list(fs.values()); bad = 0
    for r in allrows:
        a = r["annotation"]; mono = a["palette"].startswith("monochrome")
        if mono and (a["saturation"] != "none" or a["colour_temperature"] != "not_applicable"): bad += 1
        if (not mono) and a["saturation"] == "none": bad += 1
    report["consistency"] = {"n": len(allrows), "palette_saturation_temperature_conflicts": bad, "low_confidence_share": round(float(np.mean([r["annotation"]["confidence"] == "low" for r in allrows])), 3) if allrows else None}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
