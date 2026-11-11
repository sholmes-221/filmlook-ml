"""Live engine (method D, what production runs) on exactly the renders the C08 models were tested on.

    python -m research.c08.engine_baseline   # -> reports/c08/engine_baseline.json
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool

import numpy as np

from research.c08.experiments import PAIRS, REPORTS, bootstrap, norm_id
from research.c08.pairs import _vary, render, select_luts
from research.lut.synthetic_library import load_photo

_E = {}


def _engine():
    if "e" not in _E:
        from research.lut.engine.engine import FilmStyleEngine
        _E["e"] = FilmStyleEngine()
    return _E["e"]


def _rank(img):
    o = _engine().analyze(img, top_k=14)
    return [s["id"] for s in o["styles"]], bool(o["uncertain"])


def _pool_job(row):
    return _rank(row["path"])


def _src_job(job):
    luts = [L for L in select_luts() if L["source"] in ("owner_presets", "identity")]
    src = load_photo(job["path"], 512)
    out = []
    for L in luts:
        seed = int(hashlib.sha1(f"{job['source_id']}:{L['id']}:1".encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.default_rng(seed)
        v = _vary(rng)
        x = render(src, L["lut"], v, rng)
        out.append((norm_id(L["name"]), job["source_id"], _rank(x)))
    return out


def summary(truth, ranks, groups, unc):
    t1 = np.array([r[0] == t for t, r in zip(truth, ranks)]); t3 = np.array([t in r[:3] for t, r in zip(truth, ranks)])
    return {"n": len(truth), "top1": float(t1.mean()), "top1_ci": bootstrap(t1, np.array(groups)), "top3": float(t3.mean()), "uncertain_rate": float(np.mean(unc))}


def main():
    pool_rows = [json.loads(l) for l in open(PAIRS / "extra_synth_test" / "meta.jsonl")]
    test_jobs = [j for j in json.loads((PAIRS / "sources.json").read_text()) if j["split"] == "test"]
    with Pool(24) as p:
        pr = p.map(_pool_job, pool_rows, chunksize=8)
        sr = [x for chunk in p.map(_src_job, test_jobs, chunksize=1) for x in chunk]
    rep = {"unseen_image_pool": summary([norm_id(r["preset"]) for r in pool_rows], [r[0] for r in pr], [r["scene_id"] for r in pool_rows], [r[1] for r in pr]),
           "test_sources_varied": summary([t for t, _, _ in sr], [r[0] for _, _, r in sr], [s for _, s, _ in sr], [r[1] for _, _, r in sr])}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "engine_baseline.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in rep.items()}))


if __name__ == "__main__":
    main()
