"""Shadow-mode analysis record: every model output in its own field, with its own uncertainty statement.

Research interface for the FilmLook app to call *alongside* its current pipeline (logged, not shown). It never
collapses film authenticity into film aesthetics:
  film_aesthetics      v1 film score + six rendering attributes + measured style tags (deterministic templates)
  film_authenticity    P(film scan) from the EXIF-trained capture-type head; weakly labelled, poorly calibrated
  visual_attributes    deterministic measured attributes with the raw values (badge rules can be audited)
  style_embedding      optional experimental embedding (synthetic-trained); retrieval only, never a label
Dependencies: numpy, Pillow, scipy, onnxruntime. No network, no LLM, no GPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from filmlook_infer.analyzer import FilmLookAnalyzer
from filmlook_infer.compact import CompactAnalyzer, preprocess
from filmlook_infer.features import compute_features
from filmlook_infer.imageio import load_image
from filmlook_infer.style import THRESHOLDS, style_tags

SHADOW_SCHEMA_VERSION = "shadow-1.0"
AUTHENTICITY_NOTE = ("P(film_scan) from a MobileNetV3 head trained on EXIF capture medium. Within-source AUROC 0.78 "
                     "[0.69, 0.88] on 89 test photos, ECE 0.20: not calibrated, not provenance proof.")
BADGE_RULES = {   # deterministic; thresholds from filmlook_infer.style.THRESHOLDS (population quantiles, not human-validated)
    "warm_palette": lambda f: (not f["is_mono"]) and f["warmth"] > THRESHOLDS["warm"],
    "cool_palette": lambda f: (not f["is_mono"]) and f["warmth"] < THRESHOLDS["cool"],
    "muted_colours": lambda f: (not f["is_mono"]) and f["sat_mean"] < THRESHOLDS["muted"],
    "pastel_tones": lambda f: (not f["is_mono"]) and f["sat_mean"] < THRESHOLDS["muted"] and f["y_mean"] > 0.55 and f["y_std"] < 0.2,
    "high_contrast": lambda f: f["y_std"] > THRESHOLDS["high_contrast"],
    "soft_highlights": lambda f: f["clip_high"] < 0.002,
    "deep_shadows": lambda f: f["p01"] < 0.03 and f["clip_low"] < 0.01,
    "fine_grain": lambda f: THRESHOLDS["grain_none"] < f["noise_std"] < THRESHOLDS["grain_coarse"] and f["noise_autocorr1"] > 0.4,
}


class ShadowAnalyzer:
    def __init__(self, film_model_dir: str | Path, authenticity_model_dir: str | Path | None = None,
                 style_model_dir: str | Path | None = None, threads: int = 1):
        self.film = FilmLookAnalyzer(str(film_model_dir), threads=threads)
        self.auth = CompactAnalyzer(authenticity_model_dir, int8=False, threads=threads) if authenticity_model_dir else None
        self.style = None
        if style_model_dir:
            import onnxruntime as ort
            d = Path(style_model_dir)
            self.style_meta = json.loads((d / "meta.json").read_text())
            so = ort.SessionOptions(); so.intra_op_num_threads = threads
            self.style = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])

    def analyze(self, src, language: str = "en") -> dict:
        t0 = time.perf_counter()
        img = load_image(src, max_side=1280)
        feats = compute_features(img)
        out = {"schema_version": SHADOW_SCHEMA_VERSION, "model_versions": {"film_aesthetics": self.film.meta.get("model_version")},
               "film_aesthetics": self.film.analyze(src, language=language),
               "visual_attributes": {"style_tags": style_tags(feats),
                                     "measured": {k: round(float(feats[k]), 5) for k in ("warmth", "sat_mean", "y_mean", "y_std", "p01",
                                                                                           "clip_high", "clip_low", "noise_std", "noise_autocorr1", "lap_var")},
                                     "badge_candidates": sorted(k for k, rule in BADGE_RULES.items() if rule(feats)),
                                     "note": "deterministic measurements; thresholds are population quantiles, not human-validated"}}
        if self.auth:
            a = self.auth.analyze(img)
            cap = a["visual_attributes"]["capture_type"]
            out["model_versions"]["film_authenticity"] = a["model_version"]
            out["film_authenticity"] = {"p_film_scan": cap["scores"].get("film_scan", 0.0), "top": cap["label"],
                                        "scores": cap["scores"], "status": "experimental", "note": AUTHENTICITY_NOTE}
        if self.style is not None:
            x = preprocess(img, self.style_meta.get("input_size", 224))
            names = self.style_meta["outputs"]
            res = dict(zip(names, self.style.run(None, {"pixel_values": x})))
            emb = res["style_embedding"][0]
            out["model_versions"]["style_embedding"] = self.style_meta["model_version"]
            out["style_embedding"] = {"vector": [round(float(v), 5) for v in emb / (np.linalg.norm(emb) + 1e-8)],
                                      "dim": int(len(emb)), "status": "experimental",
                                      "note": "trained on synthetic treatments; use for retrieval only, never as a film-stock or photographer label"}
        out["uncertainty"] = {"film_aesthetics_confidence": out["film_aesthetics"].get("confidence"),
                              "authenticity_calibrated": False, "human_validation": "none"}
        out["timing_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out
