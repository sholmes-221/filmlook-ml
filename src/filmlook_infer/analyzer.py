"""FilmLookAnalyzer: CPU inference, ONNX Runtime only (no torch, no network, no LLM).

    from filmlook_infer import FilmLookAnalyzer
    an = FilmLookAnalyzer("path/to/model_dir")         # loads once; reuse across requests
    result = an.analyze("photo.jpg", language="zh")    # dict matching schema/filmlook_output_v1.json

model_dir contains model.onnx (or model.int8.onnx) and meta.json (written by the training repo).
Output is deterministic for identical input bytes and model version (single-threaded ORT by default).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from filmlook_infer import templates as T
from filmlook_infer.features import compute_features, feature_vector
from filmlook_infer.imageio import ImageValidationError, load_image
from filmlook_infer.pixels import dinov2_pixels
from filmlook_infer.style import is_monochrome, style_tags
from filmlook_infer.vocab import (
    ATTRIBUTE_NAMES,
    ATTRIBUTES,
    COLOR_ATTRIBUTES,
    SCHEMA_VERSION,
    STYLE_TAGS,
)

SUPPORTED_LANGUAGES = ("en", "zh")


class FilmLookError(Exception):
    """Base error with a stable machine-readable code."""
    code = "error"


class InvalidImage(FilmLookError):
    code = "invalid_image"


class UnsupportedLanguage(FilmLookError):
    code = "unsupported_language"


def confidence_from(score_std: float, ood_frac: float, cfg: dict) -> float:
    """Heuristic confidence in [0,1]: ensemble agreement and in-distribution features.
    NOT a calibrated probability."""
    c = 1.0 - cfg.get("std_scale", 6.0) * score_std - cfg.get("ood_scale", 1.5) * ood_frac
    return float(np.clip(c, 0.05, 1.0))


class FilmLookAnalyzer:
    def __init__(self, model_path: str | Path, threads: int = 1, prefer_int8: bool = True):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.meta = json.loads((self.model_path / "meta.json").read_text())
        names = (["model.int8.onnx"] if prefer_int8 else []) + ["model.onnx"]
        f = next((self.model_path / n for n in names if (self.model_path / n).exists()), None)
        if f is None:
            raise FileNotFoundError(f"no model.onnx in {self.model_path}")
        self.onnx_file = f.name
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.max_side = int(self.meta.get("max_side", 1280))
        self.crop = int(self.meta.get("texture_crop", 512))
        self.input_size = int(self.meta.get("input_size") or 224)
        self.feat_mean = np.array(self.meta["feat_mean"], np.float32)
        self.feat_std = np.array(self.meta["feat_std"], np.float32)
        self.conf_cfg = self.meta.get("confidence", {})

    # ------------------------------------------------------------------ raw
    def _run(self, px: np.ndarray | None, fv: np.ndarray) -> dict[str, np.ndarray]:
        feed = {"feats": fv.astype(np.float32)}
        if "pixel_values" in self.inputs:
            feed["pixel_values"] = px.astype(np.float32)
        return dict(zip(self.outputs, self.sess.run(None, feed)))

    def predict_raw(self, image) -> dict:
        try:
            im = load_image(image, max_side=self.max_side)
        except ImageValidationError as e:
            raise InvalidImage(str(e)) from e
        f = compute_features(im, self.crop)
        fv = feature_vector(f)[None]
        if not np.isfinite(fv).all():
            raise InvalidImage("could not measure image statistics")
        px = dinov2_pixels(im, self.input_size)[None] if "pixel_values" in self.inputs else None
        o = self._run(px, fv)
        z = np.abs((fv[0] - self.feat_mean) / self.feat_std)
        return {"score": float(o["score"][0]), "attrs": o["attrs"][0], "score_std": float(o["score_std"][0]),
                "attrs_std": o["attrs_std"][0], "ood_frac": float((z > 4).mean()), "features": f}

    # ------------------------------------------------------------------ public
    def analyze(self, image, language: str = "en") -> dict:
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguage(f"language must be one of {SUPPORTED_LANGUAGES}")
        r = self.predict_raw(image)
        f = r["features"]
        mono = is_monochrome(f)
        attrs = {}
        for i, a in enumerate(ATTRIBUTES):
            attrs[a] = None if (mono and a in COLOR_ATTRIBUTES) else round(float(r["attrs"][i]), 2)
        score = int(round(100 * r["score"]))
        conf = round(confidence_from(r["score_std"], r["ood_frac"], self.conf_cfg), 2)
        level = "high" if conf >= 0.75 else "medium" if conf >= 0.5 else "low"
        tags = style_tags(f)
        strengths, weaknesses = T.strengths_weaknesses(attrs, f, language)
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": self.meta["model_version"],
            "language": language,
            "film_score": score,
            "attributes": attrs,
            "attribute_names": {a: ATTRIBUTE_NAMES[a][language] for a in ATTRIBUTES},
            "style_tags": tags,
            "style_tag_names": [STYLE_TAGS[t][language] for t in tags],
            "monochrome": mono,
            "confidence": conf,
            "confidence_level": level,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "summary": T.summary(score, language, level == "low"),
            "disclaimer": T.DISCLAIMER[language],
        }
