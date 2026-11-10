"""Public API of the film-style engine.

    engine = FilmStyleEngine(db_dir=None, method=None)   # db_dir defaults to research/lut/db; method to engine_config.json
    result = engine.analyze(image, top_k=3)              # image: path | bytes | numpy uint8 RGB HxWx3 | PIL image

Result (dict, JSON-serialisable):
  styles: [{id, name, similarity (0-100, relative, not a probability), explanation, key_features, label_basis}]
  uncertain: bool, uncertainty_reason: str | None   -> show "uncertain / no clear match" when true
  method: which matching approach produced the ranking; distances: raw distances per candidate id
  photo: measured descriptors that matter for the explanation (black level, shadow / midtone / highlight cast, chroma, contrast)
  texture: local contrast, grain / noise indicator, edge density (information only; a colour LUT cannot describe them)
  timing_ms
Memory: the database is ~5 MB on disk; the engine holds the lattices (float16) and the reference bank in RAM.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from research.lut.engine.descriptors import NAMES, TEXTURE_IDX, describe
from research.lut.engine.lut_db import LutDB
from research.lut.engine.matching import APPROACHES, distances, similarity_0_100

_NAME_IDX = {n: i for i, n in enumerate(NAMES)}
DEFAULT_CONFIG = {"method": "CW", "margin_min": 0.08, "abs_distance_max": None, "min_chroma_p50": 0.04, "none_id": "none"}


def load_image(src) -> np.ndarray:
    """-> uint8 RGB HxWx3 with EXIF orientation applied (PIL if available, else OpenCV)."""
    if isinstance(src, np.ndarray):
        if src.ndim != 3 or src.shape[2] != 3:
            raise ValueError("expected HxWx3 RGB array")
        return src.astype(np.uint8) if src.dtype != np.uint8 else src
    try:
        from PIL import Image, ImageOps
        im = Image.open(io.BytesIO(src)) if isinstance(src, (bytes, bytearray)) else (src if hasattr(src, "convert") else Image.open(src))
        if im.format == "JPEG":
            im.draft("RGB", (1024, 1024))
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except ImportError:  # pragma: no cover
        import cv2
        buf = np.frombuffer(src, np.uint8) if isinstance(src, (bytes, bytearray)) else np.fromfile(str(src), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cannot decode image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FilmStyleEngine:
    def __init__(self, db_dir=None, method: str | None = None):
        from research.lut.engine import DEFAULT_DB_DIR
        self.db_dir = Path(db_dir or DEFAULT_DB_DIR)
        self.db = LutDB(self.db_dir)
        cfg_path = self.db_dir / "engine_config.json"
        self.config = {**DEFAULT_CONFIG, **(json.loads(cfg_path.read_text()) if cfg_path.exists() else {})}
        self.method = method or self.config["method"]
        if self.method not in APPROACHES:
            raise ValueError(f"unknown method {self.method}; choose from {APPROACHES}")

    # ---- explanations ------------------------------------------------------------------------------------------
    def _photo_summary(self, desc: np.ndarray) -> dict:
        g = lambda n: float(desc[_NAME_IDX[n]])  # noqa: E731
        return {"black_level_8bit": round(255 * g("black_level"), 1), "white_level_8bit": round(255 * g("white_level"), 1),
                "shadow_cast_ab": [round(20 * g("shadow_a"), 1), round(20 * g("shadow_b"), 1)],
                "midtone_cast_ab": [round(20 * g("mid_a"), 1), round(20 * g("mid_b"), 1)],
                "highlight_cast_ab": [round(20 * g("high_a"), 1), round(20 * g("high_b"), 1)],
                "chroma_p50": round(50 * g("chroma_p50"), 1), "colourfulness": round(g("colourfulness"), 3),
                "contrast_L_std": round(100 * g("contrast_L_std"), 1), "dynamic_range_L": round(100 * g("dynamic_range"), 1),
                "midtone_share": round(g("midtone_share"), 3), "warm_cool_b_mean": round(50 * g("warm_b_mean"), 1)}

    def _texture_summary(self, desc: np.ndarray) -> dict:
        t = {NAMES[i]: round(float(desc[i]), 5) for i in TEXTURE_IDX}
        t["note"] = "texture indicators are measured on the photo for information only; a colour LUT cannot describe grain, halation or sharpness"
        return t

    def _key_features(self, style: dict, photo: dict) -> list[str]:
        d = style["descriptors"]
        feats = []
        feats.append(f"style black lift {d['clipping']['black_lift_8bit']:.0f}/255 vs photo black level {photo['black_level_8bit']:.0f}/255")
        sb = d["shadow_colour_bias"]
        feats.append(f"style shadow bias (a,b)=({sb['a']:+.1f},{sb['b']:+.1f}) vs photo ({photo['shadow_cast_ab'][0]:+.1f},{photo['shadow_cast_ab'][1]:+.1f})")
        mb = d["midtone_colour_bias"]
        feats.append(f"style midtone bias (a,b)=({mb['a']:+.1f},{mb['b']:+.1f}) vs photo ({photo['midtone_cast_ab'][0]:+.1f},{photo['midtone_cast_ab'][1]:+.1f})")
        feats.append(f"style chroma x{d['gamut_compression']['chroma_mean_ratio']:.2f}, midtone contrast {d['luminance_response']['midtone_contrast']:.2f}")
        return feats

    # ---- main ----------------------------------------------------------------------------------------------------
    def analyze(self, image, top_k: int = 3, method: str | None = None, all_methods: bool = False) -> dict:
        t0 = time.perf_counter()
        img = load_image(image)
        desc = describe(img)
        t1 = time.perf_counter()
        method = method or self.method
        dist = distances(self.db, img, desc, approaches=APPROACHES if all_methods else (method,))
        d = dist[method]
        sim = similarity_0_100(d)
        order = np.argsort(d)
        photo = self._photo_summary(desc)
        ids = self.db.ids
        cfg = self.config
        # uncertainty: near-monochrome photo, weak margin between the two best candidates, or an out-of-range best distance
        reason = None
        chroma_p50 = float(desc[_NAME_IDX["chroma_p50"]])
        scale = max(float(np.median(d) - d.min()), 1e-6)
        margin = float((d[order[1]] - d[order[0]]) / scale)
        if chroma_p50 < cfg["min_chroma_p50"]:
            reason = "near-monochrome photo: colour styles cannot be told apart"
        elif cfg.get("abs_distance_max") is not None and d[order[0]] > cfg["abs_distance_max"]:
            reason = "photo is far from every reference rendering (unknown style or unusual content)"
        elif margin < cfg["margin_min"]:
            reason = "the two best candidates are too close to separate"
        styles = []
        for i in order[:top_k]:
            if ids[i] == cfg["none_id"]:
                styles.append({"id": "none", "name": "no film preset (untouched digital look)", "similarity": round(float(sim[i]), 1),
                               "explanation": "closest to the untouched reference renderings", "key_features": [], "label_basis": "reference"})
                continue
            st = self.db.styles[i]
            styles.append({"id": st["id"], "name": st["name"], "similarity": round(float(sim[i]), 1),
                           "explanation": "; ".join(st["explanation_phrases"][:4]) if st["explanation_phrases"] else st["approximation"],
                           "key_features": self._key_features(st, photo), "label_basis": st["label_basis"], "approximation": st["approximation"]})
        t2 = time.perf_counter()
        return {"styles": styles, "uncertain": reason is not None, "uncertainty_reason": reason, "method": method, "margin": round(margin, 3),
                "distances": {ids[i]: round(float(v), 4) for i, v in enumerate(d)},
                "all_method_rankings": {m: [ids[j] for j in np.argsort(v)[:3]] for m, v in dist.items()} if all_methods else None,
                "photo": photo, "texture": self._texture_summary(desc),
                "disclaimer": "Similarity is relative to this LUT collection (100 = closest candidate). Styles are LUT-derived aesthetics; "
                              "no claim is made about the film stock that exposed the photograph.",
                "timing_ms": {"descriptors": round(1000 * (t1 - t0), 1), "matching": round(1000 * (t2 - t1), 1), "total": round(1000 * (t2 - t0), 1)}}


    # ---- film-likeness --------------------------------------------------------------------------------------------
    def film_likeness(self, image) -> dict:
        """Deterministic scan-likeness estimate (0-100) with per-feature evidence; see engine/film_likeness.py."""
        if not hasattr(self, "_fl"):
            from research.lut.engine.film_likeness import FilmLikeness
            path = self.db_dir / "film_likeness.json"
            self._fl = FilmLikeness(path) if path.exists() else None
        if self._fl is None:
            return {"estimate": None, "band": "unavailable", "confidence": "low", "evidence": [], "note": "film_likeness.json not found in the database directory"}
        img = load_image(image)
        return self._fl.estimate(img)


def analyze_image(image, **kw) -> dict:
    return FilmStyleEngine().analyze(image, **kw)
