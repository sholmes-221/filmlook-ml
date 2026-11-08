"""The statistics-first analysis MVP: FilmLook's core photo analysis without any external model.

Composition (every piece validated or measured; nothing invented):
- measured statistics and style tags (`features`, `style`): what the image is;
- the supervised student (`checkpoints/student_sup`): a calibrated **film-likeness estimate** from its capture
  head (E58: AUROC 0.86 [0.79, 0.92] on independent film scans vs digital captures, ECE 0.07 after Platt
  scaling) and five rendering characteristics (temperature, saturation, contrast, key, sharpness; 0.81 to
  0.89 accuracy against known edits; grain is weaker, 0.59, and is reported from the measured statistics instead);
- the v1 texture model (`checkpoints/best`): the six rendering attributes and the deterministic strengths /
  weaknesses, kept as *rendering observations*; its scalar score is **not** used as film-likeness (E58: AUROC
  0.49 against real film) and is not returned;
- a rule-based **look suggestion** within a caller-supplied allowlist of defined looks (resemblance wording,
  abstention when the measured character fits none);
- deterministic Chinese and English text.

Output: schema `filmlook_output_v2.json`. Determinism: identical bytes and artifacts give identical JSON.
No network, no LLM, no filesystem paths from the caller except the artifact directories at construction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from filmlook_infer import templates as T
from filmlook_infer.analyzer import FilmLookAnalyzer, InvalidImage, UnsupportedLanguage, SUPPORTED_LANGUAGES
from filmlook_infer.compact import CompactAnalyzer, preprocess
from filmlook_infer.imageio import ImageValidationError, load_image
from filmlook_infer.style import is_monochrome, style_tags
from filmlook_infer.vocab import ATTRIBUTES, COLOR_ATTRIBUTES, STYLE_TAGS

SCHEMA_VERSION = "2.0"

# The defined looks a suggestion may name. Ids match the website's film catalogue; the rules read measured
# character only. Wording is resemblance ("接近 … 的色彩气质"), never "shot on".
LOOKS = {
    "tri_x_like": {"zh": "Tri-X 400 的黑白质感", "en": "the black-and-white character of Tri-X 400"},
    "cinestill_like": {"zh": "CineStill 800T 的夜色与钨丝灯气质", "en": "the night, tungsten-lit character of CineStill 800T"},
    "velvia_like": {"zh": "Velvia 50 的浓烈色彩", "en": "the dense, saturated colour of Velvia 50"},
    "gold_like": {"zh": "Gold 200 的温暖色调", "en": "the warm tones of Gold 200"},
    "portra_like": {"zh": "Portra 的柔和中性色彩", "en": "the soft, neutral colour of Portra"},
}
NO_LOOK = {"zh": "暂时没有找到足够接近的胶卷风格", "en": "No film look is close enough to name yet"}
LOOK_LEAD = {"zh": "这张照片的色彩气质接近{look}。", "en": "This photo's colour character comes close to {look}."}
LOOK_BASIS = {"zh": "依据：{basis}。这是对画面气质的相似描述，不是对所用胶卷的判断。", "en": "Basis: {basis}. A description of resemblance, not a claim about the film used."}
BASIS_WORDS = {
    "monochrome": {"zh": "黑白影像、可见颗粒", "en": "monochrome image with visible grain"},
    "low_key_warm": {"zh": "低调画面、偏暖的人工光色", "en": "low-key image with a warm artificial-light cast"},
    "vivid_contrast": {"zh": "高饱和、高反差", "en": "high saturation and high contrast"},
    "warm_medium": {"zh": "偏暖、饱和度适中", "en": "warm cast with moderate saturation"},
    "neutral_soft": {"zh": "色彩中性偏淡、反差柔和", "en": "neutral, restrained colour and soft contrast"},
}
FILM_BANDS = [  # lower bound of the calibrated estimate, zh, en
    (0.80, "很像真实胶片扫描的成像", "renders much like a real film scan"),
    (0.60, "比较接近胶片扫描的成像", "renders fairly close to a film scan"),
    (0.40, "介于胶片与数码之间", "sits between film and digital rendering"),
    (0.20, "更接近数码成像", "renders closer to a digital capture"),
    (0.00, "明显是数码成像", "renders clearly as a digital capture"),
]
CHAR_WORDS = {
    "temperature": {"warm": ("偏暖", "warm"), "cool": ("偏冷", "cool"), "neutral": ("中性", "neutral"), "monochrome": ("黑白", "monochrome")},
    "saturation": {"high": ("高饱和", "high saturation"), "medium": ("饱和度适中", "moderate saturation"), "low": ("低饱和", "low saturation"), "monochrome": ("无彩色", "no colour")},
    "contrast": {"high": ("高反差", "high contrast"), "medium": ("反差适中", "moderate contrast"), "low": ("低反差", "low contrast")},
    "key": {"high_key": ("高调", "high key"), "balanced": ("明暗均衡", "balanced tones"), "low_key": ("低调", "low key")},
    "sharpness": {"crisp": ("锐利", "crisp"), "moderate": ("清晰度适中", "moderately sharp"), "soft": ("柔和", "soft")},
}
GRAIN_WORDS = {"no_visible_grain": ("几乎没有可见颗粒", "almost no visible grain"), "fine_grain": ("细颗粒", "fine grain"), "coarse_grain": ("粗颗粒", "coarse grain")}
DISCLAIMER = {
    "zh": "本结果由本地模型和画面测量得出，不识别相机、胶卷型号或摄影师，也不评判艺术价值。胶片感估计是与真实胶片扫描的相似程度，不是拍摄媒介的证明。",
    "en": "Produced by local models and image measurements. It does not identify the camera, film stock or photographer, and it is not a judgement of artistic merit. The film-likeness estimate is a resemblance to real film scans, not proof of the medium.",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _apply_platt(cal: dict, x: np.ndarray) -> float:
    z = (x - np.array(cal["mu"])) / np.array(cal["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(cal["w"]) + cal["b"]))))


class RetrievalMVP:
    """Load once per process; `analyze` per request."""

    def __init__(self, v1_dir: str | Path, sup_dir: str | Path, artifacts_dir: str | Path, threads: int = 1):
        self.v1 = FilmLookAnalyzer(v1_dir, threads=threads, prefer_int8=False)
        self.sup = CompactAnalyzer(sup_dir, int8=False, threads=threads)
        art = Path(artifacts_dir)
        self.calibration = json.loads((art / "calibration.json").read_text())
        self.artifact_meta = json.loads((art / "meta.json").read_text())
        if self.calibration.get("inputs") != ["sup_p_film"]:
            raise ValueError("this MVP ships the student's capture head as the film-likeness signal; the calibration artifact names another input")
        self._ci = self.sup.meta["outputs"].index("capture_type"); self._fi = self.sup.meta["heads"]["capture_type"].index("film_scan")
        self.versions = {"schema": SCHEMA_VERSION, "v1_model": self.v1.meta["model_version"] + ":" + _sha(Path(v1_dir) / self.v1.onnx_file),
                         "sup_model": self.sup.meta["model_version"] + ":" + _sha(Path(sup_dir) / self.sup.model_file), "artifacts": self.artifact_meta.get("artifact_version", "unknown")}

    # ------------------------------------------------------------------ pieces
    def _sup_heads(self, im) -> tuple[float, dict]:
        outs = self.sup.session.run(None, {"pixel_values": preprocess(im, self.sup.size)})
        heads = {}
        for name, logits in zip(self.sup.meta["outputs"], outs):
            z = logits[0] - logits[0].max(); p = np.exp(z) / np.exp(z).sum(); classes = self.sup.meta["heads"][name]
            heads[name] = {"label": classes[int(np.argmax(p))], "p": float(p.max())}
        p_film = float(np.exp(outs[self._ci][0] - outs[self._ci][0].max())[self._fi] / np.exp(outs[self._ci][0] - outs[self._ci][0].max()).sum())
        return p_film, heads

    @staticmethod
    def suggest_look(f: dict, heads: dict, allowed: Optional[set[str]]) -> dict:
        """One defined look the measured character comes close to, or an abstention. Rules only; no model picks a stock."""
        mono = is_monochrome(f)
        sat, warmth, key, contrast = f["sat_mean"], f["warmth"], heads["key"]["label"], heads["contrast"]["label"]
        grain = "fine_grain" if 0.006 <= f["noise_std"] <= 0.028 else "coarse_grain" if f["noise_std"] > 0.028 else "no_visible_grain"
        candidates: list[tuple[str, str]] = []
        if mono:
            candidates.append(("tri_x_like", "monochrome"))
        else:
            if key == "low_key" and warmth > 0.03:
                candidates.append(("cinestill_like", "low_key_warm"))
            if sat > 0.40 and contrast == "high":
                candidates.append(("velvia_like", "vivid_contrast"))
            if warmth > 0.055 and 0.17 <= sat <= 0.40:
                candidates.append(("gold_like", "warm_medium"))
            if -0.015 <= warmth <= 0.055 and sat < 0.30 and contrast != "high":
                candidates.append(("portra_like", "neutral_soft"))
        for look, basis in candidates:
            if allowed is None or look in allowed:
                return {"look_id": look, "basis": basis, "abstained": False, "grain": grain}
        return {"look_id": None, "basis": None, "abstained": True, "grain": grain}

    # ------------------------------------------------------------------ public
    def analyze(self, image, language: str = "zh", allowed_looks: Optional[set[str]] = None) -> dict:
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguage(f"language must be one of {SUPPORTED_LANGUAGES}")
        raw = self.v1.predict_raw(image)  # validates and loads once; raises InvalidImage
        if hasattr(image, "seek"):
            image.seek(0)
        try:
            im = load_image(image, max_side=self.v1.max_side)
        except ImageValidationError as e:
            raise InvalidImage(str(e)) from e
        f = raw["features"]; mono = is_monochrome(f)
        p_film_raw, heads = self._sup_heads(im)
        p_film = _apply_platt(self.calibration["platt"], np.array([p_film_raw]))
        band = next(b for b in FILM_BANDS if p_film >= b[0])
        attrs = {a: (None if (mono and a in COLOR_ATTRIBUTES) else round(float(raw["attrs"][i]), 2)) for i, a in enumerate(ATTRIBUTES)}
        strengths, weaknesses = T.strengths_weaknesses(attrs, f, language)
        tags = style_tags(f)
        look = self.suggest_look(f, heads, allowed_looks)
        lang_i = 0 if language == "zh" else 1
        chars = {}
        for head in ("temperature", "saturation", "contrast", "key", "sharpness"):
            label = heads[head]["label"]
            if mono and head in ("temperature", "saturation"):
                label = "monochrome"
            chars[head] = {"label": label, "text": CHAR_WORDS[head].get(label, (label, label))[lang_i], "p": round(heads[head]["p"], 3)}
        chars["grain"] = {"label": look["grain"], "text": GRAIN_WORDS[look["grain"]][lang_i], "p": None, "source": "measured"}
        if look["abstained"]:
            look_text = NO_LOOK[language]
        else:
            look_text = LOOK_LEAD[language].format(look=LOOKS[look["look_id"]][language]) + LOOK_BASIS[language].format(basis=BASIS_WORDS[look["basis"]][language])
        film_text = (f"胶片感估计 {round(p_film * 100)}%：{band[1]}。" if language == "zh" else f"Film-likeness estimate {round(p_film * 100)}%: it {band[2]}.")
        char_summary = ("、".join(c["text"] for c in chars.values()) if language == "zh" else ", ".join(c["text"] for c in chars.values()))
        summary = (f"画面{char_summary}。{film_text}" if language == "zh" else f"The image reads as {char_summary}. {film_text}")
        return {
            "schema_version": SCHEMA_VERSION, "versions": self.versions, "language": language,
            "film_likeness": {"estimate": round(p_film, 3), "estimate_pct": int(round(p_film * 100)), "band": band[1] if language == "zh" else band[2],
                              "calibrated": True, "reference": {"auroc": self.calibration["read_auroc"], "ci95": self.calibration["read_ci95"], "ece": self.calibration["read_ece"], "labels": "EXIF / owner provenance (weak)"},
                              "text": film_text, "raw_p_film": round(p_film_raw, 3)},
            "characteristics": chars,
            "rendering_attributes": attrs,  # v1 rubric, 0..1, higher = more natural rendering; observations, not a score
            "style_tags": tags, "style_tag_names": [STYLE_TAGS[t_][language] if language in STYLE_TAGS.get(t_, {}) else t_ for t_ in tags], "monochrome": mono,
            "look_suggestion": {"look_id": look["look_id"], "abstained": look["abstained"], "basis": look["basis"], "text": look_text,
                                "allowed_looks": sorted(allowed_looks) if allowed_looks is not None else None},
            "strengths": strengths, "weaknesses": weaknesses, "summary": summary, "disclaimer": DISCLAIMER[language],
            "not_claimed": ["film_stock_identity", "photographer_identity", "aesthetic_quality", "subject", "mood"],
        }
