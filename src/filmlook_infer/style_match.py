"""Photographer style matching at serving time (CPU, no torch): DINOv2-small via ONNX Runtime, a compact reference
index, the Visual DNA knowledge graph, and the fusion / explanation rules.

The index directory (built by ``scripts/refdb_export_index.py``) holds ``manifest.json`` with the encoder and
preprocessing strings and the sha256 of every file; ``StyleMatcher`` refuses to run if a file does not match, and
refuses a query vector whose encoder string differs (``EncoderMismatch``). No reference images are needed.

Signals available at serving time: ``dinov2`` (always) and ``dna`` (when the caller passes detected tags, e.g. mapped
from another model's scene words, labelled as such). The SigLIP semantic signal is not run here (too large for the
production instance); the fusion renormalises over the signals present, so a single-signal result is never 'strong'.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from filmlook_infer.pixels import dinov2_pixels

PREPROCESSING = "dinov2_pixels: shortest side 256 bicubic, centre crop 224, ImageNet mean/std"


class EncoderMismatch(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StyleMatcher:
    def __init__(self, index_dir: Path, verify: bool = True, threads: int = 1):
        self.dir = Path(index_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if verify:
            for name, digest in self.manifest["files"].items():
                if _sha(self.dir / name) != digest:
                    raise ValueError(f"index file changed: {name}")
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.dir / "dinov2_small_cls.onnx"), so, providers=["CPUExecutionProvider"])
        self.vectors = np.load(self.dir / "vectors_cls.f16.npy").astype(np.float32)
        db = sqlite3.connect(self.dir / "works.sqlite"); db.row_factory = sqlite3.Row
        self.works = [dict(r) for r in db.execute("SELECT * FROM works ORDER BY row")]
        db.close()
        self.candidates = sorted({w["photographer_id"] for w in self.works})
        self.labels = np.array([self.candidates.index(w["photographer_id"]) for w in self.works])
        self.cal = self.manifest["calibration"]

    @property
    def encoder(self) -> str:
        return self.manifest["encoder"]

    def embed(self, img: Image.Image) -> np.ndarray:
        x = dinov2_pixels(img.convert("RGB"), 224)[None].astype(np.float32)
        v = self.session.run(None, {"pixel_values": x})[0][0]
        return v / np.linalg.norm(v)

    def dinov2_scores(self, v: np.ndarray, k: int | None = None) -> tuple[np.ndarray, list[dict]]:
        k = k or int(self.cal["knn_k"])
        sims = self.vectors @ v
        idx = np.argsort(-sims)[:k]
        S = np.zeros(len(self.candidates))
        for j in idx:
            S[self.labels[j]] += max(float(sims[j]), 0.0)
        nearest = [{"work_id": self.works[j]["work_id"], "photographer_id": self.works[j]["photographer_id"], "title": self.works[j]["title"],
                    "year": self.works[j]["year"], "source_url": self.works[j]["source_url"], "similarity": round(float(sims[j]), 4)} for j in idx[:3]]
        return S / k, nearest

    def match_vector(self, v: np.ndarray, encoder: str) -> dict:
        if encoder != self.encoder:
            raise EncoderMismatch(f"query encoder {encoder!r} != index encoder {self.encoder!r}")
        S, nearest = self.dinov2_scores(np.asarray(v, np.float32) / np.linalg.norm(v))
        T = float(self.cal["temperature_dinov2"])
        z = S / T; z -= z.max(); p = np.exp(z) / np.exp(z).sum()
        order = np.argsort(-p)
        best = float(p[order[0]])
        return {
            "index_version": self.manifest["index_version"], "encoder": self.encoder,
            "matches": [{"photographer_id": self.candidates[i], "relative_closeness": round(float(p[i]), 4), "raw_knn_score": round(float(S[i]), 4)} for i in order[:3]],
            "abstain": best < float(self.cal["abstain_below_dinov2"]),
            "nearest_works": nearest,
            "note": "relative_closeness is a temperature-scaled share over the indexed photographers, not a probability that the photographer took the photo",
        }

    def match_image(self, img: Image.Image) -> dict:
        return self.match_vector(self.embed(img), self.encoder)
