"""Model variants (v1).

kind = "texture" : Model A - measured statistics only -> MLP            (no neural image encoder)
kind = "dino"    : Model B - frozen DINOv2-small embedding -> MLP
kind = "fused"   : Model C - DINOv2-small embedding ++ measured statistics -> MLP
(Model D, a distilled student, lives in models/student.py and is only built if B/C beat A.)

Every variant predicts: film score, 6 attributes, and a capture-medium logit (film vs digital, trained
from EXIF metadata). An ensemble of K seeds is exported as ONE graph that also returns the ensemble
spread, which drives the confidence output.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from filmlook_ml.constants import ATTRIBUTES
from filmlook_ml.labeling.features import N_FEATURES

EMB_DIM = 768  # dinov2-small: CLS (384) ++ mean patch token (384)
KINDS = ("texture", "dino", "fused")


class DinoEmbedder(nn.Module):
    def __init__(self, name: str = "facebook/dinov2-small"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.model(pixel_values=pixel_values).last_hidden_state  # post-layernorm
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)

    def freeze(self, unfreeze_last: int = 0) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        if unfreeze_last > 0:
            for blk in self.model.encoder.layer[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.model.layernorm.parameters():
                p.requires_grad = True


class FilmLookHeads(nn.Module):
    def __init__(self, kind: str = "fused", emb_dim: int = EMB_DIM, n_feat: int = N_FEATURES, hidden: int = 256,
                 dropout: float = 0.2, n_attr: int = len(ATTRIBUTES)):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        self.cfg = dict(kind=kind, emb_dim=emb_dim, n_feat=n_feat, hidden=hidden, dropout=dropout, n_attr=n_attr)
        use_emb, use_feat = kind in ("dino", "fused"), kind in ("texture", "fused")
        fdim = 128 if kind == "texture" else 64
        self.feat_proj = nn.Sequential(nn.Linear(n_feat, fdim), nn.GELU()) if use_feat else None
        d_in = (emb_dim if use_emb else 0) + (fdim if use_feat else 0)
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden, 1)
        self.attrs = nn.Linear(hidden, n_attr)
        self.medium = nn.Linear(hidden, 1)
        self.register_buffer("feat_mean", torch.zeros(n_feat))
        self.register_buffer("feat_std", torch.ones(n_feat))
        self.register_buffer("t_medium", torch.ones(1))   # temperature (calibration)

    def forward(self, emb: torch.Tensor | None, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = []
        if self.kind in ("dino", "fused"):
            parts.append(emb)
        if self.feat_proj is not None:
            parts.append(self.feat_proj((feats - self.feat_mean) / self.feat_std))
        h = self.trunk(torch.cat(parts, -1))
        return {"score": torch.sigmoid(self.score(h)).squeeze(-1), "attrs": torch.sigmoid(self.attrs(h)),
                "medium_logit": self.medium(h).squeeze(-1)}


class Ensemble(nn.Module):
    def __init__(self, members: list[FilmLookHeads]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, emb, feats):
        outs = [m(emb, feats) for m in self.members]
        s = torch.stack([o["score"] for o in outs])
        a = torch.stack([o["attrs"] for o in outs])
        med = torch.stack([torch.sigmoid(o["medium_logit"] / m.t_medium) for o, m in zip(outs, self.members)])
        return {"score": s.mean(0), "attrs": a.mean(0), "score_std": s.std(0, unbiased=False),
                "attrs_std": a.std(0, unbiased=False), "film_capture": med.mean(0)}


class ExportModel(nn.Module):
    """Single ONNX graph: (pixel_values?, raw feats) -> ensemble outputs."""

    def __init__(self, ensemble: Ensemble, embedder: DinoEmbedder | None = None):
        super().__init__()
        self.ensemble, self.embedder = ensemble, embedder

    def forward(self, feats: torch.Tensor, pixel_values: torch.Tensor | None = None):
        emb = self.embedder(pixel_values) if self.embedder is not None else None
        o = self.ensemble(emb, feats)
        return o["score"], o["attrs"], o["score_std"], o["attrs_std"], o["film_capture"]


OUTPUT_NAMES = ["score", "attrs", "score_std", "attrs_std", "film_capture"]


def save_ensemble(out_dir: Path, members: list[FilmLookHeads], meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"members": [m.state_dict() for m in members], "cfg": members[0].cfg}, out_dir / "heads.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_ensemble(ckpt_dir: Path, map_location="cpu") -> tuple[Ensemble, dict]:
    ckpt_dir = Path(ckpt_dir)
    blob = torch.load(ckpt_dir / "heads.pt", map_location=map_location, weights_only=False)
    members = []
    for sd in blob["members"]:
        m = FilmLookHeads(**blob["cfg"])
        m.load_state_dict(sd)
        members.append(m.eval())
    return Ensemble(members).eval(), json.loads((ckpt_dir / "meta.json").read_text())
