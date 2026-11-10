"""Dataset access shared by features / train / probe / evaluate: rows, label maps, torch datasets and transforms."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from research.species import common as C

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPLITS = ("train", "val", "test", "unknown_taxa", "unknown_nonwild")


def load_rows() -> list[dict]:
    rows = C.read_jsonl(C.DATA / "dataset.jsonl")
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def abs_path(r: dict) -> Path:
    return Path(r["path"]) if r["path"].startswith("/") else C.DATA / r["path"]


def label_space(rows: list[dict]) -> dict:
    """Classes = encyclopedia species with at least one training image, sorted by id. Genus / family maps from the
    GBIF backbone (taxa.json), never from the encyclopedia text."""
    taxa = {t["id"]: t for t in json.load(open(C.DATA / "taxa.json"))}
    train_ids = sorted({r["species_id"] for r in rows if r["split"] == "train"})
    classes = [{"species_id": s, "gbif_key": taxa[s]["gbif_key"], "canonical": taxa[s].get("canonical"), "genus": taxa[s].get("genus"),
                "family": taxa[s].get("family"), "kind": taxa[s]["kind"]} for s in train_ids]
    return {"classes": classes, "index": {c["species_id"]: i for i, c in enumerate(classes)}}


def eval_transform(size: int = 224):
    return T.Compose([T.Resize(round(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(size: int = 224):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.15, 0.02), T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageRows(Dataset):
    def __init__(self, rows: list[dict], index: dict, tf):
        self.rows, self.index, self.tf = rows, index, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(abs_path(r)).convert("RGB")
        except Exception:  # noqa: BLE001
            im = Image.new("RGB", (224, 224))
        y = self.index.get(r.get("species_id"), -1)
        return self.tf(im), y, r["idx"]


def save_preds(name: str, split: str, idx, logits, **extra) -> Path:
    import numpy as np
    d = C.DATA / "preds"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{split}.npz"
    np.savez_compressed(p, idx=np.asarray(idx), logits=np.asarray(logits, dtype=np.float32), **extra)
    return p


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
