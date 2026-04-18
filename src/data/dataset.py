"""Binary classification dataset builders.

Takes an ImageFolder-style directory ``data1a/{training,validation}`` and
carves a held-out *test* split out of the original validation set using a
fixed seed, so metrics are comparable across runs and across models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .transforms import build_transforms


@dataclass
class ClassificationSplits:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    class_names: List[str]

    @property
    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _split_val_test(val_ds: datasets.ImageFolder, test_frac: float, seed: int) -> Tuple[Subset, Subset]:
    """Stratified-by-class split of the validation ImageFolder into (val, test)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    targets = np.array(val_ds.targets)
    test_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_frac))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:].tolist())
    return Subset(val_ds, sorted(val_idx)), Subset(val_ds, sorted(test_idx))


def build_splits(cfg: dict) -> ClassificationSplits:
    """Build train / val / test datasets according to cfg['classification']."""
    cls_cfg = cfg["classification"]
    root = Path(cls_cfg["data_root"])
    t = build_transforms(cls_cfg["image_size"])

    train_ds = datasets.ImageFolder(root / cls_cfg["train_dir"], transform=t["training"])
    val_full = datasets.ImageFolder(root / cls_cfg["val_dir"], transform=t["eval"])

    val_ds, test_ds = _split_val_test(
        val_full, cls_cfg["test_split_frac"], cls_cfg["test_split_seed"]
    )

    return ClassificationSplits(
        train=train_ds, val=val_ds, test=test_ds, class_names=train_ds.classes,
    )


def build_loaders(splits: ClassificationSplits, cfg: dict) -> Dict[str, DataLoader]:
    """DataLoaders matching the notebook: shuffle train, no shuffle on val/test."""
    cls_cfg = cfg["classification"]
    common = dict(batch_size=cls_cfg["batch_size"], num_workers=cls_cfg["num_workers"])
    return {
        "train": DataLoader(splits.train, shuffle=True, **common),
        "val": DataLoader(splits.val, shuffle=False, **common),
        "test": DataLoader(splits.test, shuffle=False, **common),
    }
