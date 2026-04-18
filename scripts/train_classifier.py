"""Train one of the three binary classifiers on the data1a dataset.

Usage
-----
    python scripts/train_classifier.py --model cnn
    python scripts/train_classifier.py --model resnet18 --epochs 15
    python scripts/train_classifier.py --model vit --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda
from src.data.dataset import build_loaders, build_splits
from src.models.classifiers import build_model
from src.models.losses import FocalLoss
from src.models.trainer import train_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train binary damage classifier")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet18", "vit"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", type=str, default=None, help="Override config.yaml path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    splits = build_splits(cfg)
    loaders = build_loaders(splits, cfg)
    print(f"Dataset sizes: {splits.sizes} | classes: {splits.class_names}")

    model, default_lr = build_model(args.model, num_classes=len(splits.class_names))
    model = model.to(device)

    lr = args.lr if args.lr is not None else default_lr
    epochs = args.epochs if args.epochs is not None else cfg["classification"]["epochs"]
    criterion = FocalLoss(
        alpha=cfg["classification"]["focal_loss"]["alpha"],
        gamma=cfg["classification"]["focal_loss"]["gamma"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Training {args.model} | epochs={epochs} | lr={lr}")

    train_classifier(
        model=model,
        loaders={"train": loaders["train"], "val": loaders["val"]},
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        save_dir=Path(cfg["outputs"]["models"]),
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
