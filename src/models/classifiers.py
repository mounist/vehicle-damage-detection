"""Factory for the three binary-classification backbones.

All models expose the same forward signature: ``(B, 3, 224, 224) -> (B, num_classes)``.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models as tv_models


class CarDamageCNN(nn.Module):
    """Notebook baseline: 3-block Conv + ReLU + MaxPool, then MLP head."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(self.features(x)))


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with the final fc swapped for ``num_classes``. All layers trainable."""
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ViT-B/16 via timm. Falls back to torchvision if timm isn't installed."""
    try:
        import timm
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    except ImportError:
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> Tuple[nn.Module, float]:
    """Return (model, suggested_lr) for the given name."""
    name = name.lower()
    if name in ("cnn", "custom_cnn"):
        return CarDamageCNN(num_classes), 1e-3
    if name in ("resnet", "resnet18"):
        return build_resnet18(num_classes, pretrained), 1e-4
    if name == "vit":
        return build_vit(num_classes, pretrained), 1e-4
    raise ValueError(f"unknown model: {name}")
