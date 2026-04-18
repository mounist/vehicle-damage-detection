"""Image transforms for binary damaged/whole classifier.

Kept identical to the original notebook pipeline so reported metrics are
directly comparable across the refactor.
"""
from __future__ import annotations

from typing import Dict

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Training uses aggressive augmentation; val/test use only resize + normalize."""
    return {
        "training": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }


def denormalize(tensor):
    """Invert ImageNet normalization for visualization (HWC numpy in [0,1])."""
    import numpy as np

    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return arr.clip(0, 1)
