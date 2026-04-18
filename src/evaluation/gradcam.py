"""Grad-CAM overlays for ResNet-18 and ViT-B/16.

Requires ``pip install grad-cam``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from src.data.transforms import denormalize


def _resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Last conv block before avgpool.
    return model.layer4[-1]


def _vit_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Last transformer block's LayerNorm (timm naming). Fallback for torchvision.
    if hasattr(model, "blocks"):
        return model.blocks[-1].norm1
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder.layers[-1].ln_1
    raise ValueError("Unable to locate ViT target layer")


def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """Drop the CLS token and reshape tokens -> (B, C, H, W) so pytorch-grad-cam can upsample."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def generate_cam_images(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_kind: str,
    out_dir: Path,
    class_names: List[str],
    max_correct: int = 3,
    max_wrong: int = 3,
) -> List[Path]:
    """Save up to ``max_correct`` correct + ``max_wrong`` incorrect Grad-CAM overlays.

    Returns the list of saved file paths.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise ImportError("`grad-cam` not installed. `pip install grad-cam`.") from e
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    if model_kind == "resnet18":
        target_layers = [_resnet_target_layer(model)]
        reshape: Optional[Callable] = None
    elif model_kind == "vit":
        target_layers = [_vit_target_layer(model)]
        reshape = _vit_reshape_transform
    else:
        raise ValueError(f"Grad-CAM not configured for model: {model_kind}")

    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape)

    correct_saved = 0
    wrong_saved = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
        for i in range(x.size(0)):
            is_correct = bool(preds[i].item() == y[i].item())
            if is_correct and correct_saved >= max_correct:
                continue
            if (not is_correct) and wrong_saved >= max_wrong:
                continue

            grayscale = cam(input_tensor=x[i:i+1], targets=None)[0]
            rgb = denormalize(x[i])
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb)
            axes[0].set_title(f"true={class_names[y[i].item()]}")
            axes[0].set_axis_off()
            axes[1].imshow(overlay)
            axes[1].set_title(f"pred={class_names[preds[i].item()]}  ({'OK' if is_correct else 'WRONG'})")
            axes[1].set_axis_off()
            fig.tight_layout()
            tag = "correct" if is_correct else "wrong"
            idx = correct_saved if is_correct else wrong_saved
            path = out_dir / f"{model_kind}_{tag}_{idx:02d}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
            if is_correct:
                correct_saved += 1
            else:
                wrong_saved += 1
            if correct_saved >= max_correct and wrong_saved >= max_wrong:
                return saved
    return saved
