"""Generate Grad-CAM overlays for ResNet-18 and ViT on the test split."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda
from src.data.dataset import build_loaders, build_splits
from src.evaluation.gradcam import generate_cam_images
from src.models.classifiers import build_model


def main() -> None:
    require_cuda()
    cfg = load_config()
    device = torch.device("cuda")

    splits = build_splits(cfg)
    loaders = build_loaders(splits, cfg)
    out_dir = Path(cfg["outputs"]["figures"]) / "grad_cam"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["outputs"]["models"])

    for name in ("resnet18", "vit"):
        ckpt = models_dir / f"{name}_best.pth"
        if not ckpt.is_file():
            print(f"[skip Grad-CAM] {name} — missing {ckpt}")
            continue
        model, _ = build_model(name, num_classes=len(splits.class_names))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        try:
            saved = generate_cam_images(
                model, loaders["test"], device, model_kind=name,
                out_dir=out_dir, class_names=splits.class_names,
            )
            print(f"{name}: saved {len(saved)} Grad-CAM overlays -> {out_dir}")
        except ImportError as e:
            print(f"[skip Grad-CAM] {name}: {e}")


if __name__ == "__main__":
    main()
