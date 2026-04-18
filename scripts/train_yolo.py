"""Train YOLOv8-seg on the CarDD dataset. Params tuned for 8GB VRAM."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda


def main() -> None:
    require_cuda()
    from ultralytics import YOLO

    cfg = load_config()
    data_yaml = Path(cfg["yolo"]["data_yaml"])
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO data YAML missing: {data_yaml}. Run scripts/prepare_yolo_data.py first."
        )

    project = Path(cfg["outputs"]["models"]) / "yolo_runs"
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg["yolo"]["model"])
    model.train(
        data=str(data_yaml),
        epochs=cfg["yolo"]["epochs"],
        imgsz=cfg["yolo"]["imgsz"],
        batch=cfg["yolo"]["batch"],
        device=0,
        project=str(project),
        name="cardd_seg",
        patience=cfg["yolo"]["patience"],
        cos_lr=cfg["yolo"]["cos_lr"],
        close_mosaic=cfg["yolo"]["close_mosaic"],
        plots=True,
        exist_ok=True,
    )
    print(f"Best weights: {project / 'cardd_seg/weights/best.pt'}")


if __name__ == "__main__":
    main()
