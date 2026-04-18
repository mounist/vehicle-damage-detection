"""Convert CarDD COCO -> YOLOv8-seg layout under the configured yolo_root."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.coco_to_yolo import build_yolo_yaml, coco_to_yolo_labels, copy_images


def main() -> None:
    cfg = load_config()
    coco_root = Path(cfg["yolo"]["coco_root"])
    yolo_root = Path(cfg["yolo"]["yolo_root"])

    if not coco_root.is_dir():
        print(f"[skip] COCO root missing: {coco_root}")
        return

    ann_dir = coco_root / "annotations"
    train_json = next((p for p in ann_dir.glob("*train*.json")), None)
    val_json = next((p for p in ann_dir.glob("*val*.json")), None)
    if not train_json or not val_json:
        print(f"[skip] train/val COCO JSON not found under {ann_dir}")
        return

    (yolo_root / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images/val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/val").mkdir(parents=True, exist_ok=True)

    n_train = copy_images(coco_root / "train2017", yolo_root / "images/train")
    n_val = copy_images(coco_root / "val2017", yolo_root / "images/val")
    print(f"Copied {n_train} train / {n_val} val images")

    names = coco_to_yolo_labels(train_json, yolo_root / "labels/train")
    coco_to_yolo_labels(val_json, yolo_root / "labels/val")
    yaml_path = build_yolo_yaml(yolo_root, names)
    print(f"Wrote {yaml_path}")

    # Drop stale ultralytics label caches so the new labels take effect
    for p in yolo_root.rglob("*.cache"):
        p.unlink()


if __name__ == "__main__":
    main()
