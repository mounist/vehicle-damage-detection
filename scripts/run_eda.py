"""Run full EDA on both datasets. Saves figures + summary JSON under outputs/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import eda


def main() -> None:
    cfg = load_config()
    figs_dir = Path(cfg["outputs"]["figures"]) / "eda"
    reports_dir = Path(cfg["outputs"]["reports"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"classification": {}, "yolo": {}, "missing": []}

    # ---------------- Binary classification ----------------
    data_root = Path(cfg["classification"]["data_root"])
    if data_root.is_dir():
        dist = eda.class_distribution(data_root)
        dist.to_csv(reports_dir / "eda_classification_distribution.csv", index=False)
        summary["classification"]["distribution"] = dist.to_dict(orient="records")

        eda.plot_class_distribution(dist, figs_dir / "cls_class_distribution.png", "data1a — damaged vs whole")
        eda.sample_grid(data_root, "training", figs_dir / "cls_samples_training.png")
        eda.sample_grid(data_root, "validation", figs_dir / "cls_samples_validation.png")

        size_df = eda.image_size_stats(data_root, "training")
        size_df.describe().to_csv(reports_dir / "eda_classification_imgsize.csv")
        eda.plot_image_size_distribution(size_df, figs_dir / "cls_imgsize_training.png", "training")
        summary["classification"]["image_size_stats"] = size_df.describe().to_dict()

        eda.augmentation_preview(data_root, "training", figs_dir / "cls_augmentation_preview.png",
                                 image_size=cfg["classification"]["image_size"])

        # imbalance summary for Focal Loss motivation
        train_counts = dist[dist["split"] == "training"].set_index("class")["count"]
        summary["classification"]["imbalance_ratio"] = float(train_counts.max() / max(train_counts.min(), 1))
    else:
        summary["missing"].append(f"classification data_root not found: {data_root}")

    # ---------------- YOLO / CarDD ----------------
    coco_root = Path(cfg["yolo"]["coco_root"])
    ann_dir = coco_root / "annotations"
    train_json = next(ann_dir.glob("*train*.json"), None) if ann_dir.is_dir() else None
    val_json = next(ann_dir.glob("*val*.json"), None) if ann_dir.is_dir() else None

    if train_json and val_json:
        train_df = eda.coco_class_distribution(train_json)
        val_df = eda.coco_class_distribution(val_json)
        train_df.to_csv(reports_dir / "eda_yolo_train_distribution.csv", index=False)
        val_df.to_csv(reports_dir / "eda_yolo_val_distribution.csv", index=False)
        eda.plot_yolo_class_distribution(train_df, val_df, figs_dir / "yolo_class_distribution.png")
        eda.coco_sample_grid(val_json, coco_root / "val2017", figs_dir / "yolo_samples_val.png")
        summary["yolo"]["train_distribution"] = train_df.to_dict(orient="records")
        summary["yolo"]["val_distribution"] = val_df.to_dict(orient="records")
        summary["yolo"]["imbalance"] = eda.imbalance_report(train_df)
    else:
        summary["missing"].append(f"CarDD COCO annotations not found under {coco_root}")

    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EDA complete.")
    print(f"  figures -> {figs_dir}")
    print(f"  reports -> {reports_dir}")
    if summary["missing"]:
        print("  MISSING:")
        for m in summary["missing"]:
            print(f"   - {m}")


if __name__ == "__main__":
    main()
