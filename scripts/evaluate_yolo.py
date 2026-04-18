"""Evaluate the trained YOLO model: per-class metrics, confidence sweep, FP/FN viz."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda


def find_best_weights(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("yolo_runs/*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    require_cuda()
    from ultralytics import YOLO

    from src.evaluation.yolo_error_analysis import (
        confidence_threshold_sweep,
        find_fp_fn_examples,
        per_class_metrics,
        plot_confidence_tradeoff,
    )

    cfg = load_config()
    models_dir = Path(cfg["outputs"]["models"])
    figs_dir = Path(cfg["outputs"]["figures"]) / "error_analysis"
    reports_dir = Path(cfg["outputs"]["reports"])
    figs_dir.mkdir(parents=True, exist_ok=True)

    best = find_best_weights(models_dir)
    if best is None:
        print("[skip YOLO eval] no best.pt found under outputs/models/yolo_runs/.")
        return
    print(f"Loading weights: {best}")

    data_yaml = cfg["yolo"]["data_yaml"]
    if not Path(data_yaml).is_file():
        print(f"[skip YOLO eval] data YAML missing: {data_yaml}")
        return

    model = YOLO(str(best))
    val_res = model.val(data=data_yaml, split="val", device=0, conf=0.25, iou=0.60, imgsz=cfg["yolo"]["imgsz"])

    per_cls = per_class_metrics(val_res, cfg["yolo"]["class_names"])
    per_cls.to_csv(reports_dir / "yolo_per_class_metrics.csv", index=False)
    print(per_cls.to_string(index=False))

    sweep = confidence_threshold_sweep(model, data_yaml, cfg["yolo"]["imgsz"])
    sweep.to_csv(reports_dir / "yolo_confidence_sweep.csv", index=False)
    plot_confidence_tradeoff(sweep, figs_dir / "yolo_confidence_tradeoff.png")

    val_img_dir = Path(cfg["yolo"]["yolo_root"]) / "images" / "val"
    val_label_dir = Path(cfg["yolo"]["yolo_root"]) / "labels" / "val"
    if val_img_dir.is_dir() and val_label_dir.is_dir():
        examples = find_fp_fn_examples(
            model, val_img_dir, val_label_dir, figs_dir,
            class_names=cfg["yolo"]["class_names"],
            conf=0.25, imgsz=cfg["yolo"]["imgsz"],
        )
        print(f"Saved {len(examples['fp'])} FP + {len(examples['fn'])} FN examples in {figs_dir}")


if __name__ == "__main__":
    main()
