"""Run the two-stage pipeline on a single image or a whole directory.

Usage:
    python scripts/run_pipeline.py --input path/to/image.jpg
    python scripts/run_pipeline.py --input path/to/images_dir/ --output outputs/pipeline_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda
from src.models.classifiers import build_model
from src.pipeline.two_stage import DamageAnalysisSystem, process_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage damage analysis pipeline")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory (dir mode only)")
    parser.add_argument("--classifier", type=str, default="resnet18", choices=["cnn", "resnet18", "vit"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Stage 1 damaged-probability cutoff")
    return parser.parse_args()


def find_yolo_weights(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("yolo_runs/*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained YOLO weights found under outputs/models/yolo_runs/")
    return candidates[0]


def main() -> None:
    args = parse_args()
    require_cuda()
    cfg = load_config()
    device = torch.device("cuda")

    models_dir = Path(cfg["outputs"]["models"])
    cls_ckpt = models_dir / f"{args.classifier}_best.pth"
    if not cls_ckpt.is_file():
        raise FileNotFoundError(f"Classifier weights missing: {cls_ckpt}")
    yolo_weights = find_yolo_weights(models_dir)

    classifier, _ = build_model(args.classifier, num_classes=len(cfg["classification"]["class_names"]))
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device))

    system = DamageAnalysisSystem(
        classifier=classifier,
        classifier_class_names=cfg["classification"]["class_names"],
        yolo_weights=yolo_weights,
        device=device,
        image_size=cfg["classification"]["image_size"],
        yolo_imgsz=cfg["yolo"]["imgsz"],
        stage1_threshold=args.threshold,
    )

    src = Path(args.input)
    if src.is_dir():
        out_dir = Path(args.output or Path(cfg["outputs"]["root"]) / "pipeline_results")
        manifest = process_directory(system, src, out_dir)
        print(f"Wrote manifest: {manifest}")
    else:
        report = system.process_claim(src)
        print(json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
