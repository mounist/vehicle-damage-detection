"""Two-stage inference pipeline.

Stage 1: a lightweight classifier decides whether damage is present.
Stage 2: YOLOv8-seg localizes and classifies the damage only when Stage 1 says yes.

Why two stages when YOLO alone already outputs "no detections" on clean cars?

- The classifier is ~10-20x cheaper per image. On edge devices or streaming
  workloads, skipping YOLO on obviously clean frames is a real win.
- When the cost of a false positive is high (insurance fraud triage, recall
  campaigns), a second opinion from a simpler decision boundary reduces the
  chance of an over-eager YOLO detection slipping through.

This module deliberately keeps the classifier threshold configurable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms


@dataclass
class ClaimReport:
    file: str
    stage1_label: str
    stage1_prob_damaged: float
    status: str  # "NO DAMAGE FOUND" | "DAMAGE DETECTED" | "DAMAGE DETECTED (contradiction)"
    detections: List[Dict[str, Any]] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "stage1_label": self.stage1_label,
            "stage1_prob_damaged": self.stage1_prob_damaged,
            "status": self.status,
            "detections": self.detections,
        }


class DamageAnalysisSystem:
    """End-to-end pipeline wrapping Stage 1 (classifier) and Stage 2 (YOLO)."""

    def __init__(
        self,
        classifier: torch.nn.Module,
        classifier_class_names: List[str],
        yolo_weights: str | Path,
        device: torch.device,
        image_size: int = 224,
        yolo_imgsz: int = 640,
        yolo_conf: float = 0.25,
        stage1_threshold: float = 0.50,
    ) -> None:
        from ultralytics import YOLO

        self.classifier = classifier.to(device).eval()
        self.class_names = classifier_class_names
        self.device = device
        self.transform = build_transforms(image_size)["eval"]
        self.segmenter = YOLO(str(yolo_weights))
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.stage1_threshold = stage1_threshold

        self.damaged_idx = (
            classifier_class_names.index("00-damage")
            if "00-damage" in classifier_class_names
            else 0
        )

    def _stage1_predict(self, img: Image.Image) -> tuple[str, float]:
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)[0]
        damaged_prob = float(probs[self.damaged_idx].item())
        label = self.class_names[self.damaged_idx] if damaged_prob >= self.stage1_threshold \
            else self.class_names[1 - self.damaged_idx]
        return label, damaged_prob

    def process_claim(self, image_path: str | Path) -> ClaimReport:
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")
        label, damaged_prob = self._stage1_predict(img)

        report = ClaimReport(
            file=image_path.name,
            stage1_label=label,
            stage1_prob_damaged=damaged_prob,
            status="NO DAMAGE FOUND",
        )
        if damaged_prob < self.stage1_threshold:
            return report  # skip YOLO entirely — that's the point of Stage 1

        yolo_res = self.segmenter.predict(
            str(image_path), conf=self.yolo_conf, imgsz=self.yolo_imgsz, verbose=False
        )[0]
        report.annotated_image = yolo_res.plot()
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            for box in yolo_res.boxes:
                report.detections.append({
                    "class": yolo_res.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                })
            report.status = "DAMAGE DETECTED"
        else:
            # Stage 1 flagged damage but YOLO saw nothing — keep a trace in the status
            report.status = "DAMAGE SUSPECTED (no localization)"
        return report


def process_directory(
    system: DamageAnalysisSystem, input_dir: Path, output_dir: Path
) -> Path:
    """Run the pipeline on every image in ``input_dir``; save annotated images + a JSON manifest."""
    from PIL import Image as PILImage

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated").mkdir(exist_ok=True)
    manifest = []
    for img_path in sorted(input_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        report = system.process_claim(img_path)
        if report.annotated_image is not None:
            PILImage.fromarray(report.annotated_image).save(output_dir / "annotated" / img_path.name)
        manifest.append(report.to_json_dict())
    out_json = output_dir / "pipeline_results.json"
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_json
