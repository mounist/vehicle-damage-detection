"""YOLO-side error analysis.

Produces:
- per-class Precision / Recall / F1 / mAP table
- confidence-threshold sweep table (for the tradeoff plot)
- visualizations of false-positive and false-negative predictions
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def per_class_metrics(val_results, class_names: Dict[int, str]) -> pd.DataFrame:
    """Turn an Ultralytics ``val()`` result object into a per-class dataframe.

    Works against the Mask head (segmentation).
    """
    seg = val_results.seg
    rows: List[dict] = []
    for idx, name in class_names.items():
        try:
            p = float(seg.p[idx]) if hasattr(seg, "p") else None
            r = float(seg.r[idx]) if hasattr(seg, "r") else None
            ap50 = float(seg.ap50[idx]) if hasattr(seg, "ap50") else None
            ap = float(seg.ap[idx]) if hasattr(seg, "ap") else None
        except (IndexError, TypeError):
            p = r = ap50 = ap = None
        f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None
        rows.append({
            "class": name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP50": ap50,
            "mAP50_95": ap,
        })
    return pd.DataFrame(rows)


def confidence_threshold_sweep(
    model, data_yaml: str, imgsz: int, thresholds=(0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90)
) -> pd.DataFrame:
    """Sweep confidence thresholds and record overall mask/box metrics."""
    rows = []
    for c in thresholds:
        m = model.val(data=data_yaml, split="val", device=0, conf=c, iou=0.60, verbose=False, imgsz=imgsz)
        rows.append({
            "conf": c,
            "mask_P": float(m.seg.mp),
            "mask_R": float(m.seg.mr),
            "mask_mAP50": float(m.seg.map50),
            "mask_mAP50_95": float(m.seg.map),
            "box_P": float(m.box.mp),
            "box_R": float(m.box.mr),
            "box_mAP50_95": float(m.box.map),
        })
    return pd.DataFrame(rows)


def plot_confidence_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["conf"], df["mask_P"], marker="o", label="Mask Precision")
    ax.plot(df["conf"], df["mask_R"], marker="o", label="Mask Recall")
    ax.plot(df["conf"], df["mask_mAP50_95"], marker="o", label="Mask mAP50-95")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_title("Confidence threshold tradeoff (YOLOv8-seg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_fp_fn_examples(
    model, val_image_dir: Path, label_dir: Path, out_dir: Path,
    class_names: Dict[int, str], conf: float = 0.25, imgsz: int = 640,
    max_fp: int = 5, max_fn: int = 5,
) -> Dict[str, List[Path]]:
    """Scan the val split and save example images where the model over/undershoots."""
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    fp_saved: List[Path] = []
    fn_saved: List[Path] = []

    img_paths = sorted([p for p in val_image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img_path in img_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        n_gt = 0
        if label_path.is_file():
            n_gt = sum(1 for line in label_path.read_text().splitlines() if line.strip())

        results = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]
        n_pred = 0 if r.boxes is None else len(r.boxes)

        if n_gt == 0 and n_pred > 0 and len(fp_saved) < max_fp:
            path = out_dir / f"fp_{len(fp_saved):02d}_{img_path.stem}.png"
            _save_prediction_vis(r, img_path, path, f"FP: predicted {n_pred} object(s) on a clean image")
            fp_saved.append(path)
        elif n_gt > 0 and n_pred == 0 and len(fn_saved) < max_fn:
            path = out_dir / f"fn_{len(fn_saved):02d}_{img_path.stem}.png"
            _save_prediction_vis(r, img_path, path, f"FN: missed {n_gt} ground-truth object(s)")
            fn_saved.append(path)
        if len(fp_saved) >= max_fp and len(fn_saved) >= max_fn:
            break

    return {"fp": fp_saved, "fn": fn_saved}


def _save_prediction_vis(result, img_path: Path, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.imshow(result.plot())
    except Exception:
        ax.imshow(Image.open(img_path).convert("RGB"))
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
