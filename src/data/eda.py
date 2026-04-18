"""EDA for the binary classification and YOLO-segmentation datasets.

Everything here reads real images / labels off disk. No synthetic numbers.
Figures are saved to ``outputs/figures/eda/``.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from .transforms import build_transforms, denormalize

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Binary classification EDA
# ---------------------------------------------------------------------------

def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


def class_distribution(data_root: Path, splits=("training", "validation")) -> pd.DataFrame:
    """Count images per class per split for data1a."""
    rows = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                rows.append({"split": split, "class": cls_dir.name, "count": len(_list_images(cls_dir))})
    return pd.DataFrame(rows)


def plot_class_distribution(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="class", y="count", hue="split", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Image count")
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sample_grid(data_root: Path, split: str, out_path: Path, n_per_class: int = 4, seed: int = 42) -> None:
    """Grid of sample images (rows = classes, cols = samples)."""
    rng = random.Random(seed)
    split_dir = data_root / split
    classes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(3 * n_per_class, 3 * len(classes)))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
    for r, cls_dir in enumerate(classes):
        imgs = _list_images(cls_dir)
        picks = rng.sample(imgs, min(n_per_class, len(imgs)))
        for c, img_path in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(Image.open(img_path).convert("RGB"))
            ax.set_axis_off()
            if c == 0:
                ax.set_title(cls_dir.name, loc="left", fontsize=11)
    fig.suptitle(f"Sample images — {split}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def image_size_stats(data_root: Path, split: str, max_images: int = 1500) -> pd.DataFrame:
    """Sample image dimensions — catches tiny/huge/irregular inputs."""
    imgs = _list_images(data_root / split)
    if len(imgs) > max_images:
        imgs = random.Random(42).sample(imgs, max_images)
    rows = []
    for p in tqdm(imgs, desc=f"size {split}"):
        try:
            w, h = Image.open(p).size
            rows.append({"width": w, "height": h, "aspect": w / h})
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_image_size_distribution(df: pd.DataFrame, out_path: Path, split: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df["width"], bins=40, color="#4C72B0")
    axes[0].set_title("Width (px)")
    axes[1].hist(df["height"], bins=40, color="#55A868")
    axes[1].set_title("Height (px)")
    axes[2].hist(df["aspect"], bins=40, color="#C44E52")
    axes[2].set_title("Aspect ratio (W/H)")
    for ax in axes:
        ax.set_ylabel("count")
    fig.suptitle(f"Image size distribution — {split}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def augmentation_preview(data_root: Path, split: str, out_path: Path, image_size: int = 224, n: int = 4, seed: int = 42) -> None:
    """Show the same image before/after the training augmentation pipeline."""
    rng = random.Random(seed)
    t = build_transforms(image_size)
    classes = sorted([d for d in (data_root / split).iterdir() if d.is_dir()])
    picks: List[Path] = []
    for cls_dir in classes:
        imgs = _list_images(cls_dir)
        if imgs:
            picks.append(rng.choice(imgs))
    picks = picks[:n]

    fig, axes = plt.subplots(len(picks), 5, figsize=(14, 3 * len(picks)))
    if len(picks) == 1:
        axes = np.expand_dims(axes, 0)
    for r, img_path in enumerate(picks):
        pil = Image.open(img_path).convert("RGB")
        axes[r, 0].imshow(pil)
        axes[r, 0].set_title("original")
        axes[r, 0].set_axis_off()
        for c in range(1, 5):
            aug = t["training"](pil)
            axes[r, c].imshow(denormalize(aug))
            axes[r, c].set_title(f"aug {c}")
            axes[r, c].set_axis_off()
    fig.suptitle("Training augmentation samples (RandomResizedCrop + Rotation + ColorJitter + HFlip)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# YOLO segmentation EDA (reads COCO JSON directly)
# ---------------------------------------------------------------------------

def coco_class_distribution(coco_json: Path) -> pd.DataFrame:
    """Per-class instance counts in one COCO annotation file."""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    counts = Counter(cat_by_id[a["category_id"]] for a in data["annotations"])
    return pd.DataFrame([{"class": k, "count": v} for k, v in counts.items()]).sort_values("count", ascending=False).reset_index(drop=True)


def plot_yolo_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, out_path: Path) -> None:
    merged = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=merged, x="class", y="count", hue="split", ax=ax)
    ax.set_title("CarDD — damage class distribution")
    ax.set_ylabel("Instance count")
    for container in ax.containers:
        ax.bar_label(container, padding=2, fontsize=9)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def coco_sample_grid(coco_json: Path, image_dir: Path, out_path: Path, n: int = 6, seed: int = 42) -> None:
    """Overlay segmentation polygons on random images — quick sanity check."""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    imgs = data["images"]
    anns_by_img: Dict[int, list] = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    rng = random.Random(seed)
    picks = rng.sample([im for im in imgs if im["id"] in anns_by_img], min(n, len(imgs)))

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    for idx, im in enumerate(picks):
        ax = axes[idx // cols, idx % cols]
        img_path = image_dir / im["file_name"]
        if not img_path.exists():
            ax.set_axis_off()
            continue
        ax.imshow(Image.open(img_path).convert("RGB"))
        labels = []
        for a in anns_by_img[im["id"]]:
            seg = a.get("segmentation", [])
            labels.append(cat_by_id[a["category_id"]])
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) >= 6:
                        xs = poly[0::2]
                        ys = poly[1::2]
                        ax.fill(xs, ys, alpha=0.35)
        ax.set_title(", ".join(sorted(set(labels))), fontsize=9)
        ax.set_axis_off()
    for k in range(len(picks), rows * cols):
        axes[k // cols, k % cols].set_axis_off()
    fig.suptitle("CarDD — sample images with damage polygons")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def imbalance_report(df: pd.DataFrame) -> Dict[str, float]:
    """Summarize class imbalance — used to motivate Focal Loss in the README."""
    counts = df["count"].astype(float)
    return {
        "min": float(counts.min()),
        "max": float(counts.max()),
        "ratio_max_over_min": float(counts.max() / max(counts.min(), 1.0)),
        "entropy": float(-(counts / counts.sum() * np.log(counts / counts.sum())).sum()),
    }
