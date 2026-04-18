"""Plotting helpers: confusion matrices, training curves, comparison charts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, class_names, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_history(history: Dict[str, list], title: str, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title(f"{title} — accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title(f"{title} — loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_history_comparison(histories: Dict[str, Dict[str, list]], out_path: Path) -> None:
    """All models overlaid on the same (loss, accuracy) figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["val_acc"], label=f"{name} (val)")
        axes[0].plot(epochs, h["train_acc"], "--", alpha=0.5)
        axes[1].plot(epochs, h["val_loss"], label=f"{name} (val)")
        axes[1].plot(epochs, h["train_loss"], "--", alpha=0.5)
    axes[0].set_title("Validation accuracy (solid) vs training (dashed)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].set_title("Validation loss (solid) vs training (dashed)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_history(path: Path) -> Dict[str, list]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
