"""Evaluate all trained classifiers on the held-out test split.

Writes:
    outputs/reports/classifier_comparison.csv
    outputs/reports/classifier_per_class.json
    outputs/figures/training/{model}_confusion.png
    outputs/figures/training/{model}_history.png
    outputs/figures/training/history_comparison.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, require_cuda
from src.data.dataset import build_loaders, build_splits
from src.evaluation.metrics import EvalResult, evaluate_classifier, results_to_dataframe
from src.evaluation.visualize import (
    load_history,
    plot_confusion_matrix,
    plot_training_history,
    plot_training_history_comparison,
)
from src.models.classifiers import build_model


MODEL_NAMES = ["cnn", "resnet18", "vit"]


def main() -> None:
    require_cuda()
    cfg = load_config()
    device = torch.device("cuda")

    splits = build_splits(cfg)
    loaders = build_loaders(splits, cfg)
    models_dir = Path(cfg["outputs"]["models"])
    figs_dir = Path(cfg["outputs"]["figures"]) / "training"
    reports_dir = Path(cfg["outputs"]["reports"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: List[EvalResult] = []
    histories: Dict[str, dict] = {}

    for name in MODEL_NAMES:
        ckpt = models_dir / f"{name}_best.pth"
        hist = models_dir / f"{name}_history.json"
        if not ckpt.is_file():
            print(f"[skip] {name} — no checkpoint at {ckpt}")
            continue
        model, _ = build_model(name, num_classes=len(splits.class_names))
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        res = evaluate_classifier(model, loaders["test"], device, splits.class_names, name)
        results.append(res)
        plot_confusion_matrix(
            res.cm, splits.class_names, f"{name} — held-out test", figs_dir / f"{name}_confusion.png"
        )
        if hist.is_file():
            h = load_history(hist)
            histories[name] = h
            plot_training_history(h, name, figs_dir / f"{name}_history.png")
        print(f"{name}: acc={res.accuracy:.4f} P={res.precision:.4f} R={res.recall:.4f} F1={res.f1:.4f}")

    if not results:
        print("No trained classifiers found. Run train_classifier.py first.")
        return

    df = results_to_dataframe(results).sort_values("f1_macro", ascending=False)
    df.to_csv(reports_dir / "classifier_comparison.csv", index=False)
    print("\nModel comparison:")
    print(df.to_string(index=False))

    per_class = {r.model_name: r.per_class for r in results}
    (reports_dir / "classifier_per_class.json").write_text(json.dumps(per_class, indent=2))

    if histories:
        plot_training_history_comparison(histories, figs_dir / "history_comparison.png")
        print(f"Wrote comparison plot -> {figs_dir / 'history_comparison.png'}")


if __name__ == "__main__":
    main()
