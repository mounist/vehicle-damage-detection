"""Classification metrics on the held-out test set."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cm: np.ndarray
    per_class: Dict[str, Dict[str, float]]
    preds: np.ndarray
    labels: np.ndarray


def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    model_name: str,
) -> EvalResult:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            all_preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    acc = float((preds == labels).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    per_class = {c: {k: float(v) for k, v in report[c].items()} for c in class_names}
    return EvalResult(
        model_name=model_name,
        accuracy=acc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        cm=confusion_matrix(labels, preds),
        per_class=per_class,
        preds=preds,
        labels=labels,
    )


def results_to_dataframe(results: List[EvalResult]):
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "accuracy": r.accuracy,
            "precision_macro": r.precision,
            "recall_macro": r.recall,
            "f1_macro": r.f1,
        })
    return pd.DataFrame(rows)
