"""Generic classifier training loop.

Writes best-weights and a history JSON (loss / acc per epoch) so downstream
evaluation can compare models without rerunning training.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss": self.train_loss, "train_acc": self.train_acc,
            "val_loss": self.val_loss, "val_acc": self.val_acc,
        }


def train_classifier(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_dir: Path,
    model_name: str,
) -> TrainHistory:
    """Train a classifier; keep best val accuracy weights.

    ``loaders`` must contain ``train`` and ``val``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = TrainHistory()
    sizes = {k: len(dl.dataset) for k, dl in loaders.items()}

    for epoch in range(num_epochs):
        t0 = time.time()
        for phase in ("train", "val"):
            model.train(mode=(phase == "train"))
            running_loss = 0.0
            running_correct = 0
            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]
            if phase == "train":
                history.train_loss.append(epoch_loss)
                history.train_acc.append(epoch_acc)
            else:
                history.val_loss.append(epoch_loss)
                history.val_acc.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(best_state, save_dir / f"{model_name}_best.pth")
        print(
            f"[{model_name}] epoch {epoch+1:02d}/{num_epochs} "
            f"train_loss={history.train_loss[-1]:.4f} train_acc={history.train_acc[-1]:.4f} "
            f"val_loss={history.val_loss[-1]:.4f} val_acc={history.val_acc[-1]:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

    model.load_state_dict(best_state)
    (save_dir / f"{model_name}_history.json").write_text(
        json.dumps({"best_val_acc": best_acc, **history.as_dict()}, indent=2)
    )
    print(f"[{model_name}] best val acc: {best_acc:.4f} — saved {save_dir / f'{model_name}_best.pth'}")
    return history
