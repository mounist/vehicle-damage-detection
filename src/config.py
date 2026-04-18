"""Config loader. Resolves relative paths against the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config and resolve path fields against the project root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_project_root"] = str(PROJECT_ROOT)

    def resolve(p: str) -> str:
        pp = Path(p)
        return str((PROJECT_ROOT / pp).resolve()) if not pp.is_absolute() else str(pp)

    cfg["classification"]["data_root"] = resolve(cfg["classification"]["data_root"])
    cfg["yolo"]["coco_root"] = resolve(cfg["yolo"]["coco_root"])
    cfg["yolo"]["yolo_root"] = resolve(cfg["yolo"]["yolo_root"])
    cfg["yolo"]["data_yaml"] = resolve(cfg["yolo"]["data_yaml"])

    for key in ("root", "models", "figures", "reports"):
        cfg["outputs"][key] = resolve(cfg["outputs"][key])

    return cfg


def require_cuda() -> None:
    """Assert CUDA is available; fail loudly with a helpful message."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This project requires a GPU. "
            "Install a CUDA-enabled PyTorch build and re-run."
        )
