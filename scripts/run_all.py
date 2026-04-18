"""End-to-end driver: EDA -> train CNN/ResNet/ViT + YOLO -> evaluate -> Grad-CAM -> diagram.

Each step is guarded: missing data / missing packages are logged and the
run continues. A ``run_summary.json`` is written at the end listing what
succeeded, what failed, and where the outputs live.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def _run(label: str, fn: Callable[[], None], summary: Dict[str, dict]) -> None:
    t0 = time.time()
    print(f"\n==== {label} ====")
    try:
        fn()
        summary[label] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        summary[label] = {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[FAILED] {label}: {e}")


def _subprocess(script: str, *extra) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *extra]
    subprocess.run(cmd, check=True)


def main() -> None:
    cfg = load_config()
    summary: Dict[str, dict] = {}

    _run("1_eda", lambda: _subprocess("run_eda.py"), summary)
    _run("2a_prepare_yolo", lambda: _subprocess("prepare_yolo_data.py"), summary)
    _run("2b_train_cnn", lambda: _subprocess("train_classifier.py", "--model", "cnn"), summary)
    _run("2c_train_resnet18", lambda: _subprocess("train_classifier.py", "--model", "resnet18"), summary)
    _run("2d_train_vit", lambda: _subprocess("train_classifier.py", "--model", "vit"), summary)
    _run("2e_train_yolo", lambda: _subprocess("train_yolo.py"), summary)
    _run("3a_eval_classifiers", lambda: _subprocess("evaluate_classifiers.py"), summary)
    _run("3b_gradcam", lambda: _subprocess("run_gradcam.py"), summary)
    _run("3c_eval_yolo", lambda: _subprocess("evaluate_yolo.py"), summary)
    _run("4_diagram", lambda: _subprocess("make_architecture_diagram.py"), summary)

    out = Path(cfg["outputs"]["reports"]) / "run_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary -> {out}")


if __name__ == "__main__":
    main()
