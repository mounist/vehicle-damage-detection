"""One-shot helper: rewrite Windows backslash paths in outputs/reports/*.json to forward slashes.

Intended to be run once after a Windows training session. Idempotent — re-running is a no-op.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

BACKSLASH = chr(92)


def normalize(obj):
    if isinstance(obj, str):
        return obj.replace(BACKSLASH, "/") if BACKSLASH in obj else obj
    if isinstance(obj, list):
        return [normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    return obj


def main() -> None:
    cfg = load_config()
    reports = Path(cfg["outputs"]["reports"])
    changed = 0
    for p in sorted(reports.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        fixed = normalize(data)
        if json.dumps(fixed) != json.dumps(data):
            p.write_text(json.dumps(fixed, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"normalized {p.name}")
            changed += 1
        else:
            print(f"clean      {p.name}")
    print(f"total normalized: {changed}")


if __name__ == "__main__":
    main()
