"""Convert COCO-format polygon annotations into YOLOv8-seg text labels."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm


def coco_to_yolo_labels(coco_json_path: Path, out_labels_dir: Path) -> Dict[int, str]:
    """Emit one ``<stem>.txt`` per image with normalized polygon coords.

    Returns the 0-indexed ``{class_id: name}`` mapping used by the YAML file.
    """
    coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))
    cats = sorted(coco["categories"], key=lambda x: x["id"])
    cat2idx = {c["id"]: i for i, c in enumerate(cats)}
    names = {i: c["name"] for i, c in enumerate(cats)}

    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    out_labels_dir.mkdir(parents=True, exist_ok=True)
    for im in tqdm(coco["images"], desc=f"COCO->YOLO {coco_json_path.stem}"):
        w, h = im["width"], im["height"]
        file_stem = Path(im["file_name"]).stem
        lines = []
        for ann in ann_by_img[im["id"]]:
            cls = cat2idx.get(ann["category_id"])
            if cls is None:
                continue
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if len(poly) < 6:
                    continue
                coords = []
                for i in range(0, len(poly), 2):
                    coords.append(f"{poly[i]/w:.6f}")
                    coords.append(f"{poly[i+1]/h:.6f}")
                lines.append(f"{cls} " + " ".join(coords))
        (out_labels_dir / f"{file_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return names


def copy_images(src_dir: Path, dst_dir: Path) -> int:
    """Copy images from src to dst, skipping files that already exist. Returns count copied."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob("*"):
        if not p.is_file():
            continue
        target = dst_dir / p.name
        if not target.exists():
            shutil.copy2(p, target)
            n += 1
    return n


def build_yolo_yaml(yolo_root: Path, names: Dict[int, str]) -> Path:
    """Write the ``cardd_LOCAL.yaml`` the YOLO CLI consumes."""
    out = yolo_root / "cardd_LOCAL.yaml"
    out.write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": "images/train",
                "val": "images/val",
                "names": {int(k): str(v) for k, v in sorted(names.items())},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return out
