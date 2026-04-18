"""Render a matplotlib architecture diagram of the two-stage pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def main() -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    cfg = load_config()
    out_dir = Path(cfg["outputs"]["figures"]) / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "architecture.png"

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.set_axis_off()

    def block(x, y, w, h, text, color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", linewidth=1.2,
                                            edgecolor="black", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", fontsize=9, style="italic")

    # Input
    block(0.2, 2.1, 1.9, 1.0, "Input image\n(RGB)", "#e8f0ff")

    # Stage 1
    block(2.8, 3.0, 2.6, 1.2, "Stage 1\nBinary classifier\n(ResNet-18 / ViT)", "#d1e7dd")
    arrow(2.1, 2.6, 2.8, 3.6, "224x224")

    # Decision diamond
    ax.add_patch(patches.Polygon([(7.0, 3.6), (7.9, 4.1), (8.8, 3.6), (7.9, 3.1)],
                                 closed=True, edgecolor="black", facecolor="#fff3cd"))
    ax.text(7.9, 3.6, "damaged\nprob >= tau ?", ha="center", va="center", fontsize=9)
    arrow(5.4, 3.6, 7.0, 3.6)

    # No branch
    block(9.3, 4.1, 2.5, 0.9, "NO DAMAGE\n(skip YOLO)", "#f8d7da")
    arrow(8.0, 4.1, 9.3, 4.55, "no")

    # Stage 2
    block(2.8, 0.4, 2.6, 1.2, "Stage 2\nYOLOv8s-seg\n(6 damage classes)", "#cfe2ff")
    arrow(7.9, 3.1, 7.9, 1.9, "yes")
    arrow(7.9, 1.9, 5.4, 1.0)

    # Output JSON
    block(6.0, 0.4, 2.8, 1.2, "Mask + class + conf\n+ JSON report", "#fff3cd")
    arrow(5.4, 1.0, 6.0, 1.0)

    # Annotated image
    block(9.3, 0.4, 2.5, 1.2, "Annotated image\n(outputs/)", "#e8f0ff")
    arrow(8.8, 1.0, 9.3, 1.0)

    ax.set_title("Two-stage vehicle damage analysis pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Diagram saved: {out_path}")


if __name__ == "__main__":
    main()
